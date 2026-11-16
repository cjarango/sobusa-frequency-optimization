"""
walk_forward
============

Walk-forward validation with Optuna for SOBUSA demand models.

The model receives a tabular feature matrix ``X`` built once per split
with ``feature_builder``. Lags are already columns (``lag_32``,
``lag_33``, ``lag_65``, ``lag_455``), so no sliding window is needed.

The ``window`` hyperparameter controls how many rows of history the
training set of each fold uses. ``None`` (or a very large value) means
expanding window. The validation block of each fold is fixed regardless
of ``window``, so trials remain comparable on the same hold-out slices.
"""

import functools
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from eval.scorer import DemandScorer
from models.base import BaseDemandModel
from utils.logger import setup_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

__all__ = ["WalkForwardTrainer"]


def _early_stopping_callback(
    study: optuna.Study, trial: optuna.trial.FrozenTrial, patience: int
) -> None:
    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) < 2:
        return
    if (trial.number - study.best_trial.number) >= patience:
        study.stop()


class WalkForwardTrainer:
    """Expanding or sliding walk-forward with Optuna delegating rolling refits to the model."""

    def __init__(
        self,
        model_class: type[BaseDemandModel],
        scorer: DemandScorer,
        output_dir: str | Path,
        feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
        horizon_slots: int = 1,
        n_splits: int = 5,
        warmup: int = 1200,
        embargo: int | None = None,
        n_trials: int = 30,
        suggest_params_fn: Callable[[optuna.Trial], dict] | None = None,
        early_stop_rounds: int | None = None,
        inner_val_frac: float = 0.1,
        model_kwargs: dict | None = None,
    ) -> None:
        if horizon_slots < 1:
            raise ValueError(f"horizon_slots must be >= 1, got {horizon_slots}.")
        if embargo is None:
            embargo = max(1, horizon_slots - 1)
        if embargo < max(1, horizon_slots - 1):
            raise ValueError(
                f"Embargo must be >= {horizon_slots - 1} for horizon "
                f"{horizon_slots}, got {embargo}."
            )
        if not 0.0 <= inner_val_frac < 1.0:
            raise ValueError(
                f"inner_val_frac must be in [0, 1), got {inner_val_frac}."
            )

        self.model_class = model_class
        self.scorer = scorer
        self.output_dir = Path(output_dir)
        self.feature_builder = feature_builder
        self.horizon_slots = horizon_slots
        self.n_splits = n_splits
        self.warmup = warmup
        self.embargo = embargo
        self.n_trials = n_trials
        self.suggest_params_fn = suggest_params_fn
        self.early_stop_rounds = early_stop_rounds
        self.inner_val_frac = inner_val_frac
        self.model_kwargs = dict(model_kwargs or {})

        self.logger = setup_logger(f"Trainer_{model_class.__name__}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / f"{self.model_class.__name__}_results.csv"
        self.predictions_path = (
            self.output_dir / f"{self.model_class.__name__}_predictions.csv"
        )

    def _build_dataset(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
        h = self.horizon_slots
        pasajeros = df["pasajeros"]

        y_all = pasajeros.rolling(h).sum().shift(-(h - 1))
        y_naive_all = pasajeros.rolling(h).sum().shift(455 - h + 1)

        features = self.feature_builder(df)

        y = y_all.reindex(features.index)
        y_naive = y_naive_all.reindex(features.index)
        valid = y.notna() & y_naive.notna()

        feature_names = [
            c for c in features.columns if c not in {"franja", "pasajeros"}
        ]
        X = features.loc[valid, feature_names].copy()
        y = y.loc[valid].to_numpy(dtype=np.float64)
        y_naive = y_naive.loc[valid].to_numpy(dtype=np.float64)

        if X.isna().any().any():
            self.logger.warning(
                "X contains NaN values after alignment; LightGBM will "
                "treat them as missing."
            )

        if len(X) <= self.warmup:
            raise ValueError(
                f"Not enough valid rows ({len(X)}) for warmup ({self.warmup})."
            )

        self.logger.info(
            f"Modelable dataset: {len(X)} rows, {len(feature_names)} "
            f"features (h={h} slots, embargo={self.embargo})."
        )
        return X, y, y_naive, feature_names

    def _build_model(self, params: dict) -> BaseDemandModel:
        model_params = {k: v for k, v in params.items() if k != "window"}
        effective = {
            "inner_val_frac": self.inner_val_frac,
            **model_params,
            **self.model_kwargs,
        }
        return self.model_class(**effective)

    def _filter_by_params(self, df: pd.DataFrame, params: dict) -> pd.DataFrame:
        for key, value in params.items():
            if key not in df.columns:
                return df.iloc[0:0]
            if isinstance(value, float):
                df = df[
                    np.isclose(
                        df[key].astype(float).to_numpy(),
                        value,
                        rtol=1e-9,
                        atol=1e-12,
                    )
                ]
            else:
                df = df[df[key] == value]
        return df

    def _load_completed_folds(self, params: dict) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return pd.DataFrame()
            df = self._filter_by_params(df, params)
            return df.sort_values("fold").drop_duplicates(subset=["fold"], keep="last")
        except Exception:
            self.logger.warning(
                f"Could not load completed folds from {self.csv_path}.",
                exc_info=True,
            )
            return pd.DataFrame()

    def objective(
        self,
        trial: optuna.Trial,
        X: pd.DataFrame,
        y: np.ndarray,
        y_naive: np.ndarray,
    ) -> float:
        if self.suggest_params_fn is not None:
            params = self.suggest_params_fn(trial)
        else:
            params = self.model_class.suggest_params(trial)

        window = params.get("window", None)
        df_prev = self._load_completed_folds(params)
        completed_folds = int(df_prev["fold"].max()) if not df_prev.empty else 0

        p = float(
            self.model_kwargs.get(
                "tweedie_variance_power",
                params.get("tweedie_variance_power", 1.5),
            )
        )

        if completed_folds >= self.n_splits:
            trial.set_user_attr(
                "train_time_sec", round(float(df_prev["train_time_sec"].sum()), 2)
            )
            trial.set_user_attr("converged", bool(df_prev["converged"].all()))
            return float(df_prev["tweedie_deviance"].mean())

        fold_scores: list[float] = (
            df_prev["tweedie_deviance"].astype(float).tolist()
            if not df_prev.empty
            else []
        )
        all_converged = bool(df_prev["converged"].all()) if not df_prev.empty else True
        total_train_time = (
            float(df_prev["train_time_sec"].sum()) if not df_prev.empty else 0.0
        )

        model = self._build_model(params)
        model.fit(X.iloc[: self.warmup], y[: self.warmup])

        X_eval = X.iloc[self.warmup :]
        y_eval = y[self.warmup :]
        y_naive_eval = y_naive[self.warmup :]

        def _on_fold_end(fold_info: dict) -> None:
            nonlocal all_converged, total_train_time
            f_idx = fold_info["fold"]
            s_local = fold_info["val_start_idx"] - self.warmup
            e_local = fold_info["val_end_idx"] - self.warmup
            val_ts = X_eval.index[s_local:e_local]

            total_train_time += fold_info["train_time_sec"]
            converged = bool(fold_info["converged"])

            if not converged or fold_info["preds"] is None:
                all_converged = False
                metrics = {
                    "tweedie_deviance": 1e10,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "bias": np.nan,
                    "smape": np.nan,
                    "mase": np.nan,
                }
            else:
                metrics = self.scorer.calculate_all(
                    fold_info["y_true"],
                    fold_info["preds"],
                    y_naive_eval[s_local:e_local],
                    p=p,
                )

            score = float(metrics["tweedie_deviance"])
            fold_scores.append(score)

            row = {
                **params,
                "trial": trial.number,
                "fold": f_idx,
                "train_start_idx": fold_info["train_start_idx"],
                "train_end_idx": fold_info["train_end_idx"],
                "val_start_idx": fold_info["val_start_idx"],
                "val_end_idx": fold_info["val_end_idx"],
                "val_start_ts": str(val_ts[0]),
                "val_end_ts": str(val_ts[-1]),
                "train_size": fold_info["train_size"],
                "val_size": fold_info["val_size"],
                "converged": converged,
                "best_iteration": fold_info["best_iteration"],
                "train_time_sec": fold_info["train_time_sec"],
                "tweedie_deviance": metrics["tweedie_deviance"],
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "bias": metrics["bias"],
                "smape": metrics["smape"],
                "mase": metrics["mase"],
                "tweedie_p": p,
            }
            pd.DataFrame([row]).to_csv(
                self.csv_path,
                mode="a",
                header=not self.csv_path.exists(),
                index=False,
            )

            trial.report(float(np.mean(fold_scores)), f_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        try:
            model.forecast(
                X_eval,
                y_eval,
                window=window,
                embargo=self.embargo,
                n_splits=self.n_splits,
                start_fold=completed_folds,
                on_fold_end=_on_fold_end,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            self.logger.error(
                f"Rolling CV failed with {params}: {e}",
                exc_info=True,
            )
            all_converged = False
            fold_scores.append(1e10)

        trial.set_user_attr("converged", all_converged)
        trial.set_user_attr("train_time_sec", round(total_train_time, 2))
        return float(np.mean(fold_scores))

    def run(self, df: pd.DataFrame) -> optuna.Study:
        X, y, y_naive, _ = self._build_dataset(df)

        study_name = f"{self.model_class.__name__}_h{self.horizon_slots}_study"
        storage_url = f"sqlite:///{self.output_dir.resolve()}/optuna_study.db"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5, n_warmup_steps=2
            ),
        )

        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        remaining = max(0, self.n_trials - len(completed_trials))
        self.logger.info(
            f"Study '{study_name}': {len(completed_trials)} completed trials, "
            f"{remaining} remaining."
        )

        if remaining > 0:
            callbacks = []
            if self.early_stop_rounds is not None:
                callbacks.append(
                    functools.partial(
                        _early_stopping_callback,
                        patience=self.early_stop_rounds,
                    )
                )
            study.optimize(
                lambda t: self.objective(t, X, y, y_naive),
                n_trials=remaining,
                catch=(Exception,),
                show_progress_bar=True,
                callbacks=callbacks,
            )

        valid_trials = [
            t
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if not valid_trials:
            self.logger.error(
                f"[{self.model_class.__name__}] No completed trials. "
                f"Check {self.csv_path}."
            )
            return study

        best = study.best_trial
        self.logger.warning(
            f"[{self.model_class.__name__}] Best Tweedie deviance: "
            f"{study.best_value:.4f} | Params: {study.best_params}"
        )

        self._save_best_predictions(best.params, X, y, y_naive)
        return study

    def _save_best_predictions(
        self,
        params: dict,
        X: pd.DataFrame,
        y: np.ndarray,
        y_naive: np.ndarray,
    ) -> None:
        window = params.get("window", None)
        model = self._build_model(params)
        model.fit(X.iloc[: self.warmup], y[: self.warmup])

        X_eval = X.iloc[self.warmup :]
        y_eval = y[self.warmup :]
        y_naive_eval = y_naive[self.warmup :]

        try:
            model.forecast(
                X_eval,
                y_eval,
                window=window,
                embargo=self.embargo,
                n_splits=self.n_splits,
            )
        except Exception as e:
            self.logger.error(
                f"Best-trial rolling forecast failed: {e}",
                exc_info=True,
            )
            return

        rows = []
        for f in model.rolling_folds_:
            s_local = f["val_start_idx"] - self.warmup
            e_local = f["val_end_idx"] - self.warmup
            rows.append(
                pd.DataFrame(
                    {
                        "fold": f["fold"],
                        "val_start_idx": f["val_start_idx"],
                        "val_end_idx": f["val_end_idx"],
                        "franja_ts": X_eval.index[s_local:e_local],
                        "y_true": f["y_true"],
                        "y_pred": f["preds"],
                        "y_naive": y_naive_eval[s_local:e_local],
                    }
                )
            )

        if not rows:
            self.logger.warning("No best-trial predictions to save.")
            return

        pd.concat(rows, ignore_index=True).to_csv(
            self.predictions_path, index=False
        )
        self.logger.info(
            f"Saved best-trial predictions to {self.predictions_path}"
        )