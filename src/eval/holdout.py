"""
holdout
=======

Out-of-sample evaluation of the best trial on the untouched holdout.

The holdout is processed with the same ``feature_builder`` as the train
split, but never concatenated with it. The implicit embargo at the
train/holdout boundary is provided by the NaN tail of the target
(``rolling(h).sum().shift(-(h-1))``), so no explicit trim is required
as long as the two splits are built independently.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from eval.scorer import DemandScorer
from models.base import BaseDemandModel

__all__ = ["build_holdout_dataset", "evaluate_on_holdout"]


def build_holdout_dataset(
    df_holdout: pd.DataFrame,
    feature_builder: Callable[[pd.DataFrame], pd.DataFrame],
    horizon_slots: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    h = horizon_slots
    pasajeros = df_holdout["pasajeros"]

    y_all = pasajeros.rolling(h).sum().shift(-(h - 1))
    y_naive_all = pasajeros.rolling(h).sum().shift(455 - h + 1)

    features = feature_builder(df_holdout)

    y = y_all.reindex(features.index)
    y_naive = y_naive_all.reindex(features.index)
    valid = y.notna() & y_naive.notna()

    feature_names = [
        c for c in features.columns if c not in {"franja", "pasajeros"}
    ]
    X = features.loc[valid, feature_names].copy()
    y = y.loc[valid].to_numpy(dtype=np.float64)
    y_naive = y_naive.loc[valid].to_numpy(dtype=np.float64)
    return X, y, y_naive, feature_names


def evaluate_on_holdout(
    model: BaseDemandModel,
    X_hold: pd.DataFrame,
    y_hold: np.ndarray,
    y_naive_hold: np.ndarray,
    scorer: DemandScorer,
    p: float,
) -> dict:
    preds = model.forecast(X_hold, y_hold)
    return scorer.calculate_all(y_hold, preds, y_naive_hold, p=p)