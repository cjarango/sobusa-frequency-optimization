"""
scorer
======

Error metrics for passenger-demand forecasts.

Primary metric aligns with the model loss (Tweedie deviance). MASE
is reported against a seasonal naive baseline so that performance is
comparable across horizons and interpretable for stakeholders.
"""

import numpy as np

__all__ = ["DemandScorer"]


class DemandScorer:
    """Calculates error metrics for non-negative demand forecasts."""

    def __init__(self) -> None:
        pass

    def _validate_inputs(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}."
            )
        if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
            raise ValueError("Inputs contain NaN or Inf values.")
        if np.any(y_true < 0):
            raise ValueError("y_true contains negative demand values.")
        if np.any(y_pred < 0):
            raise ValueError("y_pred contains negative demand values.")

    def mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(self.mse(y_true, y_pred)))

    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    def bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean signed error (y_pred - y_true). Positive = overprediction."""
        return float(np.mean(y_pred - y_true))

    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric MAPE on [0, 2]. Robust to zeros."""
        self._validate_inputs(y_true, y_pred)
        denom = np.abs(y_true) + np.abs(y_pred)
        mask = denom > 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(2.0 * np.abs(y_pred[mask] - y_true[mask]) / denom[mask]))

    def tweedie_deviance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p: float = 1.5,
        cap: float = 100.0,
    ) -> float:
        """Mean Tweedie unit deviance for 1 < p < 2.

        Matches the LightGBM ``metric='tweedie'`` definition. Per-observation
        deviances are clipped at ``cap`` before averaging to protect the
        mean from a handful of near-zero predictions on high-demand slots.

        Args:
            y_true (np.ndarray): Observed demand.
            y_pred (np.ndarray): Predicted mean demand.
            p (float): Tweedie variance power, in (1.0, 2.0).
            cap (float): Per-observation deviance cap. Defaults to 100.0.

        Returns:
            float: Mean capped Tweedie deviance.
        """
        if not 1.0 < p < 2.0:
            raise ValueError(
                f"'p' must be in the open interval (1.0, 2.0). Got {p}."
            )
        if cap <= 0:
            raise ValueError(f"'cap' must be > 0. Got {cap}.")
        self._validate_inputs(y_true, y_pred)

        eps = 1e-12
        mu = np.clip(y_pred, eps, None)
        y = np.asarray(y_true, dtype=np.float64)

        term_y = np.where(
            y > 0, np.power(y, 2.0 - p) / ((1.0 - p) * (2.0 - p)), 0.0
        )
        term_cross = y * np.power(mu, 1.0 - p) / (1.0 - p)
        term_mu = np.power(mu, 2.0 - p) / (2.0 - p)

        deviance = 2.0 * (term_y - term_cross + term_mu)
        deviance = np.minimum(deviance, cap)
        return float(np.mean(deviance))

    def mase(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_naive: np.ndarray,
    ) -> float:
        """Mean Absolute Scaled Error against a naive baseline.

        ``y_naive`` must contain the baseline forecast aligned with
        ``y_true``. MASE < 1 means the model beats the baseline.
        """
        self._validate_inputs(y_true, y_pred)
        self._validate_inputs(y_true, y_naive)
        baseline_mae = float(np.mean(np.abs(y_true - y_naive)))
        if baseline_mae == 0.0:
            return float("nan")
        return self.mae(y_true, y_pred) / baseline_mae

    def pinball_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: float = 0.5,
    ) -> float:
        if not 0.0 < tau < 1.0:
            raise ValueError(f"'tau' must be in (0, 1). Got {tau}.")
        self._validate_inputs(y_true, y_pred)
        diff = y_true - y_pred
        loss = np.maximum(tau * diff, (tau - 1.0) * diff)
        return float(np.mean(loss))

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_naive: np.ndarray,
        p: float = 1.5,
        cap: float = 100.0,
    ) -> dict:
        self._validate_inputs(y_true, y_pred)
        return {
            "tweedie_deviance": self.tweedie_deviance(y_true, y_pred, p=p, cap=cap),
            "mae": self.mae(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mse": self.mse(y_true, y_pred),
            "bias": self.bias(y_true, y_pred),
            "smape": self.smape(y_true, y_pred),
            "mase": self.mase(y_true, y_pred, y_naive),
            "pinball_50": self.pinball_loss(y_true, y_pred, tau=0.5),
        }