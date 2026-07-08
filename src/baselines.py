"""Non-ML baselines. These are not optional — [HARD] they sit in the final
results table next to LinearRegression and XGBoost. If XGBoost cannot beat
seasonal-naive, run.py reports that honestly.

All four are pure functions of already-computed, leakage-safe feature columns
(each is a shifted view of the true series), so they are automatically
1-step-ahead walk-forward with true history, exactly like the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive(df: pd.DataFrame) -> np.ndarray:
    """Predict yesterday's value: sales[d-1]."""
    return df["lag_1"].to_numpy(dtype="float64")


def seasonal_naive_7(df: pd.DataFrame) -> np.ndarray:
    """Predict the value one week ago: sales[d-7]."""
    return df["lag_7"].to_numpy(dtype="float64")


def seasonal_naive_365(df: pd.DataFrame) -> np.ndarray:
    """Predict the value one year ago: sales[d-365]."""
    return df["lag_365"].to_numpy(dtype="float64")


def moving_average_28(df: pd.DataFrame) -> np.ndarray:
    """Predict the trailing 28-day mean (shift(1).rolling(28).mean)."""
    return df["roll_mean_28"].to_numpy(dtype="float64")


# Order preserved for the results table.
BASELINES = {
    "naive": naive,
    "seasonal_naive_7": seasonal_naive_7,
    "seasonal_naive_365": seasonal_naive_365,
    "moving_avg_28": moving_average_28,
}


def predict_all(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return {baseline_name: predictions} for every row of df."""
    return {name: fn(df) for name, fn in BASELINES.items()}
