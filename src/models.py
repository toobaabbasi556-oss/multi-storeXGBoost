"""Models: LinearRegression (scaled) and XGBoost (CV-tuned).

[HARD] Hyperparameter tuning uses ONLY the expanding-window time-series CV built
from the training portion; the final holdout is never seen here. Any scaler is
fit on the training fold only.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .evaluate import wape
from .features import FeatureInfo
from .splits import Fold
from .utils import SEED


# --------------------------------------------------------------------------
# Linear Regression
# --------------------------------------------------------------------------
class LinearModel:
    """LinearRegression on standardized numeric features (no categorical ids;
    the lag features already carry each series' level). Scaler fit on train only.
    Predictions clipped at 0 (sales are non-negative)."""

    def __init__(self, info: FeatureInfo):
        self.info = info
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def fit(self, train: pd.DataFrame) -> "LinearModel":
        X = self.scaler.fit_transform(train[self.info.numeric].to_numpy("float64"))
        self.model.fit(X, train[self.info.target].to_numpy("float64"))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.info.numeric].to_numpy("float64"))
        return np.clip(self.model.predict(X), 0, None)


# --------------------------------------------------------------------------
# XGBoost
# --------------------------------------------------------------------------
def _xgb(params: dict) -> XGBRegressor:
    return XGBRegressor(
        enable_categorical=True,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
        objective="reg:squarederror",
        **params,
    )


class XGBModel:
    def __init__(self, info: FeatureInfo, params: dict):
        self.info = info
        self.params = params
        self.model = _xgb(params)

    def fit(self, train: pd.DataFrame) -> "XGBModel":
        self.model.fit(train[self.info.all_features], train[self.info.target])
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.clip(self.model.predict(df[self.info.all_features]), 0, None)


@dataclass
class TuningResult:
    best_params: dict
    best_wape: float
    table: list[dict]           # every combo's mean CV WAPE, for the report


def _grid(param_grid: dict) -> list[dict]:
    keys = list(param_grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*param_grid.values())]


def tune_xgboost(
    train: pd.DataFrame, folds: list[Fold], info: FeatureInfo, cfg: dict
) -> TuningResult:
    """Grid search over the time-series CV folds, scored by WAPE. Holdout untouched."""
    grid = _grid(cfg["model"]["xgboost"]["param_grid"])
    target = info.target
    table: list[dict] = []
    best = None
    for params in grid:
        fold_wapes = []
        for f in folds:
            tr = train.iloc[f.train_idx]
            va = train.iloc[f.val_idx]
            m = _xgb(params).fit(tr[info.all_features], tr[target])
            pred = np.clip(m.predict(va[info.all_features]), 0, None)
            fold_wapes.append(wape(va[target].to_numpy("float64"), pred))
        mean_wape = float(np.mean(fold_wapes))
        table.append({**params, "cv_wape": mean_wape})
        if best is None or mean_wape < best[1]:
            best = (params, mean_wape)
    return TuningResult(best_params=best[0], best_wape=best[1], table=table)
