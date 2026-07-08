"""Time-based splitting. [HARD] Splits are by DATE only — never by shuffled row
index, never KFold. Expanding-window CV for tuning; a single final holdout
(last 6 months of 2017) scored exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .utils import load_config


@dataclass
class Fold:
    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def cv_folds(df: pd.DataFrame, cfg: dict | None = None) -> list[Fold]:
    """Expanding-window folds from config date cutoffs.

    Fold k: train = {date <= train_end}, val = {train_end < date <= val_end}.
    Each successive fold's training window strictly expands.
    """
    cfg = cfg or load_config()
    date_c = cfg["data"]["date_col"]
    dates = pd.to_datetime(df[date_c])
    folds: list[Fold] = []
    for i, f in enumerate(cfg["splits"]["cv_folds"], start=1):
        train_end = pd.Timestamp(f["train_end"])
        val_end = pd.Timestamp(f["val_end"])
        train_idx = np.flatnonzero((dates <= train_end).to_numpy())
        val_idx = np.flatnonzero(((dates > train_end) & (dates <= val_end)).to_numpy())
        folds.append(Fold(name=f"fold{i}", train_idx=train_idx, val_idx=val_idx))
    return folds


def holdout_split(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, holdout_df). Holdout = date >= holdout_start; untouched
    during all CV/tuning and scored once."""
    cfg = cfg or load_config()
    date_c = cfg["data"]["date_col"]
    start = pd.Timestamp(cfg["splits"]["holdout_start"])
    dates = pd.to_datetime(df[date_c])
    train_df = df[dates < start].reset_index(drop=True)
    holdout_df = df[dates >= start].reset_index(drop=True)
    return train_df, holdout_df


def scale_train_apply(
    train: pd.DataFrame, other: pd.DataFrame, numeric_cols: list[str]
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """[HARD] Fit StandardScaler on the TRAIN portion only, then transform both.

    Returns (train_scaled, other_scaled, fitted_scaler).
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train[numeric_cols].to_numpy(dtype="float64"))
    other_scaled = scaler.transform(other[numeric_cols].to_numpy(dtype="float64"))
    return train_scaled, other_scaled, scaler
