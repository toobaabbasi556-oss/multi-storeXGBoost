"""Leakage-safe feature engineering.

[HARD] Every time-derived feature is computed group-wise per (store, item) on
date-sorted data using `.shift()`, so a row never sees its own or any future
`sales`. Rolling/expanding stats are built on the `.shift(1)` series, never the
raw target. The leakage unit test (src/test_leakage.py) enforces this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import load_config


@dataclass
class FeatureInfo:
    """Column bookkeeping so downstream code never hardcodes feature names."""
    numeric: list[str] = field(default_factory=list)   # scaled for LinearRegression
    categorical: list[str] = field(default_factory=list)  # store, item for XGBoost
    target: str = "sales"

    @property
    def all_features(self) -> list[str]:
        return self.numeric + self.categorical


def build_features(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, FeatureInfo]:
    """Return (feature frame, FeatureInfo). Rows with NaN lags are dropped."""
    cfg = cfg or load_config()
    date_c = cfg["data"]["date_col"]
    store_c = cfg["data"]["store_col"]
    item_c = cfg["data"]["item_col"]
    target_c = cfg["data"]["target_col"]
    lags = cfg["features"]["lags"]
    windows = cfg["features"]["rolling_windows"]

    df = df.sort_values([store_c, item_c, date_c]).reset_index(drop=True)
    grp = df.groupby([store_c, item_c], observed=True)[target_c]
    numeric: list[str] = []

    # --- Lags: shift(k) of the true series, within group -------------------
    for k in lags:
        col = f"lag_{k}"
        df[col] = grp.shift(k)
        numeric.append(col)

    # --- Rolling stats on the shift(1) series (never the current value) -----
    base = grp.shift(1)                      # yesterday-and-earlier, per series
    df["_base"] = base
    base_grp = df.groupby([store_c, item_c], observed=True)["_base"]
    for w in windows:
        df[f"roll_mean_{w}"] = base_grp.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"roll_std_{w}"] = base_grp.transform(lambda s, w=w: s.rolling(w).std())
        df[f"roll_min_{w}"] = base_grp.transform(lambda s, w=w: s.rolling(w).min())
        df[f"roll_max_{w}"] = base_grp.transform(lambda s, w=w: s.rolling(w).max())
        numeric += [f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}", f"roll_max_{w}"]

    # --- Expanding mean on the shift(1) series ------------------------------
    df["expanding_mean"] = base_grp.transform(lambda s: s.expanding().mean())
    numeric.append("expanding_mean")
    df = df.drop(columns=["_base"])

    # --- Calendar features (no leakage: derived from the date only) ---------
    d = df[date_c].dt
    df["year"] = d.year
    df["month"] = d.month
    df["day"] = d.day
    df["dayofweek"] = d.dayofweek
    df["weekofyear"] = d.isocalendar().week.astype("int32")
    df["is_weekend"] = (d.dayofweek >= 5).astype("int8")
    df["is_month_start"] = d.is_month_start.astype("int8")
    df["is_month_end"] = d.is_month_end.astype("int8")
    numeric += ["year", "month", "day", "dayofweek", "weekofyear",
                "is_weekend", "is_month_start", "is_month_end"]

    # --- Cyclical encodings for month and dayofweek -------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    numeric += ["month_sin", "month_cos", "dow_sin", "dow_cos"]

    # --- store & item kept as categoricals for XGBoost ----------------------
    df[store_c] = df[store_c].astype("category")
    df[item_c] = df[item_c].astype("category")

    # --- Drop rows with NaN lags (series start), AFTER building features -----
    lag_cols = [f"lag_{k}" for k in lags]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    info = FeatureInfo(numeric=numeric, categorical=[store_c, item_c], target=target_c)
    return df, info


if __name__ == "__main__":  # smoke check
    from .data import ensure_raw_data, load_raw
    cfg = load_config()
    ensure_raw_data(cfg)
    feats, info = build_features(load_raw(cfg), cfg)
    print(f"rows={len(feats):,} numeric={len(info.numeric)} cat={info.categorical}")
    print("features:", info.all_features)
