"""End-to-end training + evaluation pipeline.

Order: ensure data -> features -> hold out last 6 months of 2017 -> tune XGBoost
on expanding-window CV (holdout untouched) -> fit LinearRegression and XGBoost on
the full training portion -> score every model + baseline ONCE on the holdout ->
build breakdowns, plots, and SHAP. run.py consumes the returned PipelineResult.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import baselines, evaluate
from .data import data_is_synthetic, ensure_raw_data, load_raw
from .features import FeatureInfo, build_features
from .models import LinearModel, TuningResult, XGBModel, tune_xgboost
from .splits import cv_folds, holdout_split
from .utils import load_config, resolve, set_seed


@dataclass
class PipelineResult:
    results_table: dict[str, dict[str, float]]      # model -> {MAE, RMSE, WAPE}
    tuning: TuningResult
    per_store: pd.DataFrame
    monthly: pd.DataFrame
    shap_top: list[tuple[str, float]]
    plot_paths: list[Path] = field(default_factory=list)
    is_synthetic: bool = True
    n_train: int = 0
    n_holdout: int = 0
    holdout_range: tuple[str, str] = ("", "")
    feature_names: list[str] = field(default_factory=list)


def train_and_evaluate(cfg: dict | None = None) -> PipelineResult:
    cfg = cfg or load_config()
    set_seed(cfg.get("seed", 42))
    reports_dir = resolve(cfg["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- data + features ---------------------------------------------------
    ensure_raw_data(cfg)
    raw = load_raw(cfg)
    feats, info = build_features(raw, cfg)
    date_c, target_c = cfg["data"]["date_col"], info.target

    # --- split: expanding CV folds + single final holdout ------------------
    train_df, holdout = holdout_split(feats, cfg)
    folds = cv_folds(train_df, cfg)

    y_hold = holdout[target_c].to_numpy("float64")
    table: dict[str, dict[str, float]] = {}

    # --- baselines (scored once on holdout) --------------------------------
    for name, pred in baselines.predict_all(holdout).items():
        table[name] = evaluate.metrics(y_hold, pred)

    # --- LinearRegression --------------------------------------------------
    lin = LinearModel(info).fit(train_df)
    table["LinearRegression"] = evaluate.metrics(y_hold, lin.predict(holdout))

    # --- XGBoost: tune on CV (holdout untouched), fit on full train --------
    tuning = tune_xgboost(train_df, folds, info, cfg)
    xgb = XGBModel(info, tuning.best_params).fit(train_df)
    xgb_pred = xgb.predict(holdout)
    table["XGBoost"] = evaluate.metrics(y_hold, xgb_pred)

    # --- breakdowns, plots, SHAP (all on the XGBoost holdout predictions) ---
    holdout = holdout.assign(_pred=xgb_pred)
    per_store = evaluate.per_store_breakdown(holdout, target_c, "_pred")
    monthly = evaluate.monthly_totals(holdout, date_c, target_c, "_pred")

    plot_paths = evaluate.plot_example_series(
        holdout, cfg["evaluate"]["example_series"], date_c, target_c, "_pred", reports_dir)
    plot_paths.append(evaluate.plot_monthly(monthly, reports_dir))

    n_shap = min(cfg["evaluate"]["shap_sample_size"], len(train_df))
    X_sample = train_df[info.all_features].sample(n=n_shap, random_state=cfg.get("seed", 42))
    shap_top = evaluate.shap_summary(xgb, X_sample, info.all_features, reports_dir)
    plot_paths.append(reports_dir / "shap_summary.png")

    # keep an ordered results table
    order = ["naive", "seasonal_naive_7", "seasonal_naive_365", "moving_avg_28",
             "LinearRegression", "XGBoost"]
    ordered = {k: table[k] for k in order if k in table}

    return PipelineResult(
        results_table=ordered,
        tuning=tuning,
        per_store=per_store,
        monthly=monthly,
        shap_top=shap_top,
        plot_paths=plot_paths,
        is_synthetic=data_is_synthetic(cfg),
        n_train=len(train_df),
        n_holdout=len(holdout),
        holdout_range=(str(holdout[date_c].min().date()), str(holdout[date_c].max().date())),
        feature_names=info.all_features,
    )
