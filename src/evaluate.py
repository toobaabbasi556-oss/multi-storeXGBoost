"""Metrics, breakdowns, plots, and SHAP.

Headline metric is WAPE = sum|y - yhat| / sum y. Also reports MAE and RMSE, a
per-store breakdown, a monthly-aggregated actual-vs-predicted comparison, example
predicted-vs-actual plots, and a SHAP summary for the XGBoost model.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.sum(np.abs(y))
    return float(np.sum(np.abs(y - yhat)) / denom) if denom else float("nan")


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    return {"MAE": mae(y, yhat), "RMSE": rmse(y, yhat), "WAPE": wape(y, yhat)}


# --------------------------------------------------------------------------
# Breakdowns
# --------------------------------------------------------------------------
def per_store_breakdown(df: pd.DataFrame, y_col: str, pred_col: str,
                        store_col: str = "store") -> pd.DataFrame:
    """WAPE and MAE per store."""
    rows = []
    for store, g in df.groupby(store_col, observed=True):
        y, yhat = g[y_col].to_numpy("float64"), g[pred_col].to_numpy("float64")
        rows.append({"store": int(store), "MAE": mae(y, yhat), "WAPE": wape(y, yhat)})
    return pd.DataFrame(rows).sort_values("store").reset_index(drop=True)


def monthly_totals(df: pd.DataFrame, date_col: str, y_col: str,
                   pred_col: str) -> pd.DataFrame:
    """Aggregate daily predictions to MONTHLY totals (spec: report at month level)."""
    m = df.copy()
    m["month"] = pd.to_datetime(m[date_col]).dt.to_period("M").dt.to_timestamp()
    out = m.groupby("month").agg(actual=(y_col, "sum"),
                                 predicted=(pred_col, "sum")).reset_index()
    out["abs_pct_err"] = (out["predicted"] - out["actual"]).abs() / out["actual"]
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_example_series(df: pd.DataFrame, examples: list[dict], date_col: str,
                        y_col: str, pred_col: str, reports_dir: Path) -> list[Path]:
    """Daily actual-vs-predicted over the holdout for each example (store, item)."""
    paths = []
    for ex in examples:
        s, it = ex["store"], ex["item"]
        g = df[(df["store"] == s) & (df["item"] == it)].sort_values(date_col)
        if g.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(g[date_col], g[y_col], label="actual", color="#1f77b4", lw=1.2)
        ax.plot(g[date_col], g[pred_col], label="XGBoost", color="#d62728",
                lw=1.2, alpha=0.8)
        ax.set_title(f"Holdout: store {s}, item {it} — actual vs XGBoost (daily)")
        ax.set_ylabel("sales"); ax.legend(); fig.autofmt_xdate()
        p = reports_dir / f"pred_vs_actual_s{s}_i{it}.png"
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        paths.append(p)
    return paths


def plot_monthly(monthly: pd.DataFrame, reports_dir: Path) -> Path:
    """Monthly total actual vs predicted (summed over all series)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["month"], monthly["actual"], marker="o", label="actual")
    ax.plot(monthly["month"], monthly["predicted"], marker="o", label="XGBoost")
    ax.set_title("Holdout monthly totals — actual vs XGBoost")
    ax.set_ylabel("total sales"); ax.legend(); fig.autofmt_xdate()
    p = reports_dir / "monthly_actual_vs_pred.png"
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    return p


# --------------------------------------------------------------------------
# SHAP (TreeSHAP via XGBoost's exact pred_contribs — robust with categoricals)
# --------------------------------------------------------------------------
def shap_summary(xgb_model, X_sample: pd.DataFrame, feature_names: list[str],
                 reports_dir: Path, top_n: int = 15) -> list[tuple[str, float]]:
    """Compute exact TreeSHAP values and save a beeswarm summary. Returns the
    top features by mean(|SHAP|)."""
    import xgboost as xgb

    booster = xgb_model.model.get_booster()
    dm = xgb.DMatrix(X_sample, enable_categorical=True)
    contribs = booster.predict(dm, pred_contribs=True)   # (n, n_feat + 1); last = bias
    shap_values = contribs[:, :-1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    ranking = sorted(zip(feature_names, mean_abs), key=lambda t: t[1], reverse=True)

    # Numeric copy for the beeswarm's colour/position (categoricals -> codes).
    X_num = X_sample.copy()
    for c in X_num.columns:
        if str(X_num[c].dtype) == "category":
            X_num[c] = X_num[c].cat.codes
    X_num = X_num.astype("float64")

    saved = reports_dir / "shap_summary.png"
    try:
        import warnings

        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # shap's use of the global RNG
            shap.summary_plot(shap_values, features=X_num, feature_names=feature_names,
                              show=False, max_display=top_n)
        plt.tight_layout(); plt.savefig(saved, dpi=110); plt.close()
    except Exception:
        # Fallback: horizontal bar of mean(|SHAP|).
        top = ranking[:top_n][::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([t[0] for t in top], [t[1] for t in top], color="#1f77b4")
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title("XGBoost SHAP feature importance")
        fig.tight_layout(); fig.savefig(saved, dpi=110); plt.close(fig)
    return ranking[:top_n]
