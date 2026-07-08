"""Reproduce the entire project from data/raw/train.csv.

    python run.py

Ensures data exists (synthesises a schema-identical train.csv if the real Kaggle
file is absent), builds leakage-safe features, tunes + fits models, scores every
model and baseline once on the held-out last 6 months of 2017, writes
reports/results.md, and prints the results table plus a one-line verdict.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.train import PipelineResult, train_and_evaluate
from src.utils import load_config, resolve


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> markdown table (avoids a tabulate dependency).

    Formats per COLUMN dtype so integer columns don't get upcast to float
    (which happens if you iterate all-numeric rows via iterrows)."""
    cols = list(df.columns)

    def fmt(col: str):
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt):
            return lambda v: str(int(v))
        if pd.api.types.is_float_dtype(dt):
            return lambda v: f"{v:.4f}"
        return str

    formatters = {c: fmt(c) for c in cols}
    data = {c: df[c].tolist() for c in cols}
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = [
        "| " + " | ".join(formatters[c](data[c][i]) for c in cols) + " |"
        for i in range(len(df))
    ]
    return "\n".join([header, sep, *rows])


def _fmt_table(results: dict[str, dict[str, float]]) -> str:
    lines = ["| Model | MAE | RMSE | WAPE |", "|---|---:|---:|---:|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['MAE']:.3f} | {m['RMSE']:.3f} | {m['WAPE']:.4f} |")
    return "\n".join(lines)


def _verdict(results: dict[str, dict[str, float]]) -> tuple[str, float]:
    baselines = ["naive", "seasonal_naive_7", "seasonal_naive_365", "moving_avg_28"]
    best_name = min(baselines, key=lambda b: results[b]["WAPE"])
    best_wape = results[best_name]["WAPE"]
    xgb_wape = results["XGBoost"]["WAPE"]
    improve = (best_wape - xgb_wape) / best_wape * 100.0
    if improve > 0:
        msg = (f"XGBoost beats the best baseline ({best_name}) by {improve:.1f}% WAPE "
               f"({xgb_wape:.4f} vs {best_wape:.4f}).")
    else:
        msg = (f"XGBoost does NOT beat the best baseline ({best_name}): "
               f"{xgb_wape:.4f} vs {best_wape:.4f} WAPE ({improve:.1f}%).")
    return msg, improve


def write_results_md(res: PipelineResult, cfg: dict) -> Path:
    path = resolve(cfg["paths"]["reports_dir"]) / "results.md"
    verdict, _ = _verdict(res.results_table)
    synth = ("> ⚠️ **These numbers are from the SYNTHETIC fallback dataset.** Download the "
             "real Kaggle `train.csv` (see README), delete `data/raw/.synthetic` and "
             "`data/raw/train.csv`, then re-run `python run.py` for real numbers.\n\n"
             if res.is_synthetic else "")

    best = res.tuning
    tune_rows = "\n".join(
        f"| {r.get('n_estimators')} | {r.get('max_depth')} | {r.get('learning_rate')} "
        f"| {r.get('subsample')} | {r.get('colsample_bytree')} | {r['cv_wape']:.4f} |"
        for r in sorted(best.table, key=lambda r: r["cv_wape"]))
    per_store = _df_to_md(res.per_store)
    monthly = _df_to_md(res.monthly.assign(
        month=lambda d: d["month"].dt.strftime("%Y-%m"),
        actual=lambda d: d["actual"].round(0).astype(int),
        predicted=lambda d: d["predicted"].round(0).astype(int),
    ))
    shap_lines = "\n".join(f"{i+1}. `{f}` — {v:.3f}"
                           for i, (f, v) in enumerate(res.shap_top[:10]))

    md = f"""# Results — Multi-Store Sales Forecasting

{synth}**Forecasting mode:** 1-step-ahead walk-forward with true history. Model trained at
daily resolution; predictions aggregated to monthly totals for reporting.
**Holdout:** {res.holdout_range[0]} → {res.holdout_range[1]} (last 6 months of 2017),
scored exactly once. Train rows: {res.n_train:,} · Holdout rows: {res.n_holdout:,}.

## Results table (headline metric = WAPE)

{_fmt_table(res.results_table)}

**Verdict:** {verdict}

## XGBoost hyperparameter tuning (expanding-window time-series CV; holdout untouched)

Best params: `{best.best_params}` (mean CV WAPE {best.best_wape:.4f})

| n_estimators | max_depth | learning_rate | subsample | colsample_bytree | CV WAPE |
|---:|---:|---:|---:|---:|---:|
{tune_rows}

## Per-store breakdown (XGBoost, holdout)

{per_store}

## Monthly totals — actual vs XGBoost (holdout)

{monthly}

## Top SHAP features (XGBoost)

{shap_lines}

## Plots

- `pred_vs_actual_s*_i*.png` — daily actual vs predicted for example series
- `monthly_actual_vs_pred.png` — monthly totals, actual vs predicted
- `shap_summary.png` — SHAP feature-importance summary

## Honest limitations

- WAPE is dominated by high-volume series; small-volume series can be worse.
- Only light hyperparameter tuning (a small grid) was done.
- 1-step-ahead with true history is the easiest forecasting regime; multi-step
  recursive forecasting would be harder and is out of scope for this spec.
"""
    path.write_text(md)
    return path


def main() -> None:
    cfg = load_config()
    res = train_and_evaluate(cfg)
    md_path = write_results_md(res, cfg)

    # Print results table + verdict to stdout.
    print("\n" + _fmt_table(res.results_table).replace("|", " ").replace("---", ""))
    print()
    verdict, _ = _verdict(res.results_table)
    print(verdict)
    if res.is_synthetic:
        print("(numbers above are from the SYNTHETIC fallback dataset — see README)")
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
