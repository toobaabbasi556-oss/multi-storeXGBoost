# Multi-Store Sales Forecasting — XGBoost + Feature Engineering

Forecasting daily `sales` for **500 series** (10 stores × 50 items) from the Kaggle
*Store Item Demand Forecasting Challenge*. The project is built to be **correct first**:
every time-derived feature is leakage-safe, splits are strictly by date, and a
single held-out window (the last 6 months of 2017) is scored exactly once against a
full set of naive baselines. XGBoost is the headline model, with LinearRegression and
four baselines for honest comparison, plus SHAP for interpretability.

## Problem statement
Given ~913k daily rows (2013–2017) of `(date, store, item, sales)`, predict future
`sales`. We model at **daily resolution** and aggregate to **monthly totals** for
reporting. The headline metric is **WAPE** (`sum|y − ŷ| / sum y`).

## Dataset & download
Kaggle *Store Item Demand Forecasting Challenge* (`demand-forecasting-kernels-only`),
file `train.csv` — columns `date, store (1–10), item (1–50), sales (int)`.

```bash
# one-time: accept the competition rules on the Kaggle website, then:
pip install kaggle                       # if not already installed
mkdir -p data/raw
kaggle competitions download -c demand-forecasting-kernels-only -p data/raw
unzip -o data/raw/demand-forecasting-kernels-only.zip -d data/raw
rm -f data/raw/.synthetic                # remove the synthetic marker
python run.py                            # reproduces every number below
```

> **No Kaggle file yet?** `run.py` writes a **schema-identical synthetic `train.csv`**
> (same columns, same 2013–2017 daily grid, seasonal + weekly + trend structure) so the
> whole pipeline runs end-to-end. Results are clearly marked synthetic until you swap in
> the real file — **no code changes needed**. Never invent a mirror URL for the real data.

## Forecasting mode
**1-step-ahead walk-forward with true history.** For each holdout day *d*, the model
predicts using only true `sales` up to *d−1*. Because every feature is a `.shift()` of
the true series, all holdout features are computable from known history at once — this
is exactly 1-step-ahead walk-forward, without recursive error feedback (which would be a
different, harder problem, out of scope here).

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                      # end-to-end: data → features → models → reports
pytest src/test_leakage.py -v      # the leakage guardrail (must pass before modelling)
jupyter notebook notebooks/01_eda.ipynb   # optional EDA
```
Outputs land in `reports/` (`results.md` + PNG plots + SHAP summary). `SEED = 42`
everywhere; `python run.py` is fully reproducible from `data/raw/train.csv`.

## Method
- **Features** (`src/features.py`, all group-wise per `(store,item)`, `.shift()`-based):
  lags {1,7,14,28,365}; rolling mean/std/min/max over {7,14,28} on the `shift(1)` series;
  shifted expanding mean; calendar (year, month, day, day-of-week, week-of-year,
  is_weekend, month start/end) + sin/cos cyclical encodings; `store`/`item` as
  categoricals (XGBoost `enable_categorical=True`, no manual target encoding).
- **Splits** (`src/splits.py`): expanding-window time-series CV by **date cutoffs**;
  final holdout = last 6 months of 2017, untouched during tuning. Any scaler is fit on
  the training portion only.
- **Models** (`src/models.py`): LinearRegression on standardized features; XGBoost
  (`hist`, categorical) with a light grid tuned on the CV folds — **never** the holdout.
- **Baselines** (`src/baselines.py`): naive, seasonal-naive-7, seasonal-naive-365,
  moving-average-28.

## Results
See [`reports/results.md`](reports/results.md) for the full table, per-store breakdown,
monthly comparison, tuning grid, and plots.

<!-- RESULTS_TABLE -->
Holdout = last 6 months of 2017 (scored once). **Headline metric: WAPE.**
_Numbers below are from the **synthetic** fallback dataset; re-run after downloading the
real Kaggle file to refresh them._

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| naive | 5.167 | 6.790 | 0.2530 |
| seasonal_naive_7 | 4.922 | 6.415 | 0.2410 |
| seasonal_naive_365 | 5.150 | 6.759 | 0.2522 |
| moving_avg_28 | 3.948 | 5.127 | 0.1933 |
| LinearRegression | 3.601 | 4.665 | 0.1763 |
| **XGBoost** | **3.490** | **4.540** | **0.1709** |

**Verdict:** XGBoost beats the best baseline (moving_avg_28) by **11.6% WAPE**
(0.1709 vs 0.1933).
<!-- /RESULTS_TABLE -->

<!-- SHAP -->
**Top SHAP features (XGBoost):** `roll_mean_28`, `roll_mean_14`, `expanding_mean`,
`dayofweek`, `month_cos`, `weekofyear`, `is_weekend`, `lag_365`, `item`, `month` — i.e.
recent trailing averages plus weekly/yearly seasonality drive the forecast, exactly what
the EDA suggests. See `reports/shap_summary.png`.
<!-- /SHAP -->

## Repo layout
```
src/data.py        load + validate raw; synthetic fallback
src/features.py    leakage-safe feature engineering
src/splits.py      date-based CV folds + final holdout; train-only scaling
src/baselines.py   naive / seasonal-naive / moving-average
src/models.py      LinearRegression + XGBoost (CV-tuned)
src/evaluate.py    MAE/RMSE/WAPE, per-store, monthly, plots, SHAP
src/train.py       end-to-end pipeline
src/test_leakage.py  leakage unit test (section 10)
run.py             reproduces everything → reports/results.md
notebooks/01_eda.ipynb
config.yaml        all knobs (seed, features, splits, grid)
```

## Honest limitations
- WAPE is volume-weighted, so high-volume series dominate the headline number.
- Only light hyperparameter tuning (small grid) is performed.
- 1-step-ahead with true history is the easiest forecasting regime; multi-step recursive
  forecasting is deliberately out of scope.
- Until the real Kaggle file is downloaded, the numbers reflect synthetic data.
