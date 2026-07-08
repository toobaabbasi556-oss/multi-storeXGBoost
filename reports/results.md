# Results — Multi-Store Sales Forecasting

> ⚠️ **These numbers are from the SYNTHETIC fallback dataset.** Download the real Kaggle `train.csv` (see README), delete `data/raw/.synthetic` and `data/raw/train.csv`, then re-run `python run.py` for real numbers.

**Forecasting mode:** 1-step-ahead walk-forward with true history. Model trained at
daily resolution; predictions aggregated to monthly totals for reporting.
**Holdout:** 2017-07-01 → 2017-12-31 (last 6 months of 2017),
scored exactly once. Train rows: 638,500 · Holdout rows: 92,000.

## Results table (headline metric = WAPE)

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| naive | 5.167 | 6.790 | 0.2530 |
| seasonal_naive_7 | 4.922 | 6.415 | 0.2410 |
| seasonal_naive_365 | 5.150 | 6.759 | 0.2522 |
| moving_avg_28 | 3.948 | 5.127 | 0.1933 |
| LinearRegression | 3.601 | 4.665 | 0.1763 |
| XGBoost | 3.490 | 4.540 | 0.1709 |

**Verdict:** XGBoost beats the best baseline (moving_avg_28) by 11.6% WAPE (0.1709 vs 0.1933).

## XGBoost hyperparameter tuning (expanding-window time-series CV; holdout untouched)

Best params: `{'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8}` (mean CV WAPE 0.1707)

| n_estimators | max_depth | learning_rate | subsample | colsample_bytree | CV WAPE |
|---:|---:|---:|---:|---:|---:|
| 400 | 6 | 0.05 | 0.8 | 0.8 | 0.1707 |
| 400 | 8 | 0.05 | 0.8 | 0.8 | 0.1708 |
| 400 | 6 | 0.1 | 0.8 | 0.8 | 0.1708 |
| 400 | 8 | 0.1 | 0.8 | 0.8 | 0.1718 |

## Per-store breakdown (XGBoost, holdout)

| store | MAE | WAPE |
|---|---|---|
| 1 | 3.0163 | 0.2003 |
| 2 | 3.0905 | 0.1902 |
| 3 | 3.2470 | 0.1857 |
| 4 | 3.3219 | 0.1777 |
| 5 | 3.4251 | 0.1731 |
| 6 | 3.6194 | 0.1722 |
| 7 | 3.6602 | 0.1656 |
| 8 | 3.7578 | 0.1607 |
| 9 | 3.8539 | 0.1567 |
| 10 | 3.9109 | 0.1514 |

## Monthly totals — actual vs XGBoost (holdout)

| month | actual | predicted | abs_pct_err |
|---|---|---|---|
| 2017-07 | 415670 | 411200 | 0.0108 |
| 2017-08 | 378382 | 375853 | 0.0067 |
| 2017-09 | 324358 | 322446 | 0.0059 |
| 2017-10 | 286359 | 285818 | 0.0019 |
| 2017-11 | 238918 | 238774 | 0.0006 |
| 2017-12 | 235063 | 234018 | 0.0044 |

## Top SHAP features (XGBoost)

1. `roll_mean_28` — 4.529
2. `roll_mean_14` — 2.463
3. `expanding_mean` — 1.291
4. `dayofweek` — 1.239
5. `month_cos` — 0.516
6. `weekofyear` — 0.505
7. `is_weekend` — 0.280
8. `lag_365` — 0.186
9. `item` — 0.186
10. `month` — 0.179

## Plots

- `pred_vs_actual_s*_i*.png` — daily actual vs predicted for example series
- `monthly_actual_vs_pred.png` — monthly totals, actual vs predicted
- `shap_summary.png` — SHAP feature-importance summary

## Honest limitations

- WAPE is dominated by high-volume series; small-volume series can be worse.
- Only light hyperparameter tuning (a small grid) was done.
- 1-step-ahead with true history is the easiest forecasting regime; multi-step
  recursive forecasting would be harder and is out of scope for this spec.
