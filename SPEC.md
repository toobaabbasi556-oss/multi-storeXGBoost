# SPEC: Multi-Store Sales Forecasting (XGBoost + Feature Engineering)

A clean, correct, job-portfolio forecasting project. Implement this spec exactly.
Rules marked **[HARD]** are correctness rules — breaking them produces leaked/dishonest
results that an interviewer will catch. If a HARD rule seems impossible, STOP and ask.

## 0. Repo hygiene (do FIRST, before any modelling)
- `git init`, add `.gitignore` (ignore `data/`, `__pycache__/`, `.venv/`, model artifacts).
- Create a GitHub repo and push the skeleton immediately. The repo IS the portfolio artifact.
- Pin versions in `requirements.txt`. Set `SEED = 42` everywhere (numpy, random, xgboost).

## 1. Structure
```
sales-forecasting/
├── README.md
├── requirements.txt
├── config.yaml
├── data/{raw,processed}/
├── src/{data.py,features.py,splits.py,baselines.py,models.py,evaluate.py,train.py}
├── notebooks/01_eda.ipynb
├── reports/results.md
└── run.py            # python run.py reproduces everything from data/raw/train.csv
```

## 2. Data
- Kaggle "Store Item Demand Forecasting Challenge" (`demand-forecasting-kernels-only`),
  file `train.csv`. Columns: date (daily), store (1-10), item (1-50), sales (int).
  ~913k rows, 2013-2017, 500 series.
- Download in README: `kaggle competitions download -c demand-forecasting-kernels-only`
  (needs one-time rules acceptance on the site). Do not invent a mirror URL.
- `data.py`: load, parse dates, assert no missing dates per (store,item), assert sales >= 0.

## 3. Forecasting setup
- **Model at DAILY resolution.** Aggregate predictions to MONTHLY totals only for reporting
  (this matches the CV wording "monthly sales" without crippling the model).
- **1-step-ahead walk-forward**: for each day in the holdout, predict using only true history
  up to the previous day, then advance. State this assumption plainly in the README.

## 4. Features — `features.py`
**[HARD] All time features computed group-wise per (store,item), on date-sorted data, using
`.shift()` so the current row never sees its own or any future sales.**
- Lags: `groupby(['store','item'])['sales'].shift(k)` for k in {1,7,14,28,365}.
- Rolling mean/std/min/max on the SHIFTED series: `.shift(1).rolling(w)...` for w in {7,14,28}.
- Expanding mean (shifted).
- Calendar: year, month, day, dayofweek, weekofyear, is_weekend, is_month_start/end;
  plus sin/cos cyclical encodings of month and dayofweek.
- Keep store, item as categoricals (XGBoost `enable_categorical=True`). No manual target
  encoding.
- Drop rows with NaN lags (series start) AFTER building features.

## 5. Splits — `splits.py`
**[HARD] Split by DATE, never by shuffled row index. No KFold/shuffling on this data.**
- Expanding-window time-series CV via date cutoffs (config-driven).
- Final holdout = last 6 months of 2017, untouched during all CV/tuning, scored once.
- **[HARD]** Any scaler is fit on the train portion only, then applied to val/test.

## 6. Baselines — `baselines.py` (do not skip)
- naive (last value), seasonal_naive_7, seasonal_naive_365, moving_average_28.
- **[HARD]** These appear in the final results table next to LR and XGBoost. If XGBoost
  doesn't beat seasonal-naive, report that honestly.

## 7. Models — `models.py`
- LinearRegression on scaled features (scaler fit on train fold only).
- XGBRegressor: `enable_categorical=True`, `tree_method='hist'`, `random_state=SEED`.
  Light hyperparameter tuning using the time-series CV (n_estimators, max_depth,
  learning_rate, subsample, colsample_bytree). **[HARD] tuning never touches the holdout.**

## 8. Evaluation — `evaluate.py`
- Metrics: MAE, RMSE, WAPE (`sum|y-yhat|/sum y`). WAPE is the headline.
- Aggregate + a per-store breakdown. Monthly-aggregated predicted-vs-actual comparison.
- 2-3 plots (predicted vs actual for example series) saved to `reports/`.
- SHAP feature-importance summary for the XGBoost model saved to `reports/` — this gives you
  a clean "I can interpret my model" talking point for interviews.

## 9. Reporting — `reports/results.md` + `README.md`
- Results table: rows = {naive, seasonal_naive_7, seasonal_naive_365, moving_avg_28,
  LinearRegression, XGBoost}, columns = {MAE, RMSE, WAPE}.
- README: 1-paragraph problem statement, dataset + download, forecasting mode (1-step
  walk-forward), how to run, the results table, top SHAP features, short honest limitations.

## 10. Definition of done
- `python run.py` reproduces every number in `reports/results.md` from `data/raw/train.csv`.
- Leakage unit test in `src/`: for a sampled (store,item), assert `lag_k` at date d equals
  `sales` at date d-k, and assert NO feature column ever equals the current-row `sales`.
- Repo committed and pushed to GitHub.
- Print the results table and a one-line verdict to stdout:
  "XGBoost beats the best baseline by X% WAPE" (or state plainly that it does not).

## OPTIONAL (only if time permits — not required)
- Add one statistical baseline (AutoARIMA via `statsforecast`) for a stronger comparison.
- Nothing else. No dashboards, no deep learning, no experiment-tracking frameworks.
