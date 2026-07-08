"""[HARD] Leakage unit test (spec section 10).

Run: `pytest src/test_leakage.py -v`

Proves, on real generated data through the real `build_features`:
  1. lag_k at date d == sales at date d-k (for a sampled series),
  2. rolling means exclude the current row (built on shift(1)),
  3. NO feature column is identical to the current-row `sales`.

Modelling must not start until this passes.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from .data import make_synthetic
from .features import build_features
from .utils import load_config


@pytest.fixture(scope="module")
def built():
    """Small but >365-day dataset so lag_365 is exercised; deterministic."""
    cfg = copy.deepcopy(load_config())
    cfg["synthetic"].update({"n_stores": 3, "n_items": 3,
                             "start": "2013-01-01", "end": "2015-06-30"})
    raw = make_synthetic(cfg)
    feats, info = build_features(raw, cfg)
    return raw, feats, info, cfg


def test_lag_equals_shifted_sales(built):
    raw, feats, info, cfg = built
    lags = cfg["features"]["lags"]
    rng = np.random.default_rng(0)
    # sample a few (store, item) series
    series = raw[["store", "item"]].drop_duplicates().to_numpy()
    for store, item in series[rng.choice(len(series), size=3, replace=False)]:
        true = raw[(raw.store == store) & (raw.item == item)].set_index("date")["sales"]
        sub = feats[(feats.store == store) & (feats.item == item)]
        assert len(sub) > 0
        for k in lags:
            expected = sub["date"].map(lambda d, k=k: true.get(d - pd.Timedelta(days=k)))
            got = sub[f"lag_{k}"].to_numpy()
            # every kept row must have a defined lag equal to sales k days earlier
            assert not np.isnan(got).any()
            np.testing.assert_array_equal(got, expected.to_numpy())


def test_rolling_excludes_current_row(built):
    raw, feats, info, cfg = built
    w = cfg["features"]["rolling_windows"][0]
    store, item = raw[["store", "item"]].drop_duplicates().iloc[0]
    true = raw[(raw.store == store) & (raw.item == item)].set_index("date")["sales"]
    sub = feats[(feats.store == store) & (feats.item == item)]
    for d in sub["date"].sample(20, random_state=1):
        window = [true.get(d - pd.Timedelta(days=j)) for j in range(1, w + 1)]  # d-1..d-w
        expected = float(np.mean(window))
        got = float(sub.loc[sub["date"] == d, f"roll_mean_{w}"].iloc[0])
        assert got == pytest.approx(expected)
        # and it must NOT include today's sales
        assert got != pytest.approx(float(true.get(d))) or len(set(window + [true.get(d)])) == 1


def test_no_feature_column_equals_target(built):
    raw, feats, info, cfg = built
    target = feats[info.target].to_numpy()
    for col in info.all_features:
        vals = feats[col]
        # categorical store/item -> compare as numbers
        arr = vals.astype("float64").to_numpy() if str(vals.dtype) == "category" \
            else pd.to_numeric(vals, errors="coerce").to_numpy()
        assert not np.array_equal(arr, target), f"feature '{col}' is identical to sales"
