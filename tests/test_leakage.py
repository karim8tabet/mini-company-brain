"""The leakage rules as tests. Uses a small synthetic table, so it runs in a second without the dataset."""
import re

import numpy as np
import pandas as pd

from brain.clean import LAB_COLS, TARGET, to_hourly
from brain.features import LAB_DELAY, build_dataset, feature_cols, time_split


def synthetic_hourly(hours=200, drop_hours=()):
    idx = pd.date_range("2017-03-10", periods=hours, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"date": idx, "amina_flow": rng.normal(500, 10, hours)})
    for c in LAB_COLS:
        df[c] = rng.normal(2, 1, hours)
    df[TARGET] = np.arange(hours, dtype=float)  # silica == hour number, easy to check shifts
    df = df.drop(index=list(drop_hours))
    return to_hourly(df)


def test_iron_concentrate_never_used():
    cols = feature_cols(build_dataset(synthetic_hourly()))
    assert not any("iron_concentrate" in c for c in cols)


def test_lab_features_lagged_at_least_two_hours():
    cols = feature_cols(build_dataset(synthetic_hourly()))
    lab_bases = [c for c in LAB_COLS]
    for c in cols:
        m = re.fullmatch(r"(.+)_lag(\d+)", c)
        if m and m.group(1) in lab_bases:
            assert int(m.group(2)) >= LAB_DELAY, c


def test_target_is_next_hour_and_baseline_is_three_hours_before_target():
    data = build_dataset(synthetic_hourly())
    # silica == hour number, so y (t+1) - baseline (t-2) must be exactly 3
    assert (data["y"] - data["baseline"]).eq(3).all()
    assert (data["y"] - data["pct_silica_concentrate_lag2"]).eq(3).all()


def test_shifts_respect_gaps():
    # Remove hours 50-59: a lag must not silently jump across the gap.
    data = build_dataset(synthetic_hourly(drop_hours=range(50, 60)))
    assert (data["y"] - data["baseline"]).eq(3).all()


def test_split_is_chronological_and_disjoint():
    train, val, test = time_split(build_dataset(synthetic_hourly()))
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) > len(val) > 0 and len(test) > 0


def test_ship_gate_needs_margin():
    from brain.train import ship_gate
    assert ship_gate(0.90, 1.0)        # 10% better -> ship
    assert not ship_gate(0.97, 1.0)    # 3% better -> not enough
    assert not ship_gate(1.10, 1.0)    # worse -> no


def test_v2_handles_drift_that_v1_cannot():
    # Silica trends upward forever: a level model can't extrapolate, a change model can.
    from brain.train import fit, predict
    data = build_dataset(synthetic_hourly(400))
    cols = feature_cols(data)
    train, _, test = time_split(data)
    model = fit({"max_iter": 50}, train, cols)
    err = (predict(model, test, cols) - test["y"]).abs().mean()
    assert err < 0.1
