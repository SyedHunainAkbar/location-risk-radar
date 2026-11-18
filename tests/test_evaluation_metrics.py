"""Unit tests for survival evaluation metrics on small synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from lrr import evaluation as ev


def _data(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    duration = rng.exponential(scale=np.exp(-risk)) * 12 + 1
    event = (rng.random(n) < 0.7).astype(int)
    return duration, event, risk


def test_harrell_c_ci_bounds_and_empty():
    d, e, r = _data()
    lo, hi = ev.harrell_c_ci(d, e, r, n=30, seed=1)
    assert 0.0 <= lo <= hi <= 1.0
    lo, hi = ev.harrell_c_ci([], [], [], n=5)
    assert np.isnan(lo) and np.isnan(hi)


def test_ipcw_metrics_run_on_synthetic_data():
    pytest.importorskip("sksurv")
    d, e, r = _data(120)
    tr, te = slice(0, 60), slice(60, 120)
    tau = float(np.percentile(d[te], 70))
    c = ev.uno_c(d[tr], e[tr], d[te], e[te], r[te], tau=tau)
    assert 0.0 <= c <= 1.0

    lo_t, hi_t = np.percentile(d[te], [25, 60])
    times = [float(lo_t), float(hi_t)]
    aucs = ev.time_dependent_auc(d[tr], e[tr], d[te], e[te], r[te], times)
    assert set(aucs) == set(times)
    assert all(0.0 <= a <= 1.0 for a in aucs.values())

    surv = np.tile([0.8, 0.6], (len(d[te]), 1))
    ibs = ev.integrated_brier(d[tr], e[tr], d[te], e[te], surv, times)
    assert 0.0 <= ibs <= 1.0


def test_calibration_points_bins():
    pred = np.linspace(0, 1, 20)
    obs = (pred > 0.5).astype(float)
    cal = ev.calibration_points(pred, obs, bins=4)
    assert len(cal) == 4
    assert cal["n"].sum() == 20
    assert cal["mean_predicted"].is_monotonic_increasing


def test_bootstrap_metric_handles_errors_and_nan():
    x = np.arange(10, dtype=float)
    lo, hi = ev.bootstrap_metric(lambda a: float(a.mean()), [x], n=40, seed=2)
    assert 0.0 <= lo <= hi <= 9.0

    def boom(a):
        raise ValueError("bad sample")

    lo, hi = ev.bootstrap_metric(boom, [x], n=5)
    assert np.isnan(lo) and np.isnan(hi)
    lo, hi = ev.bootstrap_metric(lambda a: float("nan"), [x], n=5)
    assert np.isnan(lo) and np.isnan(hi)
