"""Survival evaluation metrics (stages 04-05).

Harrell's C index is implemented directly so it needs no survival library and is
unit-testable. Uno's C, time-dependent AUC, and the integrated Brier score wrap
scikit-survival (imported lazily, exercised offline). Bootstrap CIs, calibration
points, the Schoenfeld PH test, and the ablation table round out the module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from lrr import config

# --------------------------------------------------------------------------- #
# Harrell's concordance (library-free)
# --------------------------------------------------------------------------- #


def harrell_c(duration, event, risk) -> float:
    """Harrell's concordance index.

    A pair is comparable when the one with the shorter duration had the event.
    Concordant when the higher-risk score goes to the shorter survivor. Ties in
    risk count as half. ``risk`` must be increasing in hazard (higher = riskier).
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    n = len(duration)
    concordant = 0.0
    comparable = 0.0
    # Comparable pair: the subject that failed earlier (i, event=1) vs any subject j
    # known to survive strictly longer (duration[j] > duration[i]).
    for i in range(n):
        if event[i] != 1:
            continue
        for j in range(n):
            if i == j:
                continue
            if duration[j] <= duration[i]:
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1.0
            elif risk[i] == risk[j]:
                concordant += 0.5
    if comparable == 0:
        return float("nan")
    return concordant / comparable


# --------------------------------------------------------------------------- #
# scikit-survival wrappers (lazy)
# --------------------------------------------------------------------------- #


def harrell_c_ci(
    duration,
    event,
    risk,
    n: int = config.BOOTSTRAP_N,
    ci: float = config.BOOTSTRAP_CI,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Bootstrap confidence interval for Harrell's C index.

    Resamples rows with replacement and recomputes C each time, returning the lower
    and upper percentile bounds. This gives Tier 1 and Tier 2 concordance a CI, not
    just a point estimate.
    """
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    risk = np.asarray(risk, dtype=float)
    rng = np.random.default_rng(seed)
    m = len(duration)
    if m == 0:
        return float("nan"), float("nan")
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        val = harrell_c(duration[idx], event[idx], risk[idx])
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return float("nan"), float("nan")
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return lo, hi


def _structured(duration, event):
    """Build the scikit-survival structured array (event bool, time float)."""
    event = np.asarray(event, dtype=bool)
    duration = np.asarray(duration, dtype=float)
    return np.array(
        list(zip(event, duration)),
        dtype=[("event", "bool"), ("time", "float64")],
    )


def uno_c(train_dur, train_evt, test_dur, test_evt, risk, tau: float) -> float:
    """Uno's C index (IPCW), via scikit-survival. Imported lazily."""
    from sksurv.metrics import concordance_index_ipcw

    s_train = _structured(train_dur, train_evt)
    s_test = _structured(test_dur, test_evt)
    result = concordance_index_ipcw(s_train, s_test, np.asarray(risk), tau=tau)
    return float(result[0])


def time_dependent_auc(train_dur, train_evt, test_dur, test_evt, risk, times) -> dict:
    """Cumulative/dynamic AUC at each time point. Returns {time: auc}."""
    from sksurv.metrics import cumulative_dynamic_auc

    s_train = _structured(train_dur, train_evt)
    s_test = _structured(test_dur, test_evt)
    aucs, _ = cumulative_dynamic_auc(
        s_train, s_test, np.asarray(risk), np.asarray(times, dtype=float)
    )
    return {float(t): float(a) for t, a in zip(times, np.atleast_1d(aucs))}


def integrated_brier(train_dur, train_evt, test_dur, test_evt, surv_prob, times) -> float:
    """Integrated Brier score over ``times`` given survival probabilities."""
    from sksurv.metrics import integrated_brier_score

    s_train = _structured(train_dur, train_evt)
    s_test = _structured(test_dur, test_evt)
    return float(
        integrated_brier_score(
            s_train, s_test, np.asarray(surv_prob), np.asarray(times, dtype=float)
        )
    )


# --------------------------------------------------------------------------- #
# Calibration and bootstrap
# --------------------------------------------------------------------------- #


def calibration_points(predicted: np.ndarray, observed: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binned calibration: mean predicted vs mean observed per quantile bin."""
    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    order = np.argsort(predicted)
    pred_sorted = predicted[order]
    obs_sorted = observed[order]
    splits = np.array_split(np.arange(len(pred_sorted)), bins)
    rows = []
    for b, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": b,
                "mean_predicted": float(pred_sorted[idx].mean()),
                "mean_observed": float(obs_sorted[idx].mean()),
                "n": int(len(idx)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric(
    fn: Callable[..., float],
    arrays: Sequence[np.ndarray],
    n: int = config.BOOTSTRAP_N,
    ci: float = config.BOOTSTRAP_CI,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Bootstrap a two-sided CI for a metric computed from aligned arrays.

    ``fn`` receives the resampled arrays positionally and returns a scalar.
    """
    rng = np.random.default_rng(seed)
    arrays = [np.asarray(a) for a in arrays]
    m = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, size=m)
        try:
            val = fn(*[a[idx] for a in arrays])
        except Exception:
            continue
        if val is not None and not np.isnan(val):
            stats.append(val)
    if not stats:
        return float("nan"), float("nan")
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return lo, hi


# --------------------------------------------------------------------------- #
# Proportional hazards test and ablation
# --------------------------------------------------------------------------- #


def schoenfeld_ph_test(
    cox_model, df: pd.DataFrame, duration_col: str, event_col: str
) -> pd.DataFrame:
    """Run lifelines' proportional-hazard (Schoenfeld) test. Imported lazily.

    Returns a tidy frame of per-covariate test statistics and p-values. Covariates
    with small p-values violate proportional hazards; the model card documents the
    handling (stratification or a time interaction for the offender).
    """
    from lifelines.statistics import proportional_hazard_test

    result = proportional_hazard_test(cox_model, df, time_transform="rank")
    summary = result.summary.reset_index()
    return summary


def ablation_table(metrics_by_set: dict[str, dict]) -> pd.DataFrame:
    """Build an ablation table proving the value of text over stars.

    ``metrics_by_set`` maps feature-set name to a metrics dict (expects at least a
    ``harrell_c``). Returns rows per set plus the delta vs the stars-only baseline.
    """
    base = metrics_by_set.get("stars_only", {})
    base_c = base.get("harrell_c", float("nan"))
    rows = []
    for name in config.FEATURE_SET_NAMES:
        if name not in metrics_by_set:
            continue
        m = metrics_by_set[name]
        c = m.get("harrell_c", float("nan"))
        rows.append(
            {
                "feature_set": name,
                "harrell_c": c,
                "delta_c_vs_stars": (c - base_c) if not np.isnan(base_c) else float("nan"),
                "uno_c": m.get("uno_c", float("nan")),
                "auc_12m": m.get("auc_12m", float("nan")),
                "auc_24m": m.get("auc_24m", float("nan")),
                "ibs": m.get("ibs", float("nan")),
            }
        )
    return pd.DataFrame(rows)
