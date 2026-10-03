"""Full-corpus ingestion: restaurant universe and a streaming monthly panel.

We build `all_restaurants` (every restaurant business with chain, cluster, and an
`is_cohort` flag) and a monthly activity panel over all restaurants, computed in a
single streaming pass that never holds the 7M review texts in memory at once. We log
wall-clock time and peak memory so the one-pass claim is auditable.

Only pandas and numpy are required. The streaming pass accumulates per
(business_id, month) sums and never materializes review text beyond the current
chunk.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd

from lrr import cohort as cohort_mod
from lrr import config

# --------------------------------------------------------------------------- #
# Restaurant universe
# --------------------------------------------------------------------------- #


def build_all_restaurants(
    business_df: pd.DataFrame, cohort_business_ids: Iterable[str]
) -> pd.DataFrame:
    """Build the restaurant universe table.

    Keeps businesses whose categories contain "Restaurants" (cafes excluded exactly
    as in the cohort pipeline), attaches the canonical ``chain`` and the ``cluster``
    (same Fast Food rule, applied per chain), and flags cohort membership.

    Returns columns: business_id, name, chain, cluster, is_cohort, city, state,
    stars, review_count, is_open, latitude, longitude (as available).
    """
    cohort_ids = set(map(str, cohort_business_ids))
    rest = cohort_mod.prepare_restaurant_chains(business_df)  # adds chain, is_ff_loc

    # Cluster is defined per chain: Fast Food if ANY location of the chain is tagged.
    chain_is_ff = rest.groupby("chain")["is_ff_loc"].max()
    rest = rest.copy()
    rest["cluster"] = rest["chain"].map(
        chain_is_ff.map({True: "Fast Food", False: "Non-Fast Food"})
    )
    rest["is_cohort"] = rest["business_id"].astype(str).isin(cohort_ids)

    keep = [
        c
        for c in [
            "business_id",
            "name",
            "chain",
            "cluster",
            "is_cohort",
            "city",
            "state",
            "stars",
            "review_count",
            "is_open",
            "latitude",
            "longitude",
        ]
        if c in rest.columns
    ]
    return rest[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Resource logging
# --------------------------------------------------------------------------- #


def _peak_rss_mb() -> float | None:
    """Best-effort peak resident memory in MB, or None if unavailable."""
    try:  # POSIX
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB, macOS reports bytes.
        return ru / 1024 if ru > 10**7 else ru / 1024
    except Exception:
        pass
    try:  # cross-platform
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def log_resources(label: str, start_time: float) -> dict:
    """Log and return wall-clock seconds and peak RSS (MB) since ``start_time``."""
    elapsed = time.time() - start_time
    rss = _peak_rss_mb()
    rss_str = f"{rss:.0f} MB" if rss is not None else "unavailable"
    print(f"[resources] {label}: {elapsed:.1f}s wall, peak RSS {rss_str}")
    return {"label": label, "seconds": elapsed, "peak_rss_mb": rss}


# --------------------------------------------------------------------------- #
# Streaming monthly panel
# --------------------------------------------------------------------------- #


def _month_key(dates: pd.Series) -> pd.Series:
    """Map datetimes to a 'YYYY-MM' month string, dropping NaT."""
    d = pd.to_datetime(dates, errors="coerce")
    return d.dt.to_period("M").astype(str)


def accumulate_review_chunk(acc: dict, chunk: pd.DataFrame, restaurant_ids: set) -> None:
    """Accumulate review count and star-sum per (business_id, month) from a chunk.

    ``acc`` maps (business_id, month) -> [n_reviews, star_sum]. We keep only
    restaurant businesses and never retain the review text.
    """
    sub = chunk[chunk["business_id"].isin(restaurant_ids)]
    if sub.empty:
        return
    months = _month_key(sub["date"])
    stars = pd.to_numeric(sub["stars"], errors="coerce")
    for bid, m, s in zip(sub["business_id"].to_numpy(), months.to_numpy(), stars.to_numpy()):
        if m == "NaT" or pd.isna(s):
            continue
        cell = acc[(bid, m)]
        cell[0] += 1
        cell[1] += float(s)


def _accumulate_count(
    acc: dict, chunk: pd.DataFrame, restaurant_ids: set, date_col: str, slot: int, width: int
) -> None:
    """Accumulate a simple per (business_id, month) count into slot ``slot``."""
    if chunk is None or chunk.empty or date_col not in chunk:
        return
    sub = chunk[chunk["business_id"].isin(restaurant_ids)]
    if sub.empty:
        return
    months = _month_key(sub[date_col])
    for bid, m in zip(sub["business_id"].to_numpy(), months.to_numpy()):
        if m == "NaT":
            continue
        acc[(bid, m)][slot] += 1


def stream_activity_panel(
    review_chunks: Iterable[pd.DataFrame],
    tip_chunks: Iterable[pd.DataFrame] | None,
    checkins: pd.DataFrame | None,
    restaurant_ids: Iterable[str],
    log: bool = True,
) -> pd.DataFrame:
    """Build the monthly activity panel in one streaming pass.

    Accumulates per (business_id, month): review count and star-sum (for mean_stars),
    tip count, and check-in count. Reviews and tips arrive as chunk iterables so the
    text is never fully materialized; check-ins (already exploded and far smaller)
    may be passed as a single frame. Only restaurant business ids are kept.

    Returns a DataFrame with :data:`config.PANEL_COLUMNS`.
    """
    start = time.time()
    rid = set(map(str, restaurant_ids))
    # acc[(bid, month)] = [n_reviews, star_sum, n_tips, n_checkins]
    acc: dict = defaultdict(lambda: [0, 0.0, 0, 0])

    for chunk in review_chunks:
        chunk = chunk.copy()
        chunk["business_id"] = chunk["business_id"].astype(str)
        accumulate_review_chunk(acc, chunk, rid)

    if tip_chunks is not None:
        for chunk in tip_chunks:
            chunk = chunk.copy()
            chunk["business_id"] = chunk["business_id"].astype(str)
            _accumulate_count(acc, chunk, rid, "date", slot=2, width=4)

    if checkins is not None and not checkins.empty:
        ck = checkins.copy()
        ck["business_id"] = ck["business_id"].astype(str)
        _accumulate_count(acc, ck, rid, "checkin_at", slot=3, width=4)

    panel = finalize_panel(acc)
    if log:
        log_resources(f"activity panel ({len(panel):,} rows)", start)
    return panel


def finalize_panel(acc: dict) -> pd.DataFrame:
    """Turn the accumulator into the tidy monthly panel with mean_stars derived."""
    if not acc:
        return pd.DataFrame(columns=list(config.PANEL_COLUMNS))
    rows = []
    for (bid, month), (n_rev, star_sum, n_tips, n_ck) in acc.items():
        rows.append(
            {
                "business_id": bid,
                "month": month,
                "n_reviews": int(n_rev),
                "mean_stars": (star_sum / n_rev) if n_rev else np.nan,
                "n_tips": int(n_tips),
                "n_checkins": int(n_ck),
            }
        )
    panel = pd.DataFrame(rows, columns=list(config.PANEL_COLUMNS))
    return panel.sort_values(["business_id", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Panel -> pre-T window features (Tier 1 inputs)
# --------------------------------------------------------------------------- #


def panel_window_features(
    panel: pd.DataFrame, T, windows=config.VOLUME_WINDOWS_MONTHS
) -> pd.DataFrame:
    """Derive strictly-pre-T aggregate features per business from the monthly panel.

    Produces review/tip/check-in volume over each window ending at ``T``, review
    velocity slope over the velocity window, mean stars all-time and last 12 months,
    and a stars trend. No text and no topics (those belong to the cohort tier).
    """
    T = pd.Timestamp(T)
    p = panel.copy()
    p["month_ts"] = pd.to_datetime(p["month"], errors="coerce")
    p = p[p["month_ts"] < T]

    def window_start(w):
        return T - pd.DateOffset(months=w)

    rows = []
    for bid, g in p.groupby("business_id"):
        feats = {"business_id": bid}
        for w in windows:
            start = window_start(w)
            win = g[g["month_ts"] >= start]
            feats[f"review_vol_{w}m"] = float(win["n_reviews"].sum())
            feats[f"tip_vol_{w}m"] = float(win["n_tips"].sum())
            feats[f"checkin_vol_{w}m"] = float(win["n_checkins"].sum())
        # Velocity slope over the velocity window on monthly review counts.
        vstart = window_start(config.VELOCITY_WINDOW_MONTHS)
        vwin = g[g["month_ts"] >= vstart].sort_values("month_ts")
        feats["review_velocity_slope"] = _ols_slope(vwin["n_reviews"].to_numpy())
        # Stars all-time (review-weighted) and last 12 months, plus trend.
        total_rev = g["n_reviews"].sum()
        feats["stars_all"] = (
            float((g["mean_stars"] * g["n_reviews"]).sum() / total_rev) if total_rev else np.nan
        )
        last12 = g[g["month_ts"] >= window_start(12)]
        l12_rev = last12["n_reviews"].sum()
        feats["stars_12m"] = (
            float((last12["mean_stars"] * last12["n_reviews"]).sum() / l12_rev)
            if l12_rev
            else feats["stars_all"]
        )
        feats["stars_trend"] = _ols_slope(g.sort_values("month_ts")["mean_stars"].to_numpy())
        feats["tip_vol"] = float(g["n_tips"].sum())
        feats["checkin_vol"] = float(g["n_checkins"].sum())
        rows.append(feats)
    return pd.DataFrame(rows)


def _ols_slope(y: np.ndarray) -> float:
    """Least-squares slope of y vs 0..n-1; 0 for n < 2 or NaN-heavy input."""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - y.mean())).sum() / denom)
