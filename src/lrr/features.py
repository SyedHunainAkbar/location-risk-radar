"""Landmark survival features (stage 04).

A landmark design freezes a cutoff ``T``, builds every feature from records dated
strictly before ``T``, and observes closure over ``(T, T + 24 months]``. This module
holds the population rule, the proxy-closure event/duration logic, the leakage-safe
feature builders, and out-of-fold chain priors. It depends only on pandas and numpy
so it is fully unit-testable without any survival library.

Leakage discipline: no function here may read a record dated on or after ``T``. The
feature set deliberately excludes "days since last activity measured at T or later"
and any other post-T signal.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

import numpy as np
import pandas as pd

from lrr import config

ACTIVITY_SOURCES = ("review", "tip", "checkin")


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #


def to_ts(value) -> pd.Timestamp:
    """Coerce a date-like to a pandas Timestamp."""
    if isinstance(value, (date, datetime)):
        return pd.Timestamp(value)
    return pd.Timestamp(value)


def months_before(T, months: int) -> pd.Timestamp:
    """Return the timestamp ``months`` months before ``T`` (DateOffset)."""
    return to_ts(T) - pd.DateOffset(months=months)


def assert_no_leakage(records: pd.DataFrame, T, date_col: str = "date") -> None:
    """Raise if any record is dated on or after the landmark ``T``.

    This is the guardrail every feature source passes through before use.
    """
    if records.empty:
        return
    dates = pd.to_datetime(records[date_col], errors="coerce")
    bad = int((dates >= to_ts(T)).sum())
    if bad:
        raise AssertionError(
            f"Leakage: {bad} record(s) dated on or after landmark {to_ts(T).date()}."
        )


def _pre_T(df: pd.DataFrame, T, date_col: str = "date") -> pd.DataFrame:
    """Return the strict pre-landmark slice of a dated frame."""
    if df.empty:
        return df
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out[out[date_col] < to_ts(T)].dropna(subset=[date_col])


# --------------------------------------------------------------------------- #
# Activity consolidation and proxy closure
# --------------------------------------------------------------------------- #


def _stack_activity(
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins: pd.DataFrame,
) -> pd.DataFrame:
    """Stack the three activity sources into (business_id, date, source) rows."""
    frames = []
    specs = [
        (reviews, "date", "review"),
        (tips, "date", "tip"),
        (checkins, "checkin_at", "checkin"),
    ]
    for df, col, source in specs:
        if df is None or df.empty or col not in df:
            continue
        part = df[["business_id", col]].rename(columns={col: "date"}).copy()
        part["date"] = pd.to_datetime(part["date"], errors="coerce")
        part["source"] = source
        frames.append(part.dropna(subset=["date"]))
    if not frames:
        return pd.DataFrame(columns=["business_id", "date", "source"])
    return pd.concat(frames, ignore_index=True)


def last_activity_date(
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins: pd.DataFrame,
) -> pd.Series:
    """Proxy closure date: the last activity date per business across all sources.

    Uses ALL dates (not just pre-T); the proxy closure date defines the outcome,
    not a feature, so it is allowed to look after T.
    """
    stacked = _stack_activity(reviews, tips, checkins)
    if stacked.empty:
        return pd.Series(dtype="datetime64[ns]")
    return stacked.groupby("business_id")["date"].max()


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #


def landmark_population(
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins: pd.DataFrame,
    locations: pd.DataFrame,
    T,
) -> pd.DataFrame:
    """Select the eligible landmark population.

    A location is eligible when its first review is before ``T - 12 months`` AND it
    has any activity (review, tip, or check-in) in ``[T - 6 months, T)``. Only data
    dated before ``T`` is used for selection.

    Returns the subset of ``locations`` rows that qualify.
    """
    T = to_ts(T)
    lead_cut = months_before(T, config.POP_FIRST_REVIEW_LEAD_MONTHS)
    act_start = months_before(T, config.POP_ACTIVITY_WINDOW_MONTHS)

    rev_pre = _pre_T(reviews, T)
    first_review = rev_pre.groupby("business_id")["date"].min()
    tenured = set(first_review[first_review < lead_cut].index)

    activity = _stack_activity(reviews, tips, checkins)
    activity = activity[activity["date"] < T]
    recent = set(activity[(activity["date"] >= act_start)]["business_id"].unique())

    eligible = tenured & recent
    return locations[locations["business_id"].isin(eligible)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Event and duration
# --------------------------------------------------------------------------- #


def label_event_duration(
    population: pd.DataFrame,
    last_activity: pd.Series,
    T,
    horizon_months: int = config.HORIZON_MONTHS,
    dataset_end=config.DATASET_END,
) -> pd.DataFrame:
    """Compute the event indicator and duration for each population location.

    - proxy closure date = ``last_activity`` for the business.
    - event = 1 if ``is_open == 0`` AND ``T < proxy <= T + horizon``, else 0.
    - duration (days) = ``min(proxy, T + horizon, DATASET_END) - T``.

    Returns ``population`` with added ``proxy_closure, event, duration_days``.
    """
    T = to_ts(T)
    horizon_end = T + pd.DateOffset(months=horizon_months)
    end_cap = min(horizon_end, to_ts(dataset_end))

    out = population.copy()
    out["proxy_closure"] = out["business_id"].map(last_activity)

    proxy = out["proxy_closure"]
    is_closed = out["is_open"] == 0
    in_window = (proxy > T) & (proxy <= horizon_end)
    out["event"] = (is_closed & in_window).astype(int)

    # Duration end = min(proxy, horizon_end, dataset_end), floored at T.
    cap = proxy.where(proxy.notna(), end_cap).clip(upper=end_cap)
    cap = cap.clip(lower=T)
    out["duration_days"] = (cap - T).dt.days.astype(int)
    return out


# --------------------------------------------------------------------------- #
# Time-series feature primitives (pre-T only)
# --------------------------------------------------------------------------- #


def monthly_counts(dates: Iterable, T, months: int) -> np.ndarray:
    """Counts per month over the ``months`` months ending just before ``T``.

    Returns an array of length ``months`` ordered oldest to newest. Only dates
    strictly before ``T`` and within the window are counted.
    """
    T = to_ts(T)
    start = months_before(T, months)
    s = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    s = s[(s >= start) & (s < T)]
    counts = np.zeros(months, dtype=float)
    if s.empty:
        return counts
    # Month index 0..months-1 from the window start.
    idx = ((s.dt.year - start.year) * 12 + (s.dt.month - start.month)).to_numpy()
    idx = idx[(idx >= 0) & (idx < months)]
    for i in idx:
        counts[int(i)] += 1
    return counts


def ols_slope(y: np.ndarray) -> float:
    """Least-squares slope of ``y`` against 0..n-1. Returns 0 for n < 2."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - x_mean) * (y - y.mean())).sum() / denom)


def velocity_ratio(dates: Iterable, T) -> float:
    """Ratio of last-6-month review count to the prior-6-month count.

    Guards divide-by-zero: returns the last-6m count when the prior window is empty
    (so a cold start reads as growth), and 1.0 when both windows are empty.
    """
    T = to_ts(T)
    last_start = months_before(T, 6)
    prior_start = months_before(T, 12)
    s = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    s = s[s < T]
    last = int(((s >= last_start) & (s < T)).sum())
    prior = int(((s >= prior_start) & (s < last_start)).sum())
    if prior == 0:
        return float(last) if last else 1.0
    return float(last) / float(prior)


# --------------------------------------------------------------------------- #
# Out-of-fold chain priors
# --------------------------------------------------------------------------- #


def out_of_fold_chain_priors(
    df: pd.DataFrame,
    target: str = "event",
    group: str = "chain",
    fold: str = "fold",
    global_mean: float | None = None,
) -> pd.Series:
    """Chain-level target mean computed out of fold (no self-fold leakage).

    For each row, the prior is the mean ``target`` of its ``group`` computed from all
    folds EXCEPT the row's own fold. Chains unseen in other folds fall back to the
    global mean over the other folds.
    """
    if global_mean is None:
        global_mean = float(df[target].mean())
    priors = pd.Series(index=df.index, dtype=float)
    for f in df[fold].unique():
        in_fold = df[fold] == f
        other = df[~in_fold]
        other_mean = float(other[target].mean()) if len(other) else global_mean
        group_means = other.groupby(group)[target].mean()
        priors.loc[in_fold] = df.loc[in_fold, group].map(group_means).fillna(other_mean)
    return priors


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #


def build_features(
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins: pd.DataFrame,
    population: pd.DataFrame,
    T,
    sentiment: pd.DataFrame | None = None,
    topic_shares: pd.DataFrame | None = None,
    lexicon_terms: set[str] | None = None,
) -> pd.DataFrame:
    """Build the leakage-safe feature matrix for the population at landmark ``T``.

    All inputs are sliced to strictly-before-``T`` before any aggregation. Returns
    one row per business with the engineered features; the caller joins the labels
    (event, duration) and the out-of-fold chain priors after assigning folds.
    """
    T = to_ts(T)
    rev = _pre_T(reviews, T)
    if rev.empty:
        rev = pd.DataFrame(columns=["business_id", "date", "stars", "text"])
    tip = _pre_T(tips, T) if (tips is not None and not tips.empty) else None
    if tip is None or tip.empty:
        tip = pd.DataFrame(columns=["business_id", "date"])
    chk = (
        _pre_T(checkins, T, date_col="checkin_at")
        if (checkins is not None and not checkins.empty)
        else pd.DataFrame(columns=["business_id", "checkin_at"])
    )

    rows = []
    sent_lookup = None
    if sentiment is not None and not sentiment.empty:
        sent = sentiment.merge(reviews[["review_id", "date"]], on="review_id", how="left")
        sent["date"] = pd.to_datetime(sent["date"], errors="coerce")
        sent_lookup = sent[sent["date"] < T]

    for _, loc in population.iterrows():
        bid = loc["business_id"]
        r = rev[rev["business_id"] == bid]
        t = tip[tip["business_id"] == bid]
        c = chk[chk["business_id"] == bid]
        feats: dict[str, float] = {"business_id": bid}

        # Review volume windows.
        rdates = r["date"]
        for w in config.VOLUME_WINDOWS_MONTHS:
            start = months_before(T, w)
            feats[f"review_vol_{w}m"] = float(((rdates >= start) & (rdates < T)).sum())

        # Velocity slope (24m monthly counts) and ratio.
        feats["review_velocity_slope"] = ols_slope(
            monthly_counts(rdates, T, config.VELOCITY_WINDOW_MONTHS)
        )
        feats["review_velocity_ratio"] = velocity_ratio(rdates, T)

        # Check-ins.
        feats["checkin_vol"] = float(len(c))
        feats["checkin_slope"] = ols_slope(
            monthly_counts(c["checkin_at"], T, config.VELOCITY_WINDOW_MONTHS)
        )

        # Tips.
        feats["tip_vol"] = float(len(t))
        feats["tip_coverage"] = 1.0 if len(t) > 0 else 0.0

        # Stars all-time and last 12m, and trend.
        if len(r):
            stars = pd.to_numeric(r["stars"], errors="coerce")
            feats["stars_all"] = float(stars.mean())
            last12 = r[r["date"] >= months_before(T, 12)]
            feats["stars_12m"] = (
                float(pd.to_numeric(last12["stars"], errors="coerce").mean())
                if len(last12)
                else float(stars.mean())
            )
            monthly_star = (
                r.assign(m=r["date"].dt.to_period("M")).groupby("m")["stars"].mean().to_numpy()
            )
            feats["stars_trend"] = ols_slope(monthly_star)
            feats["review_length"] = (
                float(r["text"].astype(str).str.split().str.len().mean()) if "text" in r else 0.0
            )
            feats["share_negative"] = float(
                (pd.to_numeric(r["stars"], errors="coerce") <= 2).mean()
            )
            first = r["date"].min()
            feats["location_age_days"] = float((T - first).days)
        else:
            feats.update(
                {
                    "stars_all": np.nan,
                    "stars_12m": np.nan,
                    "stars_trend": 0.0,
                    "review_length": 0.0,
                    "share_negative": np.nan,
                    "location_age_days": 0.0,
                }
            )

        # Text sentiment mean and slope from the production scorer.
        if sent_lookup is not None:
            sl = sent_lookup[sent_lookup["business_id"] == bid]
            if len(sl):
                feats["sentiment_mean"] = float(sl["p_positive"].mean())
                monthly_sent = (
                    sl.assign(m=sl["date"].dt.to_period("M"))
                    .groupby("m")["p_positive"]
                    .mean()
                    .to_numpy()
                )
                feats["sentiment_slope"] = ols_slope(monthly_sent)
            else:
                feats["sentiment_mean"] = np.nan
                feats["sentiment_slope"] = 0.0

        # LA1 lexicon complaint rate: fraction of reviews mentioning a complaint term.
        if lexicon_terms and len(r):
            pattern = r"\b(" + "|".join(sorted(lexicon_terms)) + r")\b"
            feats["lexicon_complaint_rate"] = float(
                r["text"].astype(str).str.lower().str.contains(pattern, regex=True).mean()
            )
        elif lexicon_terms:
            feats["lexicon_complaint_rate"] = 0.0

        feats["cluster"] = loc.get("cluster")
        feats["chain"] = loc.get("chain")
        rows.append(feats)

    features = pd.DataFrame(rows)

    # Topic shares (already strictly pre-T by construction in stage 03).
    if topic_shares is not None and not topic_shares.empty:
        share_cols = [c for c in topic_shares.columns if c.startswith("topic_")]
        features = features.merge(
            topic_shares[["business_id", *share_cols]], on="business_id", how="left"
        )
        features[share_cols] = features[share_cols].fillna(0.0)

    return features


# --------------------------------------------------------------------------- #
# Full-corpus two-tier additions (Tier 1 features and labels)
# --------------------------------------------------------------------------- #


def group_key(df: pd.DataFrame, chain_col: str = "chain", id_col: str = "business_id") -> pd.Series:
    """Grouping key for leakage-safe CV: the chain when a business is chained.

    A business is "chained" when its chain has more than one location in ``df``.
    Singleton restaurants group by their own business id. This lets GroupKFold keep
    whole chains together while standalone restaurants still group by themselves.
    """
    counts = df.groupby(chain_col)[id_col].transform("nunique")
    chained = counts > 1
    return df[chain_col].astype(str).where(chained, df[id_col].astype(str))


def tier1_labels(
    all_restaurants: pd.DataFrame,
    last_activity: pd.Series,
    T,
    horizon_months: int = config.HORIZON_MONTHS,
    dataset_end=config.DATASET_END,
) -> pd.DataFrame:
    """Event and duration for the corpus tier, reusing the cohort logic.

    Thin wrapper over :func:`label_event_duration` so Tier 1 and Tier 2 share one
    definition of the outcome.
    """
    return label_event_duration(
        all_restaurants,
        last_activity,
        T,
        horizon_months=horizon_months,
        dataset_end=dataset_end,
    )


def tier1_features(
    all_restaurants: pd.DataFrame,
    panel_features: pd.DataFrame,
    sentiment_monthly: pd.DataFrame,
    T,
) -> pd.DataFrame:
    """Assemble Tier 1 features: engagement/stars from the panel + sentiment.

    ``panel_features`` is the output of :func:`lrr.corpus.panel_window_features`.
    ``sentiment_monthly`` has per (business_id, month) ``mean_p_positive`` and ``n``;
    we reduce it to strictly-pre-T mean and slope per business. No text, no topics.
    Returns one row per restaurant with ``business_id, chain, cluster`` plus features.
    """
    T = to_ts(T)
    feats = panel_features.copy()

    if sentiment_monthly is not None and not sentiment_monthly.empty:
        sm = sentiment_monthly.copy()
        sm["month_ts"] = pd.to_datetime(sm["month"], errors="coerce")
        sm = sm[sm["month_ts"] < T]
        agg = []
        for bid, g in sm.sort_values("month_ts").groupby("business_id"):
            w = g["n"].to_numpy()
            p = g["mean_p_positive"].to_numpy()
            mean_p = float(np.average(p, weights=w)) if w.sum() else np.nan
            agg.append(
                {"business_id": bid, "sentiment_mean": mean_p, "sentiment_slope": ols_slope(p)}
            )
        feats = feats.merge(pd.DataFrame(agg), on="business_id", how="left")

    meta = all_restaurants[
        [
            c
            for c in ["business_id", "chain", "cluster", "is_cohort"]
            if c in all_restaurants.columns
        ]
    ]
    return meta.merge(feats, on="business_id", how="inner").reset_index(drop=True)
