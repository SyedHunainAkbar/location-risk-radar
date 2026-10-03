"""Cohort construction: name normalization, labeling, selection, and QA.

Every function here mirrors the EDA notebook so the locked 30-chain cohort is
reproducible, then asserts the result against the approved constants in
:mod:`lrr.config`. Functions are small and pure where practical so they are easy
to test on a synthetic fixture.
"""

from __future__ import annotations

import re

import pandas as pd

from lrr import config

# Precompiled patterns (EDA-faithful).
_NON_WORD = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")
_CAFE = re.compile(config.CAFE_EXCLUDE_REGEX, flags=re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #


def normalize_name(name: object) -> str:
    """Normalize a restaurant name exactly as the EDA does.

    Lowercase, strip ends, remove every non-word and non-space character
    (punctuation, ampersands, apostrophes, trademark symbols), then collapse
    internal whitespace to single spaces.

    >>> normalize_name("Chick-fil-A")
    'chickfila'
    >>> normalize_name("  P.F.  Chang's® ")
    'pf changs'
    """
    lowered = str(name).lower().strip()
    no_punct = _NON_WORD.sub("", lowered)
    return _WHITESPACE.sub(" ", no_punct).strip()


def canonical_chain(name: object) -> str:
    """Return the canonical normalized chain key for a raw name.

    Applies :func:`normalize_name` then collapses known spelling variants via
    :data:`config.CHAIN_VARIANT_MAP`.
    """
    key = normalize_name(name)
    return config.CHAIN_VARIANT_MAP.get(key, key)


# --------------------------------------------------------------------------- #
# Category predicates
# --------------------------------------------------------------------------- #


def is_restaurant(categories: object) -> bool:
    """True if the category string contains "Restaurants" (case-insensitive)."""
    return "restaurants" in str(categories or "").lower()


def is_cafe(categories: object) -> bool:
    """True if the category string matches the cafe-exclusion regex."""
    return bool(_CAFE.search(str(categories or "")))


def is_fast_food_location(categories: object) -> bool:
    """True if the category string contains "Fast Food" (case-insensitive)."""
    return "fast food" in str(categories or "").lower()


# --------------------------------------------------------------------------- #
# Cleaning and chain summary
# --------------------------------------------------------------------------- #


def clean_business(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw business table (typed, validated, de-duplicated).

    Mirrors the EDA: drop rows missing id or name, dedup on ``business_id``,
    coerce numerics, and keep rows with valid stars, review counts, open flag,
    and coordinates.
    """
    out = df.copy()
    out = out.dropna(subset=["business_id", "name"])
    out = out.drop_duplicates(subset="business_id", keep="first")

    for col in ["business_id", "name", "address", "city", "state", "postal_code", "categories"]:
        if col in out:
            out[col] = out[col].astype("string").str.strip()
    for col in ["stars", "review_count", "is_open", "latitude", "longitude"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[out["stars"].between(1, 5)]
    out = out[out["review_count"] >= 0]
    out = out[out["is_open"].isin([0, 1])]
    out = out[out["latitude"].between(-90, 90)]
    out = out[out["longitude"].between(-180, 180)]
    return out.reset_index(drop=True)


def prepare_restaurant_chains(business: pd.DataFrame) -> pd.DataFrame:
    """Filter to restaurants, drop cafes, and attach chain and Fast Food columns.

    Adds ``chain`` (canonical key) and ``is_ff_loc`` (per-location Fast Food).
    """
    cats = business["categories"].fillna("")
    rest = business[cats.map(is_restaurant)].copy()
    rest = rest[~rest["categories"].fillna("").map(is_cafe)].copy()
    rest["chain"] = rest["name"].map(canonical_chain)
    rest["is_ff_loc"] = rest["categories"].fillna("").map(is_fast_food_location)
    return rest.reset_index(drop=True)


def build_chain_summary(rest_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate restaurant locations to one row per canonical chain."""
    summary = (
        rest_df.groupby("chain")
        .agg(
            display_name=("name", "first"),
            average_stars=("stars", "mean"),
            location_count=("business_id", "nunique"),
            total_review_count=("review_count", "sum"),
            is_fast_food=("is_ff_loc", "max"),
        )
        .reset_index()
    )
    return summary


# --------------------------------------------------------------------------- #
# Selection and assertion
# --------------------------------------------------------------------------- #


def select_cohort(chain_summary: pd.DataFrame) -> pd.DataFrame:
    """Apply the EDA selection: eligibility, then top 15 per cluster by reviews.

    Eligibility is average stars strictly greater than 2.5 and location count
    strictly greater than 3. Ties in review count are broken by chain key so the
    result is deterministic.
    """
    eligible = chain_summary[
        (chain_summary["average_stars"] > config.MIN_AVG_STARS)
        & (chain_summary["location_count"] > config.MIN_LOCATIONS)
    ].copy()

    def top_n(is_ff: bool, n: int) -> pd.DataFrame:
        subset = eligible[eligible["is_fast_food"] == is_ff]
        return subset.sort_values(["total_review_count", "chain"], ascending=[False, True]).head(n)

    fast = top_n(True, config.N_FAST_FOOD)
    non_fast = top_n(False, config.N_NON_FAST_FOOD)

    cohort = pd.concat([fast, non_fast], ignore_index=True)
    cohort["cluster"] = cohort["is_fast_food"].map({True: "Fast Food", False: "Non-Fast Food"})
    return cohort.reset_index(drop=True)


def assert_cohort(cohort: pd.DataFrame) -> None:
    """Assert the cohort is exactly the approved 30 chains, split 15/15.

    Checks the cluster split and set-equality of display names per cluster
    against the approved constants in :mod:`lrr.config`.

    Raises:
        AssertionError: If any check fails.
    """
    counts = cohort["cluster"].value_counts().to_dict()
    assert counts.get("Fast Food") == config.N_FAST_FOOD, (
        f"Fast Food count must be {config.N_FAST_FOOD}, got {counts.get('Fast Food')}."
    )
    assert counts.get("Non-Fast Food") == config.N_NON_FAST_FOOD, (
        f"Non-Fast Food count must be {config.N_NON_FAST_FOOD}, got {counts.get('Non-Fast Food')}."
    )
    assert len(cohort) == config.COHORT_SIZE, (
        f"Cohort must equal {config.COHORT_SIZE} chains, got {len(cohort)}."
    )

    got_ff = set(cohort.loc[cohort["cluster"] == "Fast Food", "display_name"])
    got_nff = set(cohort.loc[cohort["cluster"] == "Non-Fast Food", "display_name"])
    want_ff = set(config.COHORT_FAST_FOOD)
    want_nff = set(config.COHORT_NON_FAST_FOOD)

    assert got_ff == want_ff, (
        "Fast Food chains do not match the approved cohort. "
        f"Missing: {sorted(want_ff - got_ff)}; "
        f"Unexpected: {sorted(got_ff - want_ff)}."
    )
    assert got_nff == want_nff, (
        "Non-Fast Food chains do not match the approved cohort. "
        f"Missing: {sorted(want_nff - got_nff)}; "
        f"Unexpected: {sorted(got_nff - want_nff)}."
    )


# --------------------------------------------------------------------------- #
# Reconciliation and data quality
# --------------------------------------------------------------------------- #


def reconciliation_table(cohort: pd.DataFrame) -> pd.DataFrame:
    """Compare per-chain location and review counts to the EDA values.

    Returns a table with observed vs expected counts and a boolean ``matches``
    column so drift from the EDA is visible at a glance.
    """
    rows = []
    for _, row in cohort.iterrows():
        name = row["display_name"]
        exp_loc, exp_rev = config.EDA_RECONCILIATION.get(name, (None, None))
        obs_loc = int(row["location_count"])
        obs_rev = int(row["total_review_count"])
        rows.append(
            {
                "chain": name,
                "cluster": row["cluster"],
                "locations": obs_loc,
                "eda_locations": exp_loc,
                "reviews": obs_rev,
                "eda_reviews": exp_rev,
                "matches": (obs_loc == exp_loc) and (obs_rev == exp_rev),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values(["cluster", "reviews"], ascending=[True, False]).reset_index(drop=True)


def data_quality_report(
    business: pd.DataFrame,
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins_exploded: pd.DataFrame,
    cohort_business_ids: set[str],
) -> dict:
    """Build a data quality report. Flags problems; never imputes.

    Reports duplicate counts, null/empty review text, the review date range, and
    the cohort locations with zero check-ins and zero tips.
    """
    dup_business = int(business["business_id"].duplicated().sum())
    dup_reviews = int(reviews["review_id"].duplicated().sum()) if "review_id" in reviews else 0
    dup_tips = int(tips.duplicated().sum()) if not tips.empty else 0

    if "text" in reviews and not reviews.empty:
        text = reviews["text"].astype("string")
        null_text = int(text.isna().sum() + (text.fillna("").str.strip() == "").sum())
    else:
        null_text = 0

    if "date" in reviews and not reviews.empty:
        dates = pd.to_datetime(reviews["date"], errors="coerce")
        date_min = dates.min()
        date_max = dates.max()
    else:
        date_min = date_max = pd.NaT

    ids_with_checkins = set(checkins_exploded.get("business_id", pd.Series(dtype="object")))
    ids_with_tips = set(tips.get("business_id", pd.Series(dtype="object")))
    zero_checkin_ids = sorted(cohort_business_ids - ids_with_checkins)
    zero_tip_ids = sorted(cohort_business_ids - ids_with_tips)

    return {
        "duplicate_business_ids": dup_business,
        "duplicate_review_ids": dup_reviews,
        "duplicate_tip_rows": dup_tips,
        "null_or_empty_review_text": null_text,
        "review_date_min": None if pd.isna(date_min) else date_min.date().isoformat(),
        "review_date_max": None if pd.isna(date_max) else date_max.date().isoformat(),
        "n_locations_zero_checkins": len(zero_checkin_ids),
        "n_locations_zero_tips": len(zero_tip_ids),
        "locations_zero_checkins": zero_checkin_ids,
        "locations_zero_tips": zero_tip_ids,
    }


def explode_checkins(checkins: pd.DataFrame) -> pd.DataFrame:
    """Explode the comma-separated check-in timestamps to one row each.

    Returns columns ``business_id, checkin_at``. Businesses with no check-in
    string produce no rows (flagged later as zero-checkin, never imputed).
    """
    if checkins.empty or "date" not in checkins:
        return pd.DataFrame(columns=["business_id", "checkin_at"])

    work = checkins[["business_id", "date"]].copy()
    work["date"] = work["date"].fillna("").astype(str)
    work = work[work["date"].str.strip() != ""]
    work["checkin_at"] = work["date"].str.split(", ")
    exploded = work.explode("checkin_at")
    exploded["checkin_at"] = pd.to_datetime(exploded["checkin_at"].str.strip(), errors="coerce")
    exploded = exploded.dropna(subset=["checkin_at"])
    return exploded[["business_id", "checkin_at"]].reset_index(drop=True)
