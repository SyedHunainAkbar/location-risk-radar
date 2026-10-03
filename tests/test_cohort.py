"""Tests for lrr.cohort on a tiny synthetic fixture (no real Yelp data)."""

from __future__ import annotations

import pandas as pd
import pytest

from lrr import cohort as C
from lrr import config

# --------------------------------------------------------------------------- #
# Name normalization
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Chick-fil-A", "chickfila"),
        ("CHICK-FIL-A", "chickfila"),
        ("  Chili's  ", "chilis"),
        # '+' is stripped, then the resulting double space collapses to one.
        ("Applebee's Grill + Bar", "applebees grill bar"),
        ("P.F. Chang's", "pf changs"),
        ("Cheddar's\tScratch   Kitchen", "cheddars scratch kitchen"),
        # Trademark symbols are stripped; accented letters are word chars (kept).
        ("Panda Express™", "panda express"),
        ("Café Rio®", "café rio"),
    ],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert C.normalize_name(raw) == expected


def test_normalize_name_collapses_whitespace() -> None:
    assert C.normalize_name("  a    b   c ") == "a b c"


def test_canonical_chain_is_normalize_by_default() -> None:
    # The variant map ships empty to reproduce the EDA exactly, so canonical_chain
    # equals normalize_name. Punctuation and hyphens are stripped; distinct EDA
    # names stay distinct (QDOBA Mexican Eats != QDOBA Mexican Grill).
    assert C.canonical_chain("Chick-fil-A") == "chickfila"
    assert C.canonical_chain("Chick Fil A") == "chick fil a"
    assert C.canonical_chain("QDOBA Mexican Eats") == "qdoba mexican eats"
    assert C.canonical_chain("Qdoba Mexican Grill") == "qdoba mexican grill"


def test_variant_map_is_empty_for_eda_fidelity() -> None:
    # Guard against reintroducing variant collapsing that breaks EDA reconciliation.
    assert config.CHAIN_VARIANT_MAP == {}


def test_canonical_chain_applies_variant_map_when_present() -> None:
    # If a variant entry is added later it must take effect. We patch locally.
    original = dict(config.CHAIN_VARIANT_MAP)
    try:
        config.CHAIN_VARIANT_MAP["qdoba mexican grill"] = "qdoba mexican eats"
        assert C.canonical_chain("Qdoba Mexican Grill") == "qdoba mexican eats"
    finally:
        config.CHAIN_VARIANT_MAP.clear()
        config.CHAIN_VARIANT_MAP.update(original)


# --------------------------------------------------------------------------- #
# Category predicates
# --------------------------------------------------------------------------- #


def test_category_predicates() -> None:
    assert C.is_restaurant("Burgers, Restaurants, Fast Food")
    assert not C.is_restaurant("Coffee & Tea")
    assert C.is_cafe("Cafes, Restaurants")
    assert C.is_cafe("Coffee & Tea, Food")
    assert not C.is_cafe("Burgers, Restaurants")
    assert C.is_fast_food_location("Fast Food, Restaurants")
    assert not C.is_fast_food_location("Steakhouses, Restaurants")
    assert not C.is_restaurant(None)


# --------------------------------------------------------------------------- #
# Synthetic cohort fixture
# --------------------------------------------------------------------------- #

# Chains we want selected: all 30 approved display names, each given enough
# locations (> 3) and stars (> 2.5) to pass, with review counts that rank them
# into the top 15 per cluster. Fast Food status is set via the categories string.


def _approved_specs() -> list[tuple[str, bool]]:
    ff = [(name, True) for name in config.COHORT_FAST_FOOD]
    nff = [(name, False) for name in config.COHORT_NON_FAST_FOOD]
    return ff + nff


def _make_business_fixture(include_decoys: bool = True) -> pd.DataFrame:
    """Build a synthetic business table that yields the approved cohort.

    Each approved chain gets 5 locations (> 3), stars 4.0 (> 2.5), and a high
    review_count. Decoy chains are added that must be excluded: too few
    locations, low stars, cafes, and non-restaurants, plus extra eligible chains
    with lower review counts that must lose the top-15 ranking.
    """
    rows: list[dict] = []
    bid = 0

    def add(
        name: str, categories: str, stars: float, review_count: int, n_loc: int, is_open: int = 1
    ) -> None:
        nonlocal bid
        for _ in range(n_loc):
            rows.append(
                {
                    "business_id": f"b{bid}",
                    "name": name,
                    "categories": categories,
                    "stars": stars,
                    "review_count": review_count,
                    "is_open": is_open,
                    "city": "Phoenix",
                    "state": "AZ",
                    "latitude": 33.4,
                    "longitude": -112.0,
                }
            )
            bid += 1

    for name, is_ff in _approved_specs():
        cats = "Restaurants, Fast Food" if is_ff else "Restaurants, Steakhouses"
        add(name, cats, stars=4.0, review_count=1000, n_loc=5)

    if include_decoys:
        # Eligible but lower review count -> must lose the top-15 ranking.
        add("Loser Fast Food Co", "Restaurants, Fast Food", 4.0, 10, 5)
        add("Loser Diner Co", "Restaurants, Steakhouses", 4.0, 10, 5)
        # Too few locations.
        add("Tiny Grill", "Restaurants, Steakhouses", 4.5, 999, 2)
        # Stars too low.
        add("Lowrated Burgers", "Restaurants, Fast Food", 2.0, 999, 9)
        # Cafe -> excluded before grouping.
        add("Corner Cafe", "Cafes, Coffee & Tea, Restaurants", 4.5, 999, 9)
        # Non-restaurant -> excluded.
        add("Hardware Store", "Shopping, Home Services", 4.5, 999, 9)

    return pd.DataFrame(rows)


def test_select_cohort_reproduces_approved_30() -> None:
    business = C.clean_business(_make_business_fixture())
    rest = C.prepare_restaurant_chains(business)
    summary = C.build_chain_summary(rest)
    cohort = C.select_cohort(summary)

    # Assertion must pass against the approved constants.
    C.assert_cohort(cohort)

    assert len(cohort) == config.COHORT_SIZE == 30
    split = cohort["cluster"].value_counts().to_dict()
    assert split["Fast Food"] == 15
    assert split["Non-Fast Food"] == 15

    got_ff = set(cohort.loc[cohort.cluster == "Fast Food", "display_name"])
    assert got_ff == set(config.COHORT_FAST_FOOD)


def test_cafes_and_nonrestaurants_excluded() -> None:
    business = C.clean_business(_make_business_fixture())
    rest = C.prepare_restaurant_chains(business)
    chains = set(rest["chain"])
    assert C.canonical_chain("Corner Cafe") not in chains
    assert C.canonical_chain("Hardware Store") not in chains


def test_assert_cohort_rejects_wrong_split() -> None:
    # Build a cohort DataFrame with a broken 16/14 split.
    rows = []
    for name in config.COHORT_FAST_FOOD:
        rows.append({"display_name": name, "cluster": "Fast Food"})
    # Move one Non-Fast Food chain into Fast Food -> 16/14.
    nff = list(config.COHORT_NON_FAST_FOOD)
    rows.append({"display_name": nff[0], "cluster": "Fast Food"})
    for name in nff[1:]:
        rows.append({"display_name": name, "cluster": "Non-Fast Food"})
    bad = pd.DataFrame(rows)
    with pytest.raises(AssertionError):
        C.assert_cohort(bad)


def test_assert_cohort_rejects_wrong_names() -> None:
    rows = []
    for name in config.COHORT_FAST_FOOD:
        rows.append({"display_name": name, "cluster": "Fast Food"})
    nff = list(config.COHORT_NON_FAST_FOOD)
    # Replace one approved name with an impostor -> set mismatch.
    nff[0] = "Impostor Grill"
    for name in nff:
        rows.append({"display_name": name, "cluster": "Non-Fast Food"})
    bad = pd.DataFrame(rows)
    with pytest.raises(AssertionError):
        C.assert_cohort(bad)


# --------------------------------------------------------------------------- #
# Check-in explode and data quality
# --------------------------------------------------------------------------- #


def test_explode_checkins() -> None:
    checkins = pd.DataFrame(
        {
            "business_id": ["b0", "b1", "b2"],
            "date": [
                "2019-01-01 10:00:00, 2019-01-02 11:00:00",
                "",
                None,
            ],
        }
    )
    exploded = C.explode_checkins(checkins)
    assert list(exploded.columns) == ["business_id", "checkin_at"]
    # b0 contributes 2 rows; b1 (empty) and b2 (null) contribute none.
    assert len(exploded) == 2
    assert set(exploded["business_id"]) == {"b0"}


def test_data_quality_flags_zero_engagement_without_imputing() -> None:
    cohort_ids = {"b0", "b1", "b2"}
    reviews = pd.DataFrame(
        {
            "review_id": ["r0", "r1", "r1"],  # one duplicate review id
            "business_id": ["b0", "b0", "b0"],
            "stars": [5, 4, 4],
            "date": ["2019-01-01", "2020-06-01", "2020-06-01"],
            "text": ["great", "  ", "ok"],  # one empty text
        }
    )
    tips = pd.DataFrame({"business_id": ["b0"], "text": ["tip"], "date": ["2019-01-01"]})
    checkins = pd.DataFrame({"business_id": ["b0"], "checkin_at": pd.to_datetime(["2019-01-01"])})

    report = C.data_quality_report(
        business=pd.DataFrame({"business_id": ["b0", "b0"]}),
        reviews=reviews,
        tips=tips,
        checkins_exploded=checkins,
        cohort_business_ids=cohort_ids,
    )
    assert report["duplicate_business_ids"] == 1
    assert report["duplicate_review_ids"] == 1
    assert report["null_or_empty_review_text"] == 1
    assert report["review_date_min"] == "2019-01-01"
    assert report["review_date_max"] == "2020-06-01"
    # b1 and b2 have no checkins or tips -> flagged, not imputed.
    assert report["n_locations_zero_checkins"] == 2
    assert report["n_locations_zero_tips"] == 2
    assert set(report["locations_zero_tips"]) == {"b1", "b2"}
