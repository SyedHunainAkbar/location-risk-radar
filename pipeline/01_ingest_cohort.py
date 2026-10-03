"""Pipeline stage 01: ingest the Yelp dataset and build the locked cohort.

Reads the four Yelp files from ``YELP_DIR`` (plain ``.json`` or inside a ``.tar``),
reproduces the instructor-approved 30-chain cohort exactly as the EDA does, asserts
the 15/15 split and the approved names, writes the cohort artifacts, and prints a
reconciliation table and a data quality report.

Idempotent: rerunning with the same inputs overwrites the same artifacts. All
paths and constants come from ``lrr.config``.

Usage:
    python pipeline/01_ingest_cohort.py --yelp-dir ./data/yelp_dataset
    python pipeline/01_ingest_cohort.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when run as a script.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import cohort as cohort_mod  # noqa: E402
from lrr import config, io  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Yelp data and build cohort.")
    parser.add_argument(
        "--yelp-dir",
        type=Path,
        default=config.YELP_DIR,
        help="Directory with Yelp .json files or a .tar archive.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=config.ARTIFACTS_DIR,
        help="Where committed small artifacts are written.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.DATA_DIR,
        help="Where large (gitignored) parquet artifacts are written.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and assert the cohort but write no files.",
    )
    return parser.parse_args(argv)


def _seed_everything() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def _print_reconciliation(table: pd.DataFrame) -> None:
    print("\nRECONCILIATION VS EDA (locations, reviews)")
    print("=" * 72)
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(table.to_string(index=False))
    n_match = int(table["matches"].sum())
    print(f"\nChains matching EDA exactly: {n_match}/{len(table)}")


def _print_dq(report: dict) -> None:
    print("\nDATA QUALITY REPORT")
    print("=" * 72)
    for key in (
        "duplicate_business_ids",
        "duplicate_review_ids",
        "duplicate_tip_rows",
        "null_or_empty_review_text",
        "review_date_min",
        "review_date_max",
        "n_locations_zero_checkins",
        "n_locations_zero_tips",
    ):
        print(f"  {key:<30}: {report[key]}")
    if report["n_locations_zero_checkins"]:
        print(
            f"  (flagged, not imputed) zero-checkin ids: "
            f"{report['locations_zero_checkins'][:10]}"
            f"{' ...' if report['n_locations_zero_checkins'] > 10 else ''}"
        )
    if report["n_locations_zero_tips"]:
        print(
            f"  (flagged, not imputed) zero-tip ids: "
            f"{report['locations_zero_tips'][:10]}"
            f"{' ...' if report['n_locations_zero_tips'] > 10 else ''}"
        )


def build_cohort_locations(rest_df: pd.DataFrame, cohort: pd.DataFrame) -> pd.DataFrame:
    """Select cohort locations and project the artifact schema."""
    chain_to_display = dict(zip(cohort["chain"], cohort["display_name"]))
    chain_to_cluster = dict(zip(cohort["chain"], cohort["cluster"]))
    loc = rest_df[rest_df["chain"].isin(chain_to_display)].copy()
    loc["chain"] = loc["chain"].map(chain_to_display)
    loc["cluster"] = rest_df.loc[loc.index, "chain"].map(chain_to_cluster)
    out = pd.DataFrame(
        {
            "business_id": loc["business_id"],
            "chain": loc["chain"],
            "cluster": loc["cluster"],
            "city": loc.get("city"),
            "state": loc.get("state"),
            "lat": loc.get("latitude"),
            "lon": loc.get("longitude"),
            "stars": loc.get("stars"),
            "review_count": loc.get("review_count"),
            "is_open": loc.get("is_open"),
        }
    )
    return out.drop_duplicates(subset="business_id").reset_index(drop=True)


def _project_reviews(reviews: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    cols = ["review_id", "business_id", "stars", "date", "text", "useful", "funny", "cool"]
    out = reviews[reviews["business_id"].isin(ids)].copy()
    for col in cols:
        if col not in out:
            out[col] = pd.NA
    out = out[cols]
    out = out.dropna(subset=["business_id", "text", "stars"])
    out["stars"] = pd.to_numeric(out["stars"], errors="coerce")
    out = out[out["stars"].between(1, 5)]
    out["text"] = out["text"].astype("string").str.strip()
    out = out[out["text"].str.len() > 0]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.reset_index(drop=True)


def _project_tips(tips: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    if tips.empty:
        return tips
    out = tips[tips["business_id"].isin(ids)].copy()
    if "date" in out:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    """Execute the stage. Returns a summary dict for programmatic callers."""
    _seed_everything()
    yelp_dir = Path(args.yelp_dir)

    print(f"Reading business from {yelp_dir} ...")
    business_raw = io.read_yelp_frame("business", yelp_dir)
    business = cohort_mod.clean_business(business_raw)
    rest_df = cohort_mod.prepare_restaurant_chains(business)

    chain_summary = cohort_mod.build_chain_summary(rest_df)
    cohort = cohort_mod.select_cohort(chain_summary)
    cohort_mod.assert_cohort(cohort)
    print(f"Cohort assertion passed: {len(cohort)} chains (15 + 15).")

    locations = build_cohort_locations(rest_df, cohort)
    cohort_ids = set(locations["business_id"])

    print("Reading reviews (streamed) ...")
    reviews_raw = io.read_yelp_frame("review", yelp_dir)
    reviews = _project_reviews(reviews_raw, cohort_ids)

    print("Reading tips ...")
    tips_raw = io.read_yelp_frame("tip", yelp_dir)
    tips = _project_tips(tips_raw, cohort_ids)

    print("Reading checkins ...")
    checkins_raw = io.read_yelp_frame("checkin", yelp_dir)
    checkins = cohort_mod.explode_checkins(
        checkins_raw[checkins_raw["business_id"].isin(cohort_ids)]
        if not checkins_raw.empty
        else checkins_raw
    )

    recon = cohort_mod.reconciliation_table(cohort)
    dq = cohort_mod.data_quality_report(
        business=business_raw,
        reviews=reviews,
        tips=tips,
        checkins_exploded=checkins,
        cohort_business_ids=cohort_ids,
    )

    _print_reconciliation(recon)
    _print_dq(dq)

    print(
        f"\nTotals: locations={len(locations)}, "
        f"open={int(locations['is_open'].sum())}, "
        f"closed={int((locations['is_open'] == 0).sum())}"
    )

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return {"cohort": cohort, "locations": locations, "dq": dq}

    art = Path(args.artifacts_dir)
    data = Path(args.data_dir)
    io.write_parquet(locations, art / config.COHORT_LOCATIONS_FILE)
    io.write_parquet(reviews, data / config.COHORT_REVIEWS_FILE)
    io.write_parquet(tips, data / config.COHORT_TIPS_FILE)
    io.write_parquet(checkins, data / config.COHORT_CHECKINS_FILE)
    print(f"\nWrote artifacts to {art} and {data}.")

    return {"cohort": cohort, "locations": locations, "dq": dq}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
