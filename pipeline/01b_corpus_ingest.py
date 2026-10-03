"""Pipeline stage 01b: streaming corpus ingestion.

Builds the restaurant universe (``data/all_restaurants.parquet``) and the monthly
activity panel (``data/activity_monthly.parquet``) over all restaurants, in a single
streaming pass that never holds the 7M review texts in memory at once. Logs wall-clock
time and peak memory.

Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/01b_corpus_ingest.py --yelp-dir "Yelp-JSON/Yelp JSON/yelp_dataset"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import cohort as cohort_mod
from lrr import config, corpus, io  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Streaming corpus ingestion.")
    p.add_argument("--yelp-dir", type=Path, default=config.YELP_DIR)
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument("--checkins", type=Path, default=config.DATA_DIR / config.COHORT_CHECKINS_FILE)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--chunksize", type=int, default=config.CORPUS_CHUNKSIZE)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _chunked(dataset, yelp_dir, cols, chunksize):
    """Yield DataFrame chunks of a Yelp dataset without materializing it whole."""
    batch = []
    for rec in io.iter_yelp_records(dataset, yelp_dir):
        batch.append({k: rec.get(k) for k in cols})
        if len(batch) >= chunksize:
            yield pd.DataFrame(batch)
            batch = []
    if batch:
        yield pd.DataFrame(batch)


def run(args):
    start = time.time()
    yelp = Path(args.yelp_dir)

    business = io.read_yelp_frame("business", yelp)
    clean = cohort_mod.clean_business(business)

    cohort_ids = []
    locpath = Path(args.locations)
    if locpath.exists():
        cohort_ids = io.read_parquet(locpath)["business_id"].astype(str).tolist()

    all_rest = corpus.build_all_restaurants(clean, cohort_ids)
    rest_ids = set(all_rest["business_id"].astype(str))
    print(
        f"Restaurant universe: {len(all_rest):,} "
        f"({int(all_rest['is_cohort'].sum())} cohort locations)."
    )

    # Streaming panel: reviews and tips as chunk iterables; check-ins as a frame
    # (already exploded and far smaller). Review text is never accumulated.
    review_chunks = _chunked("review", yelp, ["business_id", "stars", "date"], args.chunksize)
    tip_chunks = _chunked("tip", yelp, ["business_id", "date"], args.chunksize)
    checkins = None
    ckpath = Path(args.checkins)
    if ckpath.exists():
        checkins = io.read_parquet(ckpath)
    else:
        # Explode raw check-ins on the fly if the cohort artifact is absent.
        raw_ck = io.read_yelp_frame("checkin", yelp)
        checkins = cohort_mod.explode_checkins(raw_ck)

    panel = corpus.stream_activity_panel(review_chunks, tip_chunks, checkins, rest_ids, log=True)
    corpus.log_resources("total corpus ingest", start)

    if args.dry_run:
        print("[dry-run] No files written.")
        return {"all_restaurants": all_rest, "panel": panel}

    data = Path(args.data_dir)
    io.write_parquet(all_rest, data / config.ALL_RESTAURANTS_FILE)
    io.write_parquet(panel, data / config.ACTIVITY_MONTHLY_FILE)
    print(f"Wrote all_restaurants and activity_monthly ({len(panel):,} rows) to {data}.")
    return {"all_restaurants": all_rest, "panel": panel}


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
