"""Pipeline stage 02c: corpus sentiment scorer with a transfer test.

Trains the TF-IDF + calibrated linear scorer on a stratified sample of up to 1M
NON-cohort reviews (business-grouped split), evaluates on a held-out non-cohort test
set and on all cohort reviews as an out-of-sample transfer test (both with bootstrap
CIs), scores all cohort reviews, and streams monthly sentiment aggregates for every
restaurant (storing aggregates only, never text).

Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/02c_corpus_sentiment.py --yelp-dir "Yelp-JSON/Yelp JSON/yelp_dataset"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config, io, sentiment  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Corpus sentiment + transfer test.")
    p.add_argument("--yelp-dir", type=Path, default=config.YELP_DIR)
    p.add_argument(
        "--all-restaurants", type=Path, default=config.DATA_DIR / config.ALL_RESTAURANTS_FILE
    )
    p.add_argument(
        "--cohort-reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE
    )
    p.add_argument("--sample-size", type=int, default=config.NONCOHORT_SAMPLE_SIZE)
    p.add_argument("--chunksize", type=int, default=config.CORPUS_CHUNKSIZE)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _review_chunks(yelp_dir, chunksize):
    batch = []
    for rec in io.iter_yelp_records("review", yelp_dir):
        batch.append(
            {
                "review_id": rec.get("review_id"),
                "business_id": rec.get("business_id"),
                "stars": rec.get("stars"),
                "date": rec.get("date"),
                "text": rec.get("text"),
            }
        )
        if len(batch) >= chunksize:
            yield pd.DataFrame(batch)
            batch = []
    if batch:
        yield pd.DataFrame(batch)


def run(args):
    all_rest = io.read_parquet(args.all_restaurants)
    yelp = Path(args.yelp_dir)

    # Build the non-cohort training sample by streaming reviews and keeping only
    # labeled rows from non-cohort restaurants (bounded by the sample size).
    noncohort_ids = set(all_rest.loc[~all_rest["is_cohort"], "business_id"].astype(str))
    kept = []
    for chunk in _review_chunks(yelp, args.chunksize):
        sub = chunk[chunk["business_id"].astype(str).isin(noncohort_ids)]
        if not sub.empty:
            kept.append(sentiment.map_polarity_labels(sub))
        if sum(len(k) for k in kept) >= args.sample_size * 2:
            break
    labeled = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    if len(labeled) > args.sample_size:
        labeled = labeled.sample(n=args.sample_size, random_state=config.SEED)
    print(f"Non-cohort labeled sample: {len(labeled):,}")

    train, test = sentiment.grouped_split(labeled)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"]))

    bench = sentiment.benchmark(train, test, include_vader=False)
    prod = sentiment.pick_production(bench)
    scorer = sentiment.calibrate_scorer(train, prod["features"], prod["model"])
    print(
        f"Production scorer: {prod['features']} + {prod['model']} (macro F1 {prod['macro_f1']:.4f})"
    )

    cohort_reviews = io.read_parquet(args.cohort_reviews)
    transfer = sentiment.transfer_evaluate(scorer, test, cohort_reviews)
    print("\nTRANSFER EVALUATION")
    print("=" * 60)
    cols = [
        c
        for c in [
            "split",
            "accuracy",
            "macro_f1",
            "macro_f1_lo",
            "macro_f1_hi",
            "roc_auc",
            "n_test",
        ]
        if c in transfer.columns
    ]
    print(transfer[cols].round(4).to_string(index=False))

    cohort_scored = sentiment.score_all(cohort_reviews, scorer)

    # Streaming monthly sentiment aggregates for ALL restaurants (aggregates only).
    rest_ids = set(all_rest["business_id"].astype(str))
    monthly = sentiment.score_reviews_streaming(
        _review_chunks(yelp, args.chunksize), scorer, rest_ids
    )
    print(f"Monthly sentiment aggregates: {len(monthly):,} business-months.")

    if args.dry_run:
        print("[dry-run] No files written.")
        return {"transfer": transfer, "monthly": monthly}

    art, data = Path(args.artifacts_dir), Path(args.data_dir)
    io.write_parquet(transfer, art / config.TRANSFER_EVAL_FILE)
    io.write_parquet(cohort_scored, data / config.REVIEW_SENTIMENT_FILE)
    io.write_parquet(monthly, data / config.SENTIMENT_MONTHLY_FILE)
    print("Wrote transfer eval, cohort sentiment, and monthly sentiment aggregates.")
    return {"transfer": transfer, "monthly": monthly, "scorer": scorer}


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
