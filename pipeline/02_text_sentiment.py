"""Pipeline stage 02: text lexicons (LA1) and calibrated sentiment (LA2).

Reads cohort reviews and locations, builds the descriptive noun/adjective lexicons
contrasting closed vs open and Fast Food vs Non-Fast Food, benchmarks a VADER
baseline against bag-of-words and TF-IDF models on a leakage-free grouped split
(bootstrap 95% CIs), selects the production scorer by macro F1, calibrates it to
emit P(positive), scores every cohort review including 3-star ones, and writes the
artifacts.

Online path only: spaCy small model, scikit-learn, VADER. No TensorFlow here.
Idempotent; all paths and constants come from ``lrr.config``.

Usage:
    python pipeline/02_text_sentiment.py
    python pipeline/02_text_sentiment.py --sample 50000 --skip-lexicons
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config, io, sentiment  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Text lexicons and sentiment scorer.")
    p.add_argument("--reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="If > 0, sample this many reviews (seeded) for a fast run.",
    )
    p.add_argument(
        "--skip-lexicons",
        action="store_true",
        help="Skip the spaCy lexicon step (useful if model missing).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _seed() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def load_reviews_with_context(reviews_path: Path, locations_path: Path) -> pd.DataFrame:
    """Join reviews to location cluster and derive closed/open status.

    ``is_open`` is a current status, not a closure date, so we label a review's
    location "closed" when ``is_open == 0`` and "open" otherwise. The downstream
    survival stage handles the proper time-to-event treatment; here we only need a
    descriptive contrast for the lexicons.
    """
    reviews = io.read_parquet(reviews_path)
    locations = io.read_parquet(locations_path)
    loc = locations[["business_id", "cluster", "is_open"]].copy()
    loc["closed_open"] = np.where(loc["is_open"] == 0, "closed", "open")
    merged = reviews.merge(
        loc[["business_id", "cluster", "closed_open"]], on="business_id", how="inner"
    )
    return merged


def build_and_write_lexicons(reviews_ctx: pd.DataFrame, artifacts_dir: Path) -> None:
    """Build the four lexicon tables via spaCy and write them to parquet."""
    from lrr import text  # local import so a missing model does not block LA2

    nlp = text.load_nlp()
    tables = text.contrast_lexicons(reviews_ctx, nlp)
    name_to_file = {
        "lexicon_closed_open_noun": config.LEXICON_CLOSED_OPEN_NOUN_FILE,
        "lexicon_closed_open_adj": config.LEXICON_CLOSED_OPEN_ADJ_FILE,
        "lexicon_cluster_noun": config.LEXICON_CLUSTER_NOUN_FILE,
        "lexicon_cluster_adj": config.LEXICON_CLUSTER_ADJ_FILE,
    }
    for stem, table in tables.items():
        io.write_parquet(table, artifacts_dir / name_to_file[stem])
    print(f"Wrote 4 lexicon tables to {artifacts_dir}.")


def run(args: argparse.Namespace) -> dict:
    _seed()
    reviews_ctx = load_reviews_with_context(args.reviews, args.locations)
    if args.sample and args.sample < len(reviews_ctx):
        reviews_ctx = reviews_ctx.sample(n=args.sample, random_state=config.SEED).reset_index(
            drop=True
        )
    print(f"Reviews in scope: {len(reviews_ctx):,}")

    # LA1: lexicons
    if not args.skip_lexicons and not args.dry_run:
        try:
            build_and_write_lexicons(reviews_ctx, Path(args.artifacts_dir))
        except OSError as exc:
            print(f"[warn] spaCy model unavailable, skipping lexicons: {exc}")

    # LA2: label, grouped split, benchmark
    labeled = sentiment.map_polarity_labels(reviews_ctx)
    print(
        f"Labeled (3-star dropped): {len(labeled):,} "
        f"(pos={int(labeled.label.sum())}, neg={int((labeled.label == 0).sum())})"
    )
    train, test = sentiment.grouped_split(labeled)
    assert set(train["business_id"]).isdisjoint(set(test["business_id"])), (
        "Grouped split leaked a business across folds."
    )

    bench = sentiment.benchmark(train, test)
    prod = sentiment.pick_production(bench)
    print("\nSENTIMENT BENCHMARK (LA2)")
    print("=" * 72)
    with pd.option_context("display.width", 140, "display.max_columns", None):
        print(bench.round(4).to_string(index=False))
    print(
        f"\nProduction scorer (max macro F1): {prod['features']} + {prod['model']} "
        f"(macro F1 = {prod['macro_f1']:.4f})"
    )

    # Calibrate on train and score ALL reviews (including 3-star)
    scorer = sentiment.calibrate_scorer(train, prod["features"], prod["model"])
    scored = sentiment.score_all(reviews_ctx, scorer)
    print(
        f"Scored all reviews: {len(scored):,} (mean P(positive) = {scored.p_positive.mean():.3f})"
    )

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return {"benchmark": bench, "production": prod, "scored": scored}

    art = Path(args.artifacts_dir)
    data = Path(args.data_dir)
    io.write_parquet(bench, art / config.SENTIMENT_BENCHMARK_FILE)
    io.write_parquet(scored, data / config.REVIEW_SENTIMENT_FILE)
    print(f"\nWrote benchmark to {art} and review sentiment to {data}.")
    return {"benchmark": bench, "production": prod, "scored": scored}


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
