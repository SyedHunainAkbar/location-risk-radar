"""Pipeline stage 04: build the leakage-safe landmark feature matrix.

For a landmark ``T`` we select the eligible population, label the event and
duration from the proxy closure date, and engineer features from records dated
strictly before ``T`` only. Every activity source passes an explicit leakage
assertion before use. The output feeds stage 05.

Idempotent; all paths and constants come from ``lrr.config``.

Usage:
    python pipeline/04_features.py --landmark 2018-01-01
    python pipeline/04_features.py --landmark 2019-01-01   # sensitivity
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

from sklearn.model_selection import GroupKFold  # noqa: E402

from lrr import config, features, io  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build landmark survival features.")
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument("--reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    p.add_argument("--tips", type=Path, default=config.DATA_DIR / config.COHORT_TIPS_FILE)
    p.add_argument("--checkins", type=Path, default=config.DATA_DIR / config.COHORT_CHECKINS_FILE)
    p.add_argument("--sentiment", type=Path, default=config.DATA_DIR / config.REVIEW_SENTIMENT_FILE)
    p.add_argument("--shares", type=Path, default=config.DATA_DIR / config.TOPIC_SHARES_FILE)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _seed() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def _read_optional(path: Path) -> pd.DataFrame:
    path = Path(path)
    return io.read_parquet(path) if path.exists() else pd.DataFrame()


def run(args: argparse.Namespace) -> pd.DataFrame:
    _seed()
    T = pd.Timestamp(args.landmark)

    locations = io.read_parquet(args.locations)
    reviews = io.read_parquet(args.reviews)
    tips = _read_optional(args.tips)
    checkins = _read_optional(args.checkins)
    sentiment = _read_optional(args.sentiment)
    shares = _read_optional(args.shares)

    # Proxy closure uses all activity (outcome, may look after T).
    last_activity = features.last_activity_date(reviews, tips, checkins)

    # Population and labels.
    population = features.landmark_population(reviews, tips, checkins, locations, T)
    labeled = features.label_event_duration(population, last_activity, T)
    print(
        f"Landmark T={T.date()} | population={len(population)} | "
        f"events={int(labeled['event'].sum())} | "
        f"censored={int((labeled['event'] == 0).sum())}"
    )

    # Leakage guard: every feature source must be strictly pre-T.
    pre_reviews = reviews.copy()
    pre_reviews["date"] = pd.to_datetime(pre_reviews["date"], errors="coerce")
    features.assert_no_leakage(pre_reviews[pre_reviews["date"] < T], T)
    if not tips.empty:
        pt = tips.copy()
        pt["date"] = pd.to_datetime(pt["date"], errors="coerce")
        features.assert_no_leakage(pt[pt["date"] < T], T)
    if not checkins.empty:
        pc = checkins.copy()
        pc["checkin_at"] = pd.to_datetime(pc["checkin_at"], errors="coerce")
        features.assert_no_leakage(
            pc[pc["checkin_at"] < T].rename(columns={"checkin_at": "date"}), T
        )

    # Features (strictly pre-T inside build_features).
    feats = features.build_features(
        reviews=reviews,
        tips=tips,
        checkins=checkins,
        population=population,
        T=T,
        sentiment=sentiment if not sentiment.empty else None,
        topic_shares=shares if not shares.empty else None,
    )

    # Join labels.
    merged = feats.merge(
        labeled[["business_id", "event", "duration_days", "proxy_closure"]],
        on="business_id",
        how="inner",
    )

    # Assign chain folds (GroupKFold) and compute out-of-fold chain priors.
    merged = _assign_folds(merged)
    merged["chain_prior"] = features.out_of_fold_chain_priors(
        merged, target="event", group="chain", fold="fold"
    )

    # Final leakage self-check: the proxy closure (outcome) is the only post-T column.
    _assert_feature_leakage_safe(merged, T)

    if args.dry_run:
        print("[dry-run] No files written.")
        return merged

    out_path = Path(args.data_dir) / config.features_file(T)
    io.write_parquet(merged, out_path)
    print(f"Wrote {len(merged)} feature rows to {out_path}.")
    return merged


def _assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a GroupKFold fold id per row, grouped by chain."""
    out = df.copy()
    n_folds = min(config.N_FOLDS, out["chain"].nunique())
    out["fold"] = 0
    if n_folds < 2:
        return out
    gkf = GroupKFold(n_splits=n_folds)
    folds = np.zeros(len(out), dtype=int)
    for f, (_, test_idx) in enumerate(gkf.split(out, groups=out["chain"].astype(str))):
        folds[test_idx] = f
    out["fold"] = folds
    return out


def _assert_feature_leakage_safe(df: pd.DataFrame, T: pd.Timestamp) -> None:
    """Guard: no engineered feature column should encode a post-T timestamp.

    ``proxy_closure`` is the outcome (allowed to be >= T); everything else is a
    pre-T aggregate. We check there is no stray datetime feature leaking the future.
    """
    datetime_cols = [
        c
        for c in df.columns
        if c != "proxy_closure" and pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    assert not datetime_cols, (
        f"Unexpected datetime feature columns may leak post-T info: {datetime_cols}"
    )


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
