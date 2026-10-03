"""Pipeline stage 04b: Tier 1 corpus features and labels.

Assembles the leakage-safe Tier 1 feature matrix for the restaurant universe at a
landmark T from the monthly activity panel and the monthly sentiment aggregates, and
labels each restaurant with the shared event/duration logic. No topic features at
this tier.

Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/04b_tier1_features.py --landmark 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config, corpus, features, io  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tier 1 corpus features.")
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument(
        "--all-restaurants", type=Path, default=config.DATA_DIR / config.ALL_RESTAURANTS_FILE
    )
    p.add_argument("--panel", type=Path, default=config.DATA_DIR / config.ACTIVITY_MONTHLY_FILE)
    p.add_argument(
        "--sentiment-monthly", type=Path, default=config.DATA_DIR / config.SENTIMENT_MONTHLY_FILE
    )
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _read_optional(path):
    path = Path(path)
    return io.read_parquet(path) if path.exists() else pd.DataFrame()


def run(args):
    T = pd.Timestamp(args.landmark)
    all_rest = io.read_parquet(args.all_restaurants)
    panel = io.read_parquet(args.panel)
    sent_monthly = _read_optional(args.sentiment_monthly)

    # Pre-T window features from the panel (strictly before T inside the reducer).
    panel_feats = corpus.panel_window_features(panel, T)

    # Proxy closure from the panel's last active month per business (approximate
    # last activity when the raw per-event dates are not reloaded here).
    pm = panel.copy()
    pm["month_ts"] = pd.to_datetime(pm["month"], errors="coerce")
    last_activity = pm.groupby("business_id")["month_ts"].max()

    labels = features.tier1_labels(all_rest, last_activity, T)
    feats = features.tier1_features(all_rest, panel_feats, sent_monthly, T)
    merged = feats.merge(
        labels[["business_id", "event", "duration_days"]], on="business_id", how="inner"
    )
    merged["group"] = features.group_key(merged)
    print(
        f"Tier 1 matrix: {len(merged):,} restaurants | "
        f"events={int(merged['event'].sum())} | groups={merged['group'].nunique()}"
    )

    if args.dry_run:
        print("[dry-run] No files written.")
        return merged

    out = Path(args.data_dir) / config.TIER1_FEATURES_FILE
    io.write_parquet(merged, out)
    print(f"Wrote Tier 1 features to {out}.")
    return merged


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
