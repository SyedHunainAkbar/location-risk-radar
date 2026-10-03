"""Integration test for pipeline/01_ingest_cohort.py on a synthetic dataset.

Builds tiny Yelp JSON-lines files in a temp dir, runs the stage, and checks that
the cohort asserts, artifacts are written with the right schema, and the run is
idempotent. No real data or network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd

from lrr import config, io

_PIPELINE = Path(__file__).resolve().parents[1] / "pipeline" / "01_ingest_cohort.py"


def _load_stage():
    spec = importlib.util.spec_from_file_location("stage01", _PIPELINE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_yelp_dir(tmp_path: Path) -> Path:
    yelp = tmp_path / "yelp"
    yelp.mkdir(parents=True, exist_ok=True)

    business: list[dict] = []
    reviews: list[dict] = []
    tips: list[dict] = []
    checkins: list[dict] = []
    bid = 0

    specs = [(n, True) for n in config.COHORT_FAST_FOOD] + [
        (n, False) for n in config.COHORT_NON_FAST_FOOD
    ]
    for name, is_ff in specs:
        cats = "Restaurants, Fast Food" if is_ff else "Restaurants, Steakhouses"
        for _ in range(5):
            bids = f"b{bid}"
            business.append(
                {
                    "business_id": bids,
                    "name": name,
                    "categories": cats,
                    "stars": 4.0,
                    "review_count": 1000,
                    "is_open": 1,
                    "city": "Phoenix",
                    "state": "AZ",
                    "latitude": 33.4,
                    "longitude": -112.0,
                }
            )
            # one review per location
            reviews.append(
                {
                    "review_id": f"r{bid}",
                    "business_id": bids,
                    "stars": 5,
                    "date": "2019-05-01",
                    "text": "good food here",
                    "useful": 1,
                    "funny": 0,
                    "cool": 0,
                }
            )
            # tips/checkins only for the very first business, to exercise flags
            if bid == 0:
                tips.append({"business_id": bids, "text": "try the fries", "date": "2019-05-01"})
                checkins.append(
                    {"business_id": bids, "date": "2019-05-01 12:00:00, 2019-05-02 13:00:00"}
                )
            bid += 1

    _write_jsonl(yelp / io.dataset_filename("business"), business)
    _write_jsonl(yelp / io.dataset_filename("review"), reviews)
    _write_jsonl(yelp / io.dataset_filename("tip"), tips)
    _write_jsonl(yelp / io.dataset_filename("checkin"), checkins)
    return yelp


def _args(yelp: Path, art: Path, data: Path, dry: bool = False) -> argparse.Namespace:
    return argparse.Namespace(yelp_dir=yelp, artifacts_dir=art, data_dir=data, dry_run=dry)


def test_pipeline_asserts_and_writes(tmp_path: Path) -> None:
    stage = _load_stage()
    yelp = _make_yelp_dir(tmp_path)
    art = tmp_path / "artifacts"
    data = tmp_path / "data"

    result = stage.run(_args(yelp, art, data))

    # Cohort asserted inside run(); confirm shape here too.
    assert len(result["cohort"]) == 30

    loc_path = art / config.COHORT_LOCATIONS_FILE
    assert loc_path.exists()
    loc = io.read_parquet(loc_path)
    assert list(loc.columns) == [
        "business_id",
        "chain",
        "cluster",
        "city",
        "state",
        "lat",
        "lon",
        "stars",
        "review_count",
        "is_open",
    ]
    assert len(loc) == 150  # 30 chains x 5 locations

    reviews = io.read_parquet(data / config.COHORT_REVIEWS_FILE)
    assert list(reviews.columns) == [
        "review_id",
        "business_id",
        "stars",
        "date",
        "text",
        "useful",
        "funny",
        "cool",
    ]
    assert len(reviews) == 150

    checkins = io.read_parquet(data / config.COHORT_CHECKINS_FILE)
    assert list(checkins.columns) == ["business_id", "checkin_at"]
    assert len(checkins) == 2  # exploded from the single checkin string

    # Data quality: only one location has tips/checkins -> 149 flagged.
    assert result["dq"]["n_locations_zero_tips"] == 149
    assert result["dq"]["n_locations_zero_checkins"] == 149


def test_pipeline_is_idempotent(tmp_path: Path) -> None:
    stage = _load_stage()
    yelp = _make_yelp_dir(tmp_path)
    art = tmp_path / "artifacts"
    data = tmp_path / "data"

    stage.run(_args(yelp, art, data))
    first = io.read_parquet(art / config.COHORT_LOCATIONS_FILE)
    stage.run(_args(yelp, art, data))
    second = io.read_parquet(art / config.COHORT_LOCATIONS_FILE)
    pd.testing.assert_frame_equal(
        first.sort_values("business_id").reset_index(drop=True),
        second.sort_values("business_id").reset_index(drop=True),
    )


def test_pipeline_dry_run_writes_nothing(tmp_path: Path) -> None:
    stage = _load_stage()
    yelp = _make_yelp_dir(tmp_path)
    art = tmp_path / "artifacts"
    data = tmp_path / "data"
    stage.run(_args(yelp, art, data, dry=True))
    assert not (art / config.COHORT_LOCATIONS_FILE).exists()
