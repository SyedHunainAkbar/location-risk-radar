"""Tests for lrr.io streaming readers (plain .json and .tar)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from lrr import io


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
        fh.write("\n")  # trailing blank line, must be skipped


def test_iter_records_from_plain_json(tmp_path: Path) -> None:
    recs = [{"business_id": "b0", "name": "A"}, {"business_id": "b1", "name": "B"}]
    _write_jsonl(tmp_path / io.dataset_filename("business"), recs)
    out = list(io.iter_yelp_records("business", tmp_path))
    assert out == recs


def test_iter_records_from_tar(tmp_path: Path) -> None:
    recs = [{"business_id": "b0"}, {"business_id": "b1"}]
    inner = tmp_path / "inner.json"
    _write_jsonl(inner, recs)
    tar_path = tmp_path / "yelp_dataset.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(inner, arcname=io.dataset_filename("checkin"))
    inner.unlink()  # remove plain file so only the tar remains
    out = list(io.iter_yelp_records("checkin", tmp_path))
    assert out == recs


def test_plain_json_preferred_over_tar(tmp_path: Path) -> None:
    plain = [{"business_id": "plain"}]
    tarred = [{"business_id": "tarred"}]
    _write_jsonl(tmp_path / io.dataset_filename("tip"), plain)
    inner = tmp_path / "inner.json"
    _write_jsonl(inner, tarred)
    with tarfile.open(tmp_path / "data.tar", "w") as tar:
        tar.add(inner, arcname=io.dataset_filename("tip"))
    out = list(io.iter_yelp_records("tip", tmp_path))
    assert out == plain


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(io.iter_yelp_records("review", tmp_path))


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        list(io.iter_yelp_records("not_a_dataset", tmp_path))


def test_read_yelp_frame_chunked(tmp_path: Path) -> None:
    recs = [{"business_id": f"b{i}", "stars": i % 5 + 1} for i in range(10)]
    _write_jsonl(tmp_path / io.dataset_filename("business"), recs)
    df = io.read_yelp_frame("business", tmp_path, columns=["business_id"], chunksize=3)
    assert list(df.columns) == ["business_id"]
    assert len(df) == 10


def test_parquet_roundtrip(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = io.write_parquet(df, tmp_path / "sub" / "t.parquet")
    assert path.exists()
    back = io.read_parquet(path)
    pd.testing.assert_frame_equal(df, back)
