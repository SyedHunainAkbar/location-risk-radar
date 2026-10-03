"""Streaming IO for the Yelp Open Dataset and parquet helpers.

We read the four Yelp JSON-lines files from ``YELP_DIR`` whether they sit there as
plain ``.json`` files or inside the distributed ``.tar`` archive, without extracting
the archive. Readers stream line by line so memory stays flat on the multi-GB
review file.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import IO

import pandas as pd

#: The four Yelp datasets, by short name.
YELP_DATASETS: tuple[str, ...] = ("business", "review", "checkin", "tip")


def dataset_filename(dataset: str) -> str:
    """Return the canonical JSON-lines filename for a dataset short name."""
    return f"yelp_academic_dataset_{dataset}.json"


def _find_plain_json(dataset: str, yelp_dir: Path) -> Path | None:
    candidate = yelp_dir / dataset_filename(dataset)
    return candidate if candidate.is_file() else None


def _find_tar(yelp_dir: Path) -> Path | None:
    """Return the first ``.tar`` archive in ``yelp_dir``, if any."""
    if not yelp_dir.is_dir():
        return None
    tars = sorted(p for p in yelp_dir.glob("*.tar") if p.is_file())
    return tars[0] if tars else None


def _tar_member_name(tar: tarfile.TarFile, dataset: str) -> str | None:
    """Find the archive member whose basename matches the dataset filename."""
    target = dataset_filename(dataset)
    for name in tar.getnames():
        if Path(name).name == target:
            return name
    return None


def _iter_json_lines(handle: Iterable[bytes | str]) -> Iterator[dict]:
    """Yield parsed JSON objects from an iterable of lines, skipping blanks."""
    for raw in handle:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def iter_yelp_records(dataset: str, yelp_dir: Path) -> Iterator[dict]:
    """Stream parsed records for a Yelp dataset.

    Resolution order:

    1. A plain ``yelp_academic_dataset_<dataset>.json`` in ``yelp_dir``.
    2. The matching member inside the first ``.tar`` archive in ``yelp_dir``.

    Args:
        dataset: One of :data:`YELP_DATASETS`.
        yelp_dir: Directory holding the dataset or the ``.tar`` archive.

    Yields:
        Parsed JSON records as dicts.

    Raises:
        ValueError: If ``dataset`` is unknown.
        FileNotFoundError: If the dataset is in neither a plain file nor a tar.
    """
    if dataset not in YELP_DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of {YELP_DATASETS}.")
    yelp_dir = Path(yelp_dir)

    plain = _find_plain_json(dataset, yelp_dir)
    if plain is not None:
        with plain.open("r", encoding="utf-8") as fh:
            yield from _iter_json_lines(fh)
        return

    tar_path = _find_tar(yelp_dir)
    if tar_path is not None:
        with tarfile.open(tar_path, "r") as tar:
            member = _tar_member_name(tar, dataset)
            if member is not None:
                extracted: IO[bytes] | None = tar.extractfile(member)
                if extracted is not None:
                    with extracted:
                        yield from _iter_json_lines(extracted)
                    return

    raise FileNotFoundError(
        f"Could not find dataset {dataset!r} as "
        f"{dataset_filename(dataset)!r} or inside a .tar in {yelp_dir}."
    )


def read_yelp_frame(
    dataset: str,
    yelp_dir: Path,
    columns: Sequence[str] | None = None,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """Read a Yelp dataset into a DataFrame, batching to bound memory.

    Records are accumulated in chunks of ``chunksize`` and concatenated once. If
    ``columns`` is given, only those columns are retained per chunk.

    Args:
        dataset: One of :data:`YELP_DATASETS`.
        yelp_dir: Directory holding the dataset or the ``.tar`` archive.
        columns: Optional subset of columns to keep.
        chunksize: Records per intermediate batch.

    Returns:
        A single concatenated DataFrame (possibly empty).
    """
    frames: list[pd.DataFrame] = []
    batch: list[dict] = []

    def flush() -> None:
        if not batch:
            return
        frame = pd.DataFrame(batch)
        if columns is not None:
            keep = [c for c in columns if c in frame.columns]
            frame = frame[keep]
        frames.append(frame)
        batch.clear()

    for record in iter_yelp_records(dataset, yelp_dir):
        batch.append(record)
        if len(batch) >= chunksize:
            flush()
    flush()

    if not frames:
        return pd.DataFrame(columns=list(columns) if columns else None)
    return pd.concat(frames, ignore_index=True)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to parquet, creating parent directories.

    Returns the path written. Deterministic for identical inputs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a DataFrame."""
    return pd.read_parquet(Path(path), engine="pyarrow")
