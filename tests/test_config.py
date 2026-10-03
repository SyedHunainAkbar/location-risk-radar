"""Data contract tests for lrr.config.

These guard the constants the rest of the pipeline relies on. If someone
changes the cohort math or the temporal design, these fail loudly.
"""

from __future__ import annotations

from datetime import date

from lrr import config


def test_seed_is_42() -> None:
    assert config.SEED == 42


def test_cohort_counts_sum_to_thirty() -> None:
    # The cohort must equal 30 chains = 15 Fast Food + 15 Non-Fast Food.
    assert config.N_FAST_FOOD == 15
    assert config.N_NON_FAST_FOOD == 15
    assert config.N_FAST_FOOD + config.N_NON_FAST_FOOD == config.COHORT_SIZE == 30


def test_cohort_thresholds() -> None:
    assert config.MIN_AVG_STARS == 2.5
    assert config.MIN_LOCATIONS == 3
    assert config.FAST_FOOD_TAG == "Fast Food"


def test_temporal_design() -> None:
    assert config.LANDMARK_PRIMARY == date(2018, 1, 1)
    assert config.LANDMARK_SENSITIVITY == date(2019, 1, 1)
    assert config.HORIZON_MONTHS == 24
    assert config.DATASET_END == date(2022, 1, 19)


def test_landmarks_before_dataset_end() -> None:
    assert config.LANDMARK_PRIMARY < config.LANDMARK_SENSITIVITY
    assert config.LANDMARK_SENSITIVITY < config.DATASET_END


def test_quote_grounding_thresholds() -> None:
    assert config.QUOTE_MIN_PARTIAL_RATIO == 85
    assert config.QUOTE_MIN_WORDS == 2


def test_paths_anchored_to_root() -> None:
    # Every project path must live under ROOT_DIR.
    assert config.SRC_DIR == config.ROOT_DIR / "src"
    assert config.ARTIFACTS_DIR == config.ROOT_DIR / "artifacts"
    assert config.PIPELINE_DIR == config.ROOT_DIR / "pipeline"
    assert config.NOTEBOOKS_DIR == config.ROOT_DIR / "notebooks"
    assert config.APP_DIR == config.ROOT_DIR / "app"


def test_embedding_model() -> None:
    assert config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"


def test_max_artifact_size_is_fifty_mb() -> None:
    assert config.MAX_ARTIFACT_BYTES == 50 * 1024 * 1024
