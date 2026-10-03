"""Central configuration for Location Risk Radar.

Every path and constant lives here. No module hard-codes paths or seeds
elsewhere. Importing this module has no side effects beyond resolving paths.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

#: Global random seed. Seed all randomness with this value.
SEED: int = 42

# --------------------------------------------------------------------------- #
# Temporal design (survival / landmark analysis)
# --------------------------------------------------------------------------- #

#: Primary landmark date. Features use only data strictly before this date.
LANDMARK_PRIMARY: date = date(2018, 1, 1)

#: Sensitivity landmark date for robustness checks.
LANDMARK_SENSITIVITY: date = date(2019, 1, 1)

#: Prediction horizon in months after the landmark.
HORIZON_MONTHS: int = 24

#: Last date of observed activity in the Yelp dataset snapshot.
DATASET_END: date = date(2022, 1, 19)

# --------------------------------------------------------------------------- #
# Cohort design
# --------------------------------------------------------------------------- #

#: Total chains in the instructor-approved cohort.
COHORT_SIZE: int = 30

#: Fast Food chains in the cohort.
N_FAST_FOOD: int = 15

#: Non-Fast Food chains in the cohort.
N_NON_FAST_FOOD: int = 15

#: Minimum average stars for cohort inclusion (strictly greater than).
MIN_AVG_STARS: float = 2.5

#: Minimum location count for cohort inclusion (strictly greater than).
MIN_LOCATIONS: int = 3

#: Yelp category string that marks a Fast Food location.
FAST_FOOD_TAG: str = "Fast Food"

#: Regex of cafe-style categories excluded before chain grouping (EDA-faithful).
CAFE_EXCLUDE_REGEX: str = r"Cafes|Coffee & Tea|Tea Rooms|Bubble Tea"

#: Known raw-name variants mapped to a single canonical normalized chain key.
#: Keys are already-normalized tokens (see cohort.normalize_name); values are the
#: canonical normalized key the variant should collapse into.
#:
#: This map is EMPTY by design: the EDA grouped chains with ``normalize_name`` alone
#: and did not collapse variants, so the cohort reconciles to the EDA exactly only
#: when no extra collapsing happens. An earlier map merged QDOBA Mexican Eats with
#: QDOBA Mexican Grill (and similar), inflating that chain's counts (87/3,268 vs the
#: EDA's 81/3,190). We keep the hook for future, deliberately-reconciled variants but
#: ship it empty to preserve exact EDA reproduction.
CHAIN_VARIANT_MAP: dict[str, str] = {}

#: Approved Fast Food cohort (15 chains), by display name.
COHORT_FAST_FOOD: tuple[str, ...] = (
    "Chick-fil-A",
    "Chili's",
    "Applebee's Grill + Bar",
    "Red Robin Gourmet Burgers and Brews",
    "Jimmy John's",
    "Five Guys",
    "Subway",
    "Panda Express",
    "Denny's",
    "Shake Shack",
    "Waffle House",
    "QDOBA Mexican Eats",
    "Blaze Pizza",
    "Hooters",
    "Cheddar's Scratch Kitchen",
)

#: Approved Non-Fast Food cohort (15 chains), by display name.
COHORT_NON_FAST_FOOD: tuple[str, ...] = (
    "Panera Bread",
    "Outback Steakhouse",
    "Los Agaves",
    "Olive Garden Italian Restaurant",
    "Texas Roadhouse",
    "P.F. Chang's",
    "Bonefish Grill",
    "Cracker Barrel Old Country Store",
    "Martin's Bar-B-Que Joint",
    "Metro Diner",
    "Iron Hill Brewery & Restaurant",
    "LongHorn Steakhouse",
    "The Cheesecake Factory",
    "Han Dynasty",
    "Red Lobster",
)

#: EDA reconciliation values per chain: (location_count, total_review_count).
#: total_review_count is the SUM of business.json ``review_count`` per chain,
#: matching the EDA roster (for example Chick-fil-A: 164 locations, 8,028 reviews).
EDA_RECONCILIATION: dict[str, tuple[int, int]] = {
    "Chick-fil-A": (164, 8028),
    "Chili's": (81, 6260),
    "Applebee's Grill + Bar": (112, 5940),
    "Red Robin Gourmet Burgers and Brews": (39, 4586),
    "Jimmy John's": (174, 4419),
    "Five Guys": (92, 4390),
    "Subway": (459, 4123),
    "Panda Express": (115, 3807),
    "Denny's": (76, 3307),
    "Shake Shack": (24, 3251),
    "Waffle House": (107, 3225),
    "QDOBA Mexican Eats": (81, 3190),
    "Blaze Pizza": (24, 3121),
    "Hooters": (39, 3052),
    "Cheddar's Scratch Kitchen": (13, 2848),
    "Panera Bread": (98, 5973),
    "Outback Steakhouse": (56, 5847),
    "Los Agaves": (4, 5160),
    "Olive Garden Italian Restaurant": (47, 5080),
    "Texas Roadhouse": (28, 5070),
    "P.F. Chang's": (19, 4550),
    "Bonefish Grill": (25, 4530),
    "Cracker Barrel Old Country Store": (44, 4255),
    "Martin's Bar-B-Que Joint": (6, 3969),
    "Metro Diner": (14, 3882),
    "Iron Hill Brewery & Restaurant": (12, 3752),
    "LongHorn Steakhouse": (43, 3678),
    "The Cheesecake Factory": (8, 3622),
    "Han Dynasty": (6, 3586),
    "Red Lobster": (44, 3494),
}

#: Expected cohort totals from the EDA (sanity checks).
EDA_TOTAL_LOCATIONS: int = 2054
EDA_TOTAL_OPEN: int = 1805
EDA_TOTAL_CLOSED: int = 249

#: Artifact filenames.
COHORT_LOCATIONS_FILE: str = "cohort_locations.parquet"
COHORT_REVIEWS_FILE: str = "cohort_reviews.parquet"
COHORT_TIPS_FILE: str = "cohort_tips.parquet"
COHORT_CHECKINS_FILE: str = "cohort_checkins.parquet"

# --------------------------------------------------------------------------- #
# Quote grounding (LLM discipline)
# --------------------------------------------------------------------------- #

#: Minimum rapidfuzz partial ratio for a quote to count as grounded.
QUOTE_MIN_PARTIAL_RATIO: int = 85

#: Minimum word count for a grounded quote.
QUOTE_MIN_WORDS: int = 2

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root (two levels up from this file: src/lrr/config.py -> repo).
ROOT_DIR: Path = Path(__file__).resolve().parents[2]

#: Importable package source.
SRC_DIR: Path = ROOT_DIR / "src"

#: Raw data (gitignored). Overridable via the YELP_DIR environment variable.
DATA_DIR: Path = ROOT_DIR / "data"
YELP_DIR: Path = Path(os.environ.get("YELP_DIR", DATA_DIR / "yelp_dataset"))

#: Precomputed artifacts the app reads (committed, each file < 50 MB).
ARTIFACTS_DIR: Path = ROOT_DIR / "artifacts"

#: Offline pipeline scripts.
PIPELINE_DIR: Path = ROOT_DIR / "pipeline"

#: Notebooks.
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"

#: Streamlit app.
APP_DIR: Path = ROOT_DIR / "app"

#: Raw Yelp JSON files, resolved against YELP_DIR.
YELP_BUSINESS_JSON: Path = YELP_DIR / "yelp_academic_dataset_business.json"
YELP_REVIEW_JSON: Path = YELP_DIR / "yelp_academic_dataset_review.json"
YELP_CHECKIN_JSON: Path = YELP_DIR / "yelp_academic_dataset_checkin.json"
YELP_TIP_JSON: Path = YELP_DIR / "yelp_academic_dataset_tip.json"

# --------------------------------------------------------------------------- #
# Embeddings / topics
# --------------------------------------------------------------------------- #

#: Sentence-transformers model used for embeddings.
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# --------------------------------------------------------------------------- #
# LLM gateway (src/lrr/gateway)
# --------------------------------------------------------------------------- #

#: Role-routing config.
GATEWAY_YAML: Path = ROOT_DIR / "config" / "gateway.yaml"

#: Disk cache for LLM responses (gitignored).
LLM_CACHE_DIR: Path = ROOT_DIR / ".cache" / "llm"

#: Provider health-check cache TTL in seconds (10 minutes).
HEALTH_TTL_SECONDS: int = 600

#: Retry/backoff for 429 and 5xx.
GATEWAY_MAX_RETRIES: int = 4
GATEWAY_BASE_DELAY: float = 0.5
GATEWAY_BACKOFF: float = 2.0
GATEWAY_JITTER: float = 0.25

#: Per-provider rate limit (requests per second) for the token-bucket limiter.
GATEWAY_RATE_LIMIT_RPS: float = 4.0

#: Gateway artifact filenames.
GATEWAY_MODELS_FILE: str = "gateway_models.json"
LLM_LEDGER_FILE: str = "llm_ledger.parquet"

#: Rough per-1K-token prices (USD) for cost estimates; 0.0 default for unknown/mock.
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "meta/llama-3.3-70b-instruct": (0.0, 0.0),
    "qwen/qwen2.5-72b-instruct": (0.0, 0.0),
    "qwen/qwen2.5-7b-instruct": (0.0, 0.0),
}

# --------------------------------------------------------------------------- #
# Text and sentiment (stage 02)
# --------------------------------------------------------------------------- #

#: spaCy models: small in the app/pipeline, large offline only.
SPACY_MODEL_APP: str = "en_core_web_sm"
SPACY_MODEL_OFFLINE: str = "en_core_web_lg"

#: Batch size for nlp.pipe streaming.
SPACY_BATCH_SIZE: int = 256

#: Top-N terms per POS in the descriptive lexicons.
LEXICON_TOP_N: int = 20

#: Polarity label mapping. 1-2 stars -> 0, 4-5 -> 1, 3 stars dropped from training.
NEGATIVE_STARS: tuple[int, ...] = (1, 2)
POSITIVE_STARS: tuple[int, ...] = (4, 5)
NEUTRAL_STAR: int = 3

#: Grouped-split test fraction.
TEST_SIZE: float = 0.2

#: Vectorizer settings for CountVectorizer and TF-IDF.
NGRAM_RANGE: tuple[int, int] = (1, 2)
MAX_FEATURES: int = 20_000
MIN_DF: int = 3

#: VADER positive threshold on the compound score.
VADER_POS_THRESHOLD: float = 0.05

#: Bootstrap settings for 95% confidence intervals.
BOOTSTRAP_N: int = 1_000
BOOTSTRAP_CI: float = 0.95

#: GloVe embedding dimension for the offline deep-learning benchmark.
GLOVE_DIM: int = 100

#: Max sequence length for the Keras tokenizer in the offline script.
MAX_SEQUENCE_LEN: int = 200

# --------------------------------------------------------------------------- #
# Complaint topics (stage 03)
# --------------------------------------------------------------------------- #

#: UMAP settings for BERTopic dimensionality reduction.
UMAP_N_COMPONENTS: int = 5
UMAP_N_NEIGHBORS: int = 15
UMAP_METRIC: str = "cosine"

#: Candidate HDBSCAN min_cluster_size values to compare (3 settings).
HDBSCAN_MIN_CLUSTER_SIZES: tuple[int, ...] = (15, 30, 50)

#: CountVectorizer settings for topic c-TF-IDF (English stop words + bigrams).
TOPIC_NGRAM_RANGE: tuple[int, int] = (1, 2)

#: Number of top words used for coherence and diversity.
TOPIC_TOP_K: int = 10

#: Representative docs per topic shown to the LLM labeler.
TOPIC_LABEL_N_DOCS: int = 5

#: Directory for cached MiniLM embeddings (gitignored, under data/).
EMBEDDING_CACHE_DIR: Path = DATA_DIR / "embedding_cache"

#: Operational complaint seed themes for zero-shot topic modeling.
#: Each entry is a short phrase; keyword lists are derived in topics.SEED_TOPICS.
SEED_TOPIC_PHRASES: tuple[str, ...] = (
    "slow service and long wait",
    "rude or inattentive staff",
    "wrong or missing order",
    "cold or poor quality food",
    "cleanliness and hygiene",
    "price and value",
    "management and complaint handling",
    "drive-thru and takeout",
    "reservation and seating",
)

#: Pre-landmark window in days for per-business topic shares (survival features).
#: None means use all reviews strictly before the landmark.
TOPIC_SHARE_WINDOW_DAYS: int | None = 365

#: Stage-03 artifact filenames.
TOPIC_LABELS_OVERRIDE_FILE: str = "topic_labels.csv"
TOPIC_SHARES_FILE: str = "topic_shares.parquet"


def topics_file(cluster: str) -> str:
    """Return the per-cluster topics artifact filename."""
    slug = cluster.lower().replace(" ", "_").replace("-", "_")
    return f"topics_{slug}.parquet"


# --------------------------------------------------------------------------- #
# Landmark survival (stages 04-05)
# --------------------------------------------------------------------------- #

#: A location must have its first review at least this many months before T.
POP_FIRST_REVIEW_LEAD_MONTHS: int = 12

#: A location must have activity within this many months before T to be eligible.
POP_ACTIVITY_WINDOW_MONTHS: int = 6

#: Window (months) over which review velocity slope is fit.
VELOCITY_WINDOW_MONTHS: int = 24

#: Review-volume lookback windows (months) ending at T.
VOLUME_WINDOWS_MONTHS: tuple[int, ...] = (12, 6, 3)

#: Time points (months after T) for time-dependent AUC and Brier.
METRIC_TIMES_MONTHS: tuple[int, ...] = (12, 24)

#: GroupKFold folds (grouped by chain).
N_FOLDS: int = 5

#: Risk-tier cut points on the risk percentile (0-100), inclusive lower bounds.
RISK_TIER_CUTS: dict[str, float] = {
    "High": 90.0,
    "Elevated": 75.0,
    "Watch": 50.0,
    "Low": 0.0,
}

#: Nested feature-set names for the ablation (defined concretely in survival.py).
FEATURE_SET_NAMES: tuple[str, ...] = (
    "stars_only",
    "engagement",
    "engagement_text",
    "full",
)

#: Candidate Cox penalizers to tune over.
COX_PENALIZERS: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)

#: Stage 04/05 artifact and feature filenames.
RISK_SCORES_FILE: str = "risk_scores.parquet"
MODEL_METRICS_FILE: str = "model_metrics.parquet"
SURVIVAL_CURVES_FILE: str = "survival_curves.parquet"
SURVIVAL_MODEL_CARD_FILE: str = "survival_model_card.md"


def features_file(landmark) -> str:
    """Return the feature-matrix filename for a landmark date."""
    year = landmark.year if hasattr(landmark, "year") else str(landmark)[:4]
    return f"features_T{year}.parquet"


# --------------------------------------------------------------------------- #
# Evidence assistant: aspects + RAG (stage 06)
# --------------------------------------------------------------------------- #

#: Aspect agents.
ASPECTS: tuple[str, ...] = ("food", "service", "ambience")

#: Allowed polarity labels from the agents.
POLARITIES: tuple[str, ...] = ("positive", "negative", "neutral")

#: Intensity bounds (inclusive).
INTENSITY_MIN: int = 1
INTENSITY_MAX: int = 3

#: Retry budget for schema-valid aspect JSON.
ASPECT_JSON_RETRIES: int = 2

#: Aspect sampling: top-k and bottom-k locations by risk per chain, reviews each.
ASPECT_TOP_K_LOCATIONS: int = 10
ASPECT_BOTTOM_K_LOCATIONS: int = 10
ASPECT_REVIEWS_PER_LOCATION: int = 10

#: RAG index: recent reviews per location, chunk size, retrieval settings.
RAG_RECENT_REVIEWS: int = 150
RAG_CHUNK_MAX_WORDS: int = 120
RAG_TOP_K: int = 8
RAG_MMR_LAMBDA: float = 0.5

#: RAG answerer and judge models (judge is a different family).
RAG_ANSWER_MODEL: str = "meta/llama-3.3-70b-instruct"
RAG_JUDGE_MODEL: str = "qwen/qwen2.5-7b-instruct"

#: Six standard operator questions precomputed per high-risk location.
RAG_STANDARD_QUESTIONS: tuple[str, ...] = (
    "Why is this location at risk?",
    "What are the top service complaints in the last 6 months?",
    "What do customers praise about this location?",
    "What are the most common food quality issues?",
    "Have customers mentioned cleanliness or wait-time problems?",
    "What would most improve this location's reviews?",
)

#: Stage-06 artifact and index paths/filenames.
RAG_INDEX_DIR: Path = DATA_DIR / "rag_index"
RAG_EMBEDDINGS_FILE: str = "rag_embeddings.npy"
RAG_METADATA_FILE: str = "rag_metadata.parquet"
RAG_CACHE_FILE: str = "rag_cache.parquet"
RAG_EVAL_FILE: str = "rag_eval.parquet"
ASPECT_RESULTS_FILE: str = "aspect_results.parquet"
ASPECT_EVAL_FILE: str = "aspect_eval.parquet"
ASPECT_CHECKPOINT_FILE: str = "aspect_checkpoint.jsonl"


# --------------------------------------------------------------------------- #
# Full-corpus two-tier modeling (stages 01b-05c)
# --------------------------------------------------------------------------- #

#: Streaming chunk size for the one-pass corpus panel and batch scoring.
CORPUS_CHUNKSIZE: int = 200_000

#: Max non-cohort reviews sampled to train the corpus sentiment scorer.
NONCOHORT_SAMPLE_SIZE: int = 1_000_000

#: Monthly activity panel columns (restaurant universe).
PANEL_COLUMNS: tuple[str, ...] = (
    "business_id",
    "month",
    "n_reviews",
    "mean_stars",
    "n_tips",
    "n_checkins",
)

#: Corpus artifact filenames (large, gitignored under data/).
ALL_RESTAURANTS_FILE: str = "all_restaurants.parquet"
ACTIVITY_MONTHLY_FILE: str = "activity_monthly.parquet"
SENTIMENT_MONTHLY_FILE: str = "sentiment_monthly.parquet"
TIER1_FEATURES_FILE: str = "tier1_features.parquet"
TIER1_OOF_SCORES_FILE: str = "tier1_oof_scores.parquet"
TRANSFER_EVAL_FILE: str = "sentiment_transfer_eval.parquet"

#: Risk-score tier fields added by Tier 2 stacking.
TIER_SCORE_FIELDS: tuple[str, ...] = ("tier1_score", "tier2_score", "final_score")

#: Documented final-score choice rule (also written to the model card).
FINAL_SCORE_RULE: str = (
    "final_score = tier2_score for cohort locations (the cluster-specific model, "
    "which already stacks the out-of-fold Tier 1 corpus signal), and tier1_score "
    "for all non-cohort restaurants (which have no cohort-specific model). This "
    "keeps the cohort's calibrated cluster models authoritative while extending "
    "coverage to the full restaurant universe."
)


# --------------------------------------------------------------------------- #
# Risk committee agents (stage 08)
# --------------------------------------------------------------------------- #

#: Allowed Voice-of-Customer issue labels.
VOC_ISSUES: tuple[str, ...] = (
    "service_speed",
    "staff_attitude",
    "order_accuracy",
    "food_quality",
    "cleanliness",
    "value",
    "management_response",
    "drive_thru_takeout",
    "other",
)

#: Confidence formula inputs.
CONFIDENCE_GRADE_WEIGHT: dict[str, float] = {"A": 1.0, "B": 0.7, "C": 0.4}
CONFIDENCE_EVIDENCE_TARGET: int = 5
CONFIDENCE_LABEL_CUTS: dict[str, float] = {"High": 0.66, "Medium": 0.33, "Low": 0.0}

#: Bounded retries for the Quant Analyst placeholder/digit guard.
QUANT_PLACEHOLDER_RETRIES: int = 2

#: Risk tiers for which briefs are precomputed.
BRIEF_TIERS: tuple[str, ...] = ("High", "Elevated")

#: Playbook config and stage-08 artifact filenames.
ACTIONS_YAML: Path = ROOT_DIR / "config" / "actions.yaml"
BRIEFS_FILE: str = "briefs.parquet"
AGENT_EVAL_FILE: str = "agent_eval.parquet"


#: Stage-02 artifact filenames.
LEXICON_CLOSED_OPEN_NOUN_FILE: str = "lexicon_closed_open_noun.parquet"
LEXICON_CLOSED_OPEN_ADJ_FILE: str = "lexicon_closed_open_adj.parquet"
LEXICON_CLUSTER_NOUN_FILE: str = "lexicon_cluster_noun.parquet"
LEXICON_CLUSTER_ADJ_FILE: str = "lexicon_cluster_adj.parquet"
SENTIMENT_BENCHMARK_FILE: str = "sentiment_benchmark.parquet"
REVIEW_SENTIMENT_FILE: str = "review_sentiment.parquet"

#: Max artifact file size the app is allowed to read, in bytes.
MAX_ARTIFACT_BYTES: int = 50 * 1024 * 1024


def ensure_dirs() -> None:
    """Create non-raw project directories if they do not exist.

    Does not create ``DATA_DIR`` or ``YELP_DIR``; raw data is provided
    externally and is gitignored.
    """
    for path in (ARTIFACTS_DIR, PIPELINE_DIR, NOTEBOOKS_DIR, APP_DIR):
        path.mkdir(parents=True, exist_ok=True)
