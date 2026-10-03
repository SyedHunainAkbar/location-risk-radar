# Requirements: Complaint Topics

## Introduction

We discover what customers complain about in negative reviews, separately for each
cluster, with BERTopic. We compare an unsupervised model against a zero-shot model
seeded with operational complaint themes, label every topic with a human-checkable
LLM pass, and emit per-business pre-landmark topic shares as survival features. The
heavy compute runs offline (Colab); the app only reads the small artifacts. This is
pipeline stage 03.

## Requirements

### Requirement 1: Per-cluster negative-review corpus with cached embeddings

**User story:** As a modeler, I want the negative-review corpus per cluster with
embeddings cached, so topic runs are fast and reproducible.

#### Acceptance Criteria
1. The corpus SHALL be the 1-2 star reviews of cohort locations, built separately
   for Fast Food and Non-Fast Food.
2. Embeddings SHALL use `all-MiniLM-L6-v2`.
3. Embeddings SHALL be cached to disk keyed by cluster and corpus hash, and reused
   when the corpus is unchanged.

### Requirement 2: BERTopic configuration and model selection

**User story:** As a modeler, I want a defensible BERTopic configuration and a
metric-driven choice among settings.

#### Acceptance Criteria
1. UMAP SHALL use `n_components=5`, `n_neighbors=15`, `metric="cosine"`,
   `random_state=42`.
2. HDBSCAN `min_cluster_size` SHALL be tuned for interpretable topics.
3. The vectorizer SHALL be a CountVectorizer with English stop words and bigrams,
   feeding c-TF-IDF.
4. Outliers SHALL be reduced with `reduce_outliers`.
5. The system SHALL report topic coherence (c_v via gensim) and topic diversity for
   3 settings and choose by those metrics.

### Requirement 3: Zero-shot comparison

**User story:** As an analyst, I want to compare the unsupervised model against a
zero-shot model seeded with operational complaint themes.

#### Acceptance Criteria
1. The system SHALL run zero-shot topic modeling with these seed topics: slow
   service and long wait; rude or inattentive staff; wrong or missing order; cold or
   poor quality food; cleanliness and hygiene; price and value; management and
   complaint handling; drive-thru and takeout; reservation and seating.
2. The system SHALL compare the zero-shot model against the unsupervised model on
   coherence and diversity.

### Requirement 4: Human-checkable topic labels via the LLM

**User story:** As a reviewer, I want short topic labels I can trust and override.

#### Acceptance Criteria
1. Each topic SHALL get a short label produced via `lrr.llm` from the topic's top
   words and 5 representative documents, at temperature 0.
2. The LLM SHALL return a label string only; Python assembles the final table.
3. A human SHALL be able to override labels via `artifacts/topic_labels.csv`, and
   overrides SHALL win when present.

### Requirement 5: Outputs

**User story:** As downstream stages, I want topic tables and per-business features.

#### Acceptance Criteria
1. The system SHALL write `artifacts/topics_{cluster}.parquet` with columns
   `topic, label, top_words, size`.
2. The system SHALL produce topics over time by quarter and topics for closed vs
   open locations.
3. The system SHALL write per-business, per-pre-landmark-window topic shares to
   `data/topic_shares.parquet` for the survival model, using only reviews strictly
   before the landmark date.
4. The system SHALL save small static plots as HTML for the notebook.

### Requirement 6: Offline boundary, determinism, and CLI

#### Acceptance Criteria
1. BERTopic, UMAP, HDBSCAN, and gensim SHALL run only in the offline Colab script,
   never in the app.
2. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.
3. `pipeline/03_topics_colab.py` SHALL accept CLI args and be idempotent.

### Requirement 7: Tests

#### Acceptance Criteria
1. Tests SHALL verify the seed-topic list and zero-shot config wiring.
2. Tests SHALL verify pre-landmark windowing excludes on-or-after-landmark reviews
   and that topic shares per business sum to 1 (or 0 when a window is empty).
3. Tests SHALL verify the topic-diversity metric and the LLM labeling path using the
   mock provider, with no network, no GPU, and no BERTopic install required.
