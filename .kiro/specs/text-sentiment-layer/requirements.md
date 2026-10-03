# Requirements: Text and Sentiment Layer

## Introduction

We turn cohort review text into two things: a descriptive lexicon layer (LA1) and a
calibrated sentiment scorer (LA2), with a deep-learning benchmark run offline on a
Colab GPU (LA3). Everything respects grouped validation by business so no location
leaks across train and test, and every reported number carries a bootstrap 95%
confidence interval. This is pipeline stage 02.

## Requirements

### Requirement 1 (LA1): spaCy normalization and descriptive lexicons

**User story:** As an analyst, I want normalized tokens and the words that
characterize closed vs open and Fast Food vs Non-Fast Food reviews, so the text
signal is legible before any modeling.

#### Acceptance Criteria
1. WHEN normalizing text THEN the system SHALL use spaCy with `nlp.pipe` in batches,
   lowercasing, lemmatizing, and removing stopwords and punctuation.
2. The system SHALL retain part-of-speech tags and named entities for each token
   stream so nouns and adjectives can be isolated.
3. The system SHALL produce the top 20 nouns and the top 20 adjectives for
   (a) reviews of locations that later closed vs those that stayed open, and
   (b) Fast Food vs Non-Fast Food reviews.
4. The system SHALL save these tables to `artifacts/lexicon_*.parquet`.
5. The system SHALL be memory-safe: batching via `nlp.pipe`, disabling unused
   pipeline components where possible.

### Requirement 2 (LA2): labeling and leakage-free split

**User story:** As a modeler, I want polarity labels and a grouped split so my
metrics are not inflated by the same business appearing on both sides.

#### Acceptance Criteria
1. WHEN labeling THEN reviews with 1-2 stars SHALL be class 0, 4-5 stars SHALL be
   class 1, and 3-star reviews SHALL be dropped from training.
2. WHEN splitting THEN all reviews of a given `business_id` SHALL stay in the same
   fold, using `GroupShuffleSplit` keyed on `business_id`.
3. The split SHALL be seeded with `config.SEED` and SHALL be reproducible.
4. The train and test sets SHALL share no `business_id`.

### Requirement 3 (LA2): model comparison with uncertainty

**User story:** As a reviewer, I want a fair benchmark with confidence intervals so
the production choice is defensible.

#### Acceptance Criteria
1. The system SHALL compare VADER (compound > 0.05 is positive), and
   CountVectorizer and TF-IDF features (unigram+bigram, `max_features=20000`,
   `min_df=3`) each with MultinomialNB, LinearSVC, and LogisticRegression.
2. The system SHALL report accuracy, macro F1, per-class recall, and ROC AUC.
3. Each metric SHALL carry a bootstrap 95% confidence interval.
4. The production scorer SHALL be chosen by macro F1.

### Requirement 4 (LA2): calibration and scoring

**User story:** As a downstream stage, I want a probability of positive sentiment
for every review including 3-star ones.

#### Acceptance Criteria
1. The chosen scorer SHALL be calibrated with `CalibratedClassifierCV` to output
   `P(positive)`.
2. The system SHALL score ALL cohort reviews, including 3-star reviews.
3. The system SHALL save `data/review_sentiment.parquet` with the review id,
   business id, and `p_positive`.

### Requirement 5 (LA3): offline deep-learning benchmark

**User story:** As a modeler, I want GRU and LSTM baselines on the same split so the
final benchmark table is complete.

#### Acceptance Criteria
1. The offline Colab script SHALL train GRU and LSTM models with GloVe 100d
   embeddings, both frozen and trainable, on the same grouped split.
2. Training SHALL use early stopping.
3. The script SHALL report accuracy, macro F1, per-class recall, and ROC AUC, with
   bootstrap 95% CIs, matching the LA2 metric set.
4. Results SHALL be written to `artifacts/sentiment_benchmark.parquet`, joined with
   the LA2 model rows so one table shows all models.
5. The production scorer SHALL be chosen on evidence, and the latency/accuracy
   trade-off SHALL be documented.
6. TensorFlow/Keras SHALL appear only in the offline script, never in the app or
   the online pipeline modules.

### Requirement 6: determinism, idempotency, and CLI

#### Acceptance Criteria
1. `pipeline/02_text_sentiment.py` SHALL accept CLI args and be idempotent.
2. All randomness SHALL use `config.SEED`; all paths SHALL come from `config.py`.

### Requirement 7: tests

#### Acceptance Criteria
1. Tests SHALL verify the star-to-label mapping including that 3-star rows drop.
2. Tests SHALL verify the grouped split shares no `business_id` across folds.
3. Tests SHALL run without a GPU, without TensorFlow, and without the real dataset.
