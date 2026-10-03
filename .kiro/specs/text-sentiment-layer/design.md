# Design: Text and Sentiment Layer

## Overview

Stage 02 produces the descriptive lexicon tables (LA1) and a calibrated sentiment
scorer (LA2), plus an offline Colab script for the deep-learning benchmark (LA3).
The online path uses only CPU-friendly libraries (spaCy small model, scikit-learn,
VADER). TensorFlow/Keras and GloVe live exclusively in the offline script.

Modules:

- `src/lrr/text.py`: spaCy normalization via `nlp.pipe`, token streams with POS and
  NER, and top-N noun/adjective lexicons by contrast group.
- `src/lrr/sentiment.py`: label mapping, grouped split, feature/model zoo, metrics
  with bootstrap CIs, calibration, and scoring.
- `pipeline/02_text_sentiment.py`: wires LA1 and LA2, writes artifacts.
- `pipeline/02b_deep_sentiment_colab.py`: offline GRU/LSTM + GloVe benchmark.

## text.py

```python
def load_nlp(model: str = "en_core_web_sm", disable=("parser",)) -> Language
def normalize_tokens(texts, nlp, batch_size, keep_pos=None) -> list[list[str]]
def pos_lexicon(texts, labels, nlp, pos, top_n=20) -> DataFrame
def contrast_lexicons(reviews_df, nlp) -> dict[str, DataFrame]
```

`normalize_tokens` streams with `nlp.pipe(texts, batch_size=...)`, lowercases the
lemma, and drops stopwords, punctuation, spaces, and non-alphabetic tokens. When
`keep_pos` is given (for example `{"NOUN"}` or `{"ADJ"}`) only those tags survive.
`pos_lexicon` counts the top `top_n` lemmas of a POS within each label value and
returns a tidy table (`group, pos, term, count, rank`). `contrast_lexicons`
produces the four tables: closed vs open nouns/adjectives and Fast Food vs
Non-Fast Food nouns/adjectives, keyed for `artifacts/lexicon_*.parquet`.

The spaCy model is loaded with the parser disabled (we need the tagger, lemmatizer,
attribute ruler, and NER only) to keep `nlp.pipe` fast and memory-light.

## sentiment.py

```python
def map_polarity_labels(df) -> DataFrame           # 1-2 -> 0, 4-5 -> 1, drop 3
def grouped_split(df, group_col="business_id", test_size=0.2, seed=SEED)
def vader_predict(texts) -> (labels, scores)       # compound > 0.05
def build_vectorizer(kind) -> CountVectorizer | TfidfVectorizer
def build_classifier(name) -> estimator
def evaluate(y_true, y_pred, y_score) -> dict       # acc, macroF1, recalls, AUC
def bootstrap_ci(y_true, y_pred, y_score, metric, n=1000, seed=SEED)
def benchmark(train, test) -> DataFrame             # all feature x model rows + VADER
def pick_production(benchmark_df) -> str            # argmax macro F1
def calibrate_scorer(train, vectorizer, classifier) -> fitted pipeline
def score_all(reviews_df, scorer) -> DataFrame      # p_positive for every review
```

Vectorizers: unigram+bigram, `max_features=20000`, `min_df=3`. Classifiers:
`MultinomialNB`, `LinearSVC`, `LogisticRegression`. VADER is a non-trainable
baseline scored directly on raw text. `LinearSVC` has no `predict_proba`; for ROC
AUC we use its `decision_function`, and for calibration we wrap it in
`CalibratedClassifierCV`. The production scorer is the max macro-F1 row; it is
calibrated so `P(positive)` is meaningful, then applied to every cohort review
(including 3-star), writing `data/review_sentiment.parquet`.

Bootstrap CIs resample test rows with replacement (seeded) and recompute each
metric to get the 2.5 and 97.5 percentiles.

## Metrics schema (shared across LA2 and LA3)

`model, features, accuracy, accuracy_lo, accuracy_hi, macro_f1, macro_f1_lo,
macro_f1_hi, recall_neg, recall_pos, roc_auc, roc_auc_lo, roc_auc_hi, n_test`.
Both stages write this schema so the benchmark concatenates cleanly.

## pipeline/02_text_sentiment.py

Reads `data/cohort_reviews.parquet` and `artifacts/cohort_locations.parquet`,
joins cluster and derived closed/open status, builds LA1 lexicons, runs the LA2
benchmark on the grouped split, selects and calibrates the production scorer,
scores all reviews, and writes `artifacts/lexicon_*.parquet`,
`artifacts/sentiment_benchmark.parquet` (LA2 rows), and
`data/review_sentiment.parquet`. CLI: `--reviews`, `--locations`,
`--artifacts-dir`, `--data-dir`, `--sample`, `--dry-run`.

## pipeline/02b_deep_sentiment_colab.py

Offline, GPU-friendly. Loads the same reviews and the same grouped split (seeded
identically), tokenizes with Keras, loads GloVe 100d, builds an embedding matrix,
and trains four nets: GRU and LSTM, each with frozen and trainable embeddings, with
`EarlyStopping`. Computes the shared metric schema with bootstrap CIs and appends to
`artifacts/sentiment_benchmark.parquet` so the final notebook renders one table.
Documents the latency/accuracy trade-off in a printed summary and a markdown note.

## Testing strategy

Unit tests on a tiny synthetic frame: label mapping (3-star dropped, boundaries
correct) and grouped split (no `business_id` intersection, reproducible). Tests use
scikit-learn and pandas only, no spaCy model download, no TensorFlow, no GPU. The
spaCy-dependent lexicon functions are covered by a guarded test that skips if the
model is not installed.
