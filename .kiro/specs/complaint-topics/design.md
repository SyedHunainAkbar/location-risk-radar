# Design: Complaint Topics

## Overview

Stage 03 discovers complaint themes in negative reviews per cluster with BERTopic,
compares unsupervised vs zero-shot models by coherence and diversity, labels topics
with a human-checkable LLM pass, and emits per-business pre-landmark topic shares
for the survival model. Heavy modeling runs offline (Colab); only small artifacts
reach the app.

Split of responsibility:

- `src/lrr/topics.py`: lightweight, testable logic that does not require BERTopic to
  import. Embedding cache, config builders, seed topics, diversity metric, LLM
  labeling via `lrr.llm`, label-override merge, and pre-landmark topic-share
  computation from an existing topic assignment.
- `pipeline/03_topics_colab.py`: the offline driver that imports BERTopic, UMAP,
  HDBSCAN, and gensim, runs the per-cluster models, computes coherence, writes
  artifacts, and saves HTML plots.

Heavy libraries (bertopic, umap-learn, hdbscan, gensim) are imported lazily inside
the functions and script that need them so `topics.py` imports in a plain
environment and the unit tests run without them.

## topics.py

```python
SEED_TOPICS: list[list[str]]                 # the 9 operational themes as keywords

def embedding_cache_path(cluster, corpus) -> Path
def embed_corpus(texts, cluster, cache_dir) -> np.ndarray      # MiniLM, cached npy
def build_umap(seed=SEED)                                      # lazy umap-learn
def build_hdbscan(min_cluster_size)                            # lazy hdbscan
def build_vectorizer()                                         # CountVectorizer EN + bigrams
def build_bertopic(min_cluster_size, zero_shot=False, ...)     # lazy bertopic
def topic_diversity(topic_words, top_k=10) -> float            # unique words / total
def topic_coherence(topic_words, texts, top_k=10) -> float     # lazy gensim c_v
def label_topics_llm(topics_df, provider=None) -> DataFrame    # lrr.llm, temp 0
def apply_label_overrides(labels_df, overrides_csv) -> DataFrame
def compute_topic_shares(assignments, landmark, window_days) -> DataFrame
```

- `embed_corpus` hashes the corpus text, writes `topic_emb_{cluster}_{hash}.npy`,
  and reuses it on the next run. The encoder is `all-MiniLM-L6-v2` via
  sentence-transformers, imported lazily.
- `build_umap` fixes `n_components=5, n_neighbors=15, metric="cosine",
  random_state=42`. `build_vectorizer` uses English stop words and
  `ngram_range=(1, 2)`. `build_bertopic` wires the encoder, UMAP, HDBSCAN, and the
  vectorizer, and when `zero_shot=True` passes `zeroshot_topic_list` built from
  `SEED_TOPICS`.
- `topic_diversity` is the fraction of unique words across the top-k words of all
  topics (Dieng et al.): higher means less redundant topics. `topic_coherence`
  wraps gensim's `CoherenceModel(coherence="c_v")`.
- `label_topics_llm` builds one prompt per topic from its top words and 5
  representative docs, calls `lrr.llm.chat` at temperature 0, and keeps the label
  string only (the LLM never returns numbers; Python assembles the table).
- `apply_label_overrides` left-joins `artifacts/topic_labels.csv`; a non-empty
  override wins over the LLM label.
- `compute_topic_shares` takes per-review topic assignments with `business_id`,
  `date`, and `topic`, keeps reviews strictly before `landmark` (optionally within
  `window_days` before it), and returns a per-business row of topic shares summing
  to 1 (0 when the window is empty), wide by topic id, for the survival model.

## pipeline/03_topics_colab.py

Per cluster: load negative cohort reviews, embed (cached), fit three HDBSCAN
settings, reduce outliers, score coherence and diversity, pick the best, fit the
zero-shot model and compare, label topics via the LLM, apply overrides, and write:

- `artifacts/topics_{cluster}.parquet` (`topic, label, top_words, size`)
- topics over time by quarter and topics closed vs open (small parquet + HTML)
- `data/topic_shares.parquet` (per business, per pre-landmark window)
- HTML plots via BERTopic's `visualize_*().write_html(...)`.

CLI: `--reviews`, `--locations`, `--artifacts-dir`, `--data-dir`, `--cache-dir`,
`--landmark`, `--provider`, `--dry-run`.

## Testing strategy

Unit tests on synthetic data: the seed-topic list and its shape; `topic_diversity`
on known inputs; `compute_topic_shares` windowing (strictly-before-landmark, shares
sum to 1 or 0, correct wide shape); and `label_topics_llm` through the mock LLM
provider so no network is needed. BERTopic/UMAP/HDBSCAN/gensim are never imported in
tests.
