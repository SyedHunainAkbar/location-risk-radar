# Tasks: Complaint Topics

- [ ] 1. Add topic constants to `config.py`
  - UMAP params, HDBSCAN candidate sizes, vectorizer settings, seed topics,
    embedding cache dir, quarter/window settings, artifact filenames.
  - _Requirements: 2.1, 2.2, 2.3, 3.1, 1.3, 5.1, 5.3_

- [ ] 2. Implement `src/lrr/topics.py`
  - `SEED_TOPICS`, embedding cache + `embed_corpus`.
  - `build_umap`, `build_hdbscan`, `build_vectorizer`, `build_bertopic`.
  - `topic_diversity`, `topic_coherence` (lazy gensim).
  - `label_topics_llm` (temperature 0), `apply_label_overrides`.
  - `compute_topic_shares` (strictly pre-landmark windowing).
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.3, 2.4, 2.5, 3.1, 4.1, 4.2, 4.3, 5.3_

- [ ] 3. Implement `pipeline/03_topics_colab.py`
  - Per-cluster offline run, 3-setting comparison, zero-shot vs unsupervised.
  - Topics over time by quarter and closed vs open.
  - Pre-landmark topic shares; HTML plots.
  - Idempotent CLI.
  - _Requirements: 2.5, 3.2, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3_

- [ ] 4. Tests
  - Seed topics and zero-shot wiring, diversity metric.
  - Pre-landmark windowing and share-sum invariants.
  - LLM labeling via mock provider; no BERTopic/GPU/network.
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 5. Run pytest and make it pass
  - _Requirements: all_
