# Tasks: Text and Sentiment Layer

- [ ] 1. Add text/sentiment constants to `config.py`
  - Label thresholds, vectorizer params, bootstrap settings, lexicon top-N,
    GloVe dim, artifact filenames.
  - _Requirements: 2.1, 3.1, 4.1, 5.1, 6.2_

- [ ] 2. Implement `src/lrr/text.py`
  - `load_nlp`, `normalize_tokens` via `nlp.pipe`, POS/NER retention.
  - `pos_lexicon` and `contrast_lexicons` (top 20 nouns/adjectives per group).
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Implement `src/lrr/sentiment.py`
  - `map_polarity_labels`, `grouped_split`.
  - VADER baseline, vectorizer and classifier factories.
  - `evaluate`, `bootstrap_ci`, `benchmark`, `pick_production`.
  - `calibrate_scorer`, `score_all`.
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3_

- [ ] 4. Implement `pipeline/02_text_sentiment.py`
  - Idempotent CLI; write lexicons, LA2 benchmark, review_sentiment.
  - _Requirements: 1.4, 3.4, 4.3, 6.1, 6.2_

- [ ] 5. Implement `pipeline/02b_deep_sentiment_colab.py`
  - GRU/LSTM with GloVe 100d (frozen + trainable), early stopping, same split.
  - Append to the benchmark parquet; document latency/accuracy trade-off.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [ ] 6. Tests
  - Label mapping (3-star dropped, boundaries).
  - Grouped split shares no `business_id`.
  - No GPU, no TensorFlow, no real dataset.
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 7. Run pytest and make it pass
  - _Requirements: all_
