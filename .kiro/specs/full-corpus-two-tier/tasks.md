# Tasks: Full-Corpus Two-Tier Modeling

- [ ] 1. Add constants to `config.py`
  - Corpus paths, panel schema, non-cohort sample size, streaming chunk size,
    tier score fields, final-score rule text.
  - _Requirements: 1.1, 1.2, 2.1, 5.1_

- [ ] 2. Implement `src/lrr/corpus.py`
  - `build_all_restaurants`, streaming `stream_activity_panel`, resource logging,
    `panel_window_features`.
  - _Requirements: 1.1-1.5, 3.2_

- [ ] 3. Extend `src/lrr/sentiment.py`
  - `noncohort_training_sample`, `transfer_evaluate`, `score_reviews_streaming`.
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ] 4. Extend `src/lrr/features.py` and `src/lrr/survival.py`
  - `tier1_features`, `tier1_labels`, `group_key`.
  - `fit_tier1`, `oof_tier1_scores`, `final_score`, `FINAL_SCORE_RULE`.
  - _Requirements: 3.1-3.5, 4.1-4.3, 5.1_

- [ ] 5. Implement pipelines 01b, 02c, 04b, 05b, 05c
  - Streaming ingest, corpus sentiment + transfer, Tier 1 fit + OOF scores,
    Tier 2 stacking, updated risk_scores + model card.
  - _Requirements: 1.4, 2.2, 3.5, 4.2, 4.3, 5.1, 5.2, 7.2_

- [ ] 6. Tests
  - Streaming panel equals brute force; stacking uses only out-of-fold predictions;
    is_cohort flag; final-score rule; streaming sentiment aggregates.
  - _Requirements: 5.3_

- [ ] 7. Update notebook sections 2 and 6 (Phase A vs Phase B)
  - _Requirements: 6.1_

- [ ] 8. Run pytest and make it pass
  - _Requirements: all_
