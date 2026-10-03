# Tasks: Landmark Survival

- [ ] 1. Add survival constants to `config.py`
  - Population windows (first-review lead, activity window), risk-tier cut points,
    feature-set names, metric times, artifact filenames.
  - _Requirements: 1.1, 1.2, 2.3, 4.1, 5.1, 7.1_

- [ ] 2. Implement `src/lrr/features.py`
  - `landmark_population`, `last_activity_date`, `label_event_duration`.
  - `monthly_counts`, `ols_slope`, velocity ratio, `build_features`.
  - `out_of_fold_chain_priors`, `assert_no_leakage`.
  - _Requirements: 1.2, 1.3, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

- [ ] 3. Implement `src/lrr/evaluation.py`
  - `harrell_c` (library-free), `uno_c`, `time_dependent_auc`, `integrated_brier`.
  - `calibration_points`, `bootstrap_metric`, `schoenfeld_ph_test`, `ablation_table`.
  - _Requirements: 5.2, 5.3, 5.4, 5.5_

- [ ] 4. Implement `src/lrr/survival.py`
  - `FEATURE_SETS`, `design_matrix` (pooled cluster indicator).
  - `fit_cox`, `fit_rsf`, `fit_xgb` (lazy imports).
  - `risk_tier`, `top_drivers`, `build_risk_scores`, `survival_curves`.
  - _Requirements: 4.1, 4.2, 4.3, 6.1, 6.2, 6.3, 7.1, 7.3_

- [ ] 5. Implement `pipeline/04_features.py`
  - Build and write the feature matrix for a landmark; assert no leakage.
  - _Requirements: 3.5, 8.1, 8.2_

- [ ] 6. Implement `pipeline/05_survival.py`
  - Train, validate (GroupKFold + temporal), ablation, PH test, SHAP.
  - Write risk_scores, model_metrics, survival_curves, model card.
  - _Requirements: 5.1, 5.4, 5.5, 6.1, 6.2, 7.1, 7.2, 7.3, 7.4, 8.1, 8.2_

- [ ] 7. Tests
  - Leakage test; event/duration synthetic timeline; tiering; OOF priors.
  - No heavy survival libs required.
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [ ] 8. Run pytest and make it pass
  - _Requirements: all_
