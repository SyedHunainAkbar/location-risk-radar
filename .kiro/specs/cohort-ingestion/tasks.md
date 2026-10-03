# Tasks: Cohort Ingestion

- [ ] 1. Add cohort constants to `config.py`
  - Approved Fast Food and Non-Fast Food chain name lists.
  - EDA reconciliation values (per-chain location and review counts).
  - Cafe-exclusion regex, chain variant map, artifact filenames.
  - _Requirements: 2.2, 2.4, 3.4, 3.5, 3.6_

- [ ] 2. Implement `src/lrr/io.py`
  - `iter_yelp_records` streaming from plain `.json` and from a `.tar` member.
  - Resolution order and clear `FileNotFoundError`.
  - `read_yelp_frame` (chunked), `write_parquet`, `read_parquet`.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [ ] 3. Implement `src/lrr/cohort.py`
  - `normalize_name` (EDA-faithful) and `canonical_chain` (variant map).
  - Restaurant, cafe, and Fast Food predicates.
  - `clean_business`, `build_chain_summary`, `select_cohort`.
  - `assert_cohort` (15/15 and name set-equality).
  - `reconciliation_table` and `data_quality_report`.
  - _Requirements: 2.1, 2.3, 2.5, 3.1, 3.2, 3.3, 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 4. Implement `pipeline/01_ingest_cohort.py`
  - Idempotent CLI with `--yelp-dir`, `--artifacts-dir`, `--data-dir`, `--dry-run`.
  - Write all four artifacts, scoped to cohort business ids.
  - Print reconciliation table and data quality report.
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 6.1, 6.2, 6.3, 6.4_

- [ ] 5. Tests on a synthetic fixture
  - Normalization: case, punctuation, trademark symbols, whitespace, variants.
  - Cohort selection and 15/15 assertion, plus a negative wrong-split case.
  - No network, no real dataset.
  - _Requirements: 7.1, 7.2, 7.3_

- [ ] 6. Run pytest and make it pass
  - _Requirements: all_
