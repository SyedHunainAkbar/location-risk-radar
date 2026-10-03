# Requirements: Cohort Ingestion

## Introduction

We ingest the four Yelp Open Dataset files, filter to restaurants, normalize chain
names exactly as our EDA does, reproduce the locked 30-chain cohort, and write the
precomputed artifacts the rest of the pipeline consumes. This is pipeline stage 01.
It must be deterministic, idempotent, and reconcile exactly to the EDA numbers.

## Requirements

### Requirement 1: Stream raw Yelp JSON from tar or plain files

**User story:** As an analyst, I want to read the raw Yelp files from `YELP_DIR`
whether they are inside the distributed `.tar` or extracted as plain `.json`, so I
do not have to extract 10+ GB by hand.

#### Acceptance Criteria
1. WHEN a file `yelp_academic_dataset_<name>.json` exists in `YELP_DIR` THEN the
   system SHALL stream it line by line without loading the whole file into memory.
2. WHEN only a `.tar` archive is present in `YELP_DIR` THEN the system SHALL read
   the matching member directly from the archive without extraction.
3. IF both a plain `.json` and a `.tar` member exist THEN the system SHALL prefer
   the plain `.json`.
4. WHEN a requested dataset is missing from both sources THEN the system SHALL raise
   a clear `FileNotFoundError` naming the dataset and `YELP_DIR`.
5. The reader SHALL yield parsed dict records and SHALL skip blank lines.

### Requirement 2: Restaurant filter and EDA-faithful name normalization

**User story:** As an analyst, I want chain grouping that matches the EDA exactly,
so the cohort is reproducible.

#### Acceptance Criteria
1. WHEN selecting restaurants THEN the system SHALL keep businesses whose
   `categories` contain "Restaurants" (case-insensitive).
2. WHEN preparing chains THEN the system SHALL exclude cafe-style businesses whose
   `categories` match `Cafes|Coffee & Tea|Tea Rooms|Bubble Tea` (case-insensitive).
3. WHEN normalizing a name THEN the system SHALL lowercase, strip leading/trailing
   whitespace, remove every character that is not a word character or whitespace,
   and collapse internal whitespace to single spaces. This is the EDA
   `normalize_name` behavior.
4. WHEN a known raw-name variant is encountered THEN the system SHALL map it to a
   single canonical chain via a variant map so spelling variants group together.
5. WHEN labeling Fast Food THEN a location SHALL be Fast Food if its `categories`
   contain "Fast Food" (case-insensitive), and a chain SHALL be Fast Food if any of
   its locations are Fast Food.

### Requirement 3: Reproduce and assert the locked 30-chain cohort

**User story:** As a reviewer, I want a hard assertion that the cohort is exactly
the instructor-approved 30 chains, so drift is caught immediately.

#### Acceptance Criteria
1. WHEN applying selection THEN the system SHALL keep chains with average stars
   strictly greater than 2.5 AND location count strictly greater than 3.
2. WHEN ranking THEN the system SHALL sort eligible chains by total review count
   descending and take the top 15 Fast Food and top 15 Non-Fast Food.
3. The system SHALL assert the result equals 30 chains split 15 Fast Food and 15
   Non-Fast Food.
4. The system SHALL assert the selected chain display names equal the approved
   constant list in `config.py` (set equality per cluster).
5. The approved chain list SHALL live as a constant in `config.py`.
6. WHEN run against the real dataset THEN per-chain location and review counts
   SHALL match the EDA reconciliation values stored in `config.py`.

### Requirement 4: Write artifacts

**User story:** As a downstream stage, I want clean parquet inputs so I never parse
raw JSON again.

#### Acceptance Criteria
1. The system SHALL write `artifacts/cohort_locations.parquet` with columns
   `business_id, chain, cluster, city, state, lat, lon, stars, review_count,
   is_open`.
2. The system SHALL write `data/cohort_reviews.parquet` with columns
   `review_id, business_id, stars, date, text, useful, funny, cool`.
3. The system SHALL write `data/cohort_tips.parquet`.
4. The system SHALL write `data/cohort_checkins.parquet` with exploded per-timestamp
   rows (`business_id, checkin_at`).
5. Location and review artifacts SHALL contain only cohort businesses.
6. Writes SHALL be idempotent: rerunning with the same inputs produces the same
   artifacts.

### Requirement 5: Data quality report

**User story:** As a reviewer, I want a transparent data quality report so I can
trust the inputs, with problems flagged not silently imputed.

#### Acceptance Criteria
1. The report SHALL count duplicate business ids, review ids, and tip rows.
2. The report SHALL count null or empty review text.
3. The report SHALL report the review date range.
4. The report SHALL flag cohort locations with zero check-ins and zero tips.
5. The system SHALL NOT impute missing engagement; it SHALL only flag and count.

### Requirement 6: Determinism, idempotency, and CLI

**User story:** As an operator, I want a single idempotent command with arguments.

#### Acceptance Criteria
1. `pipeline/01_ingest_cohort.py` SHALL accept CLI args for the Yelp directory,
   output directories, and a dry-run flag.
2. All randomness, if any, SHALL use `config.SEED`.
3. All paths SHALL come from `config.py`.
4. Rerunning SHALL overwrite artifacts deterministically.

### Requirement 7: Tests on a synthetic fixture

**User story:** As a developer, I want fast tests that need no real Yelp data.

#### Acceptance Criteria
1. Tests SHALL cover name normalization including punctuation, case, trademark
   symbols, whitespace, and variant mapping.
2. Tests SHALL build a tiny synthetic business table and assert the cohort selection
   and the 15/15 assertion behave correctly, including a negative case where the
   split is wrong.
3. Tests SHALL run without network or the real dataset.
