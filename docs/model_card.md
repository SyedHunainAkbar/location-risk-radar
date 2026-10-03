# Model Card: Location Risk Radar

## Overview

A landmark survival model that estimates restaurant-location closure risk over a
24-month horizon, modeled per cluster (Fast Food and Non-Fast Food) and pooled, in
two tiers (corpus-wide Tier 1 and cohort-specific Tier 2).

## Intended use

Decision support for regional operations managers, franchise owners, general
managers, and portfolio managers. The output is a ranked, explained watchlist for a
human to review. It is **not** an automated closure decision and must not be used to
justify adverse employment action.

## Data

Yelp Open Dataset (business, review, tip, checkin), academic terms. A locked cohort
of 30 chains (15 Fast Food, 15 Non-Fast Food) selected by average stars above 2.5 and
more than three locations, ranked by review volume. The dataset is not redistributed.

## Models

- Cox proportional hazards (lifelines, penalizer tuned).
- Random Survival Forest (scikit-survival).
- XGBoost 24-month event classifier (comparison).

## Evaluation

Harrell and Uno C, time-dependent AUC at 12 and 24 months, integrated Brier score,
all with bootstrap 95% CIs; calibration; a temporal 2018 to 2019 check; and a COVID
sensitivity reading. Headline numbers are generated from
`artifacts/model_metrics.parquet`.

## Limitations

- No `chain_id`; chains grouped by normalized name.
- Fast Food label noise (sit-down brands tagged Fast Food).
- `is_open` is a status, not a closure date; closure proxied by last activity.
- Right censoring at the window end; survivorship bias in late-period text.
- The COVID shock after early 2020 distorts activity; the 2018 landmark keeps the
  24-month window mostly clear, and 2019 is a sensitivity check.

## Ethical considerations

The model explains every flag with cited customer text and signed drivers. It is a
triage aid, reviewed by a human, not a decision engine.
