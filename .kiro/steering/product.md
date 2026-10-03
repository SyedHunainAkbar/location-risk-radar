# Product: Location Risk Radar

Built by team LRR Analytics for CIS 509 (Analytics for Unstructured Data), Arizona State University.

## Goal

An early warning web app that flags restaurant chain locations at elevated risk of closure and explains why using customer text. We surface the signal before the closure, and we back every flag with evidence a human can read.

## Framework

A three-stage pipeline:

1. Survival/Hazard model estimates each location's risk over time.
2. BERTopic extracts complaint themes from customer reviews and tips.
3. RAG evidence assistant retrieves grounded quotes that explain the risk.

## Users

- Regional operations managers
- Franchise owners
- General managers
- Portfolio managers

## Locked Scope

- Data source: Yelp Open Dataset only (business, review, checkin, tip).
- Instructor-approved cohort of 30 chains: 15 Fast Food and 15 Non-Fast Food.
- Cohort criteria: average stars greater than 2.5 and more than 3 locations.
- Fast Food is defined as any chain with a location tagged "Fast Food".
- We train cluster-specific models (Fast Food and Non-Fast Food modeled separately).

## Headline Evidence

The average star gap between open and closed restaurants is about 0.03. Stars alone do not separate survivors from closures. The signal lives in engagement dynamics (checkin and review volume trends) and in review language, which is why the text layer is central rather than decorative.

## Known Limitations (always state these)

- No `chain_id` in the data; chains are grouped by name normalization, which is imperfect.
- Fast Food label noise: sit-down brands such as Chili's, Applebee's, and Denny's appear in the Fast Food cluster.
- `is_open` is a current status flag, not a closure date.
- Closure date is proxied by last observed activity, not a confirmed close date.
- Right censoring: locations still open at the end of the observation window have unknown eventual outcomes.
- Survivorship in text: reviews from closed locations stop when the location stops, biasing late-period language.
