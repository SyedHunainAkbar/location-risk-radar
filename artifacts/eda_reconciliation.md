# EDA reconciliation

We recompute the EDA quantities from the raw Yelp data using the same restaurant filter, chain normalization, Fast Food rule, and cohort criteria, and compare them against the pipeline's committed artifacts. Source notebook: `EDA/Final Submission/FINAL_Location_Risk_Radar_EDA.ipynb`. We mark each row honestly against the EDA's stated value and the population it was computed on.

## Headline checks

| Metric | EDA | Pipeline | Match |
|---|---|---|---|
| Total businesses (raw) | 150,346 | 150,346 | yes |
| Restaurants (categories contain "Restaurants") | 52,268 | 52,268 | yes |
| Restaurants after cafe exclusions | 46,269 | 46,269 | yes |
| Closure rate (all restaurants, pre-exclusion) | 0.331 | 0.331 | yes |
| Star gap, all restaurants pre-exclusion (EDA Cell 5 / 25) | 0.026 | 0.026 | yes |
| Star gap, after cafe exclusions (not reported in EDA) | n/a | 0.060 | n/a |
| Star gap, 30-chain cohort (not reported in EDA) | n/a | 0.085 | n/a |
| Cohort chains | 30 (15 + 15) | 30 (15 + 15) | yes |
| Cohort locations | 2,054 | 2,054 | yes |
| Cohort reviews in scope (Cell 17A / 25) | 135,631 | 135,631 | yes |
| Fast Food closure rate (EDA Cell 25, chain-level mean) | 0.126 | 0.126 | yes |
| Fast Food avg stars (EDA Cell 25, chain-level mean) | 2.95 | 2.95 | yes |
| Non-Fast Food closure rate (EDA Cell 25, chain-level mean) | 0.072 | 0.072 | yes |
| Non-Fast Food avg stars (EDA Cell 25, chain-level mean) | 3.36 | 3.36 | yes |
| Review-volume ratio open/closed (EDA Cell 5, all restaurants) | 1.96x | 1.96x | yes |
| Check-in recency gap, closed - open (EDA Cell 8, all restaurants) | 4.7 yr | 4.7 yr | yes |
| Fast Food closure rate (cohort, location-weighted; not reported in EDA) | n/a | 0.133 | n/a |
| Non-Fast Food closure rate (cohort, location-weighted; not reported in EDA) | n/a | 0.079 | n/a |
| Review-volume ratio open/closed (cohort; not reported in EDA) | n/a | 2.37x | n/a |
| Check-in recency gap (cohort, median; not reported in EDA) | n/a | 3.5 yr | n/a |

### Notes on populations and definitions

Every headline row reconciles once computed on the exact population and aggregation the EDA used. The earlier mismatches were population errors on our side; the corrected definitions are:

- **Star gap (headline)**: EDA Cell 5 on `rest_df`, the 52,268 businesses whose categories contain "Restaurants" (before cafe exclusions, closure 33.1%). On that population the gap is **0.026** (open 3.524, closed 3.498), matching the Cell 25 headline to three decimals. The post-exclusion gap (0.060 over 46,269) and the cohort gap (0.085 over 2,054) are separate labeled rows, not the EDA figure.
- **Cluster closure and stars**: EDA Cell 25 takes the unweighted mean of the 15 chains' per-chain `closure_rate` and `average_stars` within each cluster (chain-level averages, not location-weighted). On that definition we reproduce FF 0.126 / 2.95 and Non-Fast Food 0.072 / 3.36 exactly. The cohort location-weighted closure rates (13.3% and 7.9%) are kept as separate labeled rows for reference.
- **Review-volume ratio**: EDA Cell 5 on all 52,268 restaurants, `mean(review_count | open) / mean(review_count | closed)` = 104.14 / 53.10 = **1.96x**, reproduced exactly. The cohort-level ratio (2.37x) is a different population and is labeled separately.
- **Check-in recency gap**: EDA Cell 8 on all 52,268 restaurants, median last check-in open (2021-10-07) minus closed (2017-01-15) = **4.7 years**, reproduced exactly with the last check-in parsed as the final timestamp in each raw check-in date string (34,516 open and 16,786 closed restaurants had check-ins). The cohort-level median gap (3.5 years) is a different population and is labeled separately.

The headline conclusion is unchanged: the star gap is a fraction of one star, so rating alone does not separate survivors from closures. The separation lives in engagement dynamics (open restaurants carry about twice the review volume and far more recent check-ins) and in review language.

## Per-chain location and review counts (30 chains)

EDA reviews are the sum of the business `review_count` snapshot field (as the EDA chain table used). Pipeline reviews are the actual number of rows in `review.json` for the chain's cohort locations. Difference % is `(pipeline - EDA) / EDA * 100`.

| Chain | EDA locations | Pipeline locations | EDA reviews | Pipeline reviews | Difference % |
|---|---|---|---|---|---|
| Chick-fil-A | 164 | 164 | 8,028 | 8,402 | +4.7% |
| Chili's | 81 | 81 | 6,260 | 6,534 | +4.4% |
| Panera Bread | 98 | 98 | 5,973 | 6,221 | +4.2% |
| Outback Steakhouse | 56 | 56 | 5,847 | 6,207 | +6.2% |
| Applebee's Grill + Bar | 112 | 112 | 5,940 | 6,146 | +3.5% |
| Texas Roadhouse | 28 | 28 | 5,070 | 5,362 | +5.8% |
| Olive Garden Italian Restaurant | 47 | 47 | 5,080 | 5,324 | +4.8% |
| Los Agaves | 4 | 4 | 5,160 | 5,253 | +1.8% |
| Red Robin Gourmet Burgers and Brews | 39 | 39 | 4,586 | 4,826 | +5.2% |
| P.F. Chang's | 19 | 19 | 4,550 | 4,821 | +6.0% |
| Bonefish Grill | 25 | 25 | 4,530 | 4,805 | +6.1% |
| Jimmy John's | 174 | 174 | 4,419 | 4,569 | +3.4% |
| Five Guys | 92 | 92 | 4,390 | 4,537 | +3.3% |
| Cracker Barrel Old Country Store | 44 | 44 | 4,255 | 4,431 | +4.1% |
| Subway | 459 | 459 | 4,123 | 4,278 | +3.8% |
| Metro Diner | 14 | 14 | 3,882 | 4,076 | +5.0% |
| Martin's Bar-B-Que Joint | 6 | 6 | 3,969 | 4,035 | +1.7% |
| Panda Express | 115 | 115 | 3,807 | 3,985 | +4.7% |
| Iron Hill Brewery & Restaurant | 12 | 12 | 3,752 | 3,908 | +4.2% |
| LongHorn Steakhouse | 43 | 43 | 3,678 | 3,852 | +4.7% |
| The Cheesecake Factory | 8 | 8 | 3,622 | 3,788 | +4.6% |
| Han Dynasty | 6 | 6 | 3,586 | 3,773 | +5.2% |
| Red Lobster | 44 | 44 | 3,494 | 3,682 | +5.4% |
| Denny's | 76 | 76 | 3,307 | 3,405 | +3.0% |
| Shake Shack | 24 | 24 | 3,251 | 3,372 | +3.7% |
| Waffle House | 107 | 107 | 3,225 | 3,308 | +2.6% |
| QDOBA Mexican Eats | 81 | 81 | 3,190 | 3,302 | +3.5% |
| Blaze Pizza | 24 | 24 | 3,121 | 3,262 | +4.5% |
| Hooters | 39 | 39 | 3,052 | 3,197 | +4.8% |
| Cheddar's Scratch Kitchen | 13 | 13 | 2,848 | 2,970 | +4.3% |

All 30 location counts match exactly between the EDA and the pipeline.

_Footnote: per-chain review differences arise only because the EDA chain table summed `business.review_count` (a point-in-time snapshot field Yelp stored on each business), while the pipeline counts actual rows in `review.json`. At the review level the corpus matches exactly: the pipeline's 30-chain total is 135,631 rows, identical to the EDA's cohort reviews in scope (Cell 17A / Cell 25). The per-chain snapshot lags the row count by a small, uniform 2 to 6 percent, consistent with snapshot staleness rather than any filtering difference._
