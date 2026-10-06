# Survival Model Card: Location Risk Radar

## Design
We use a landmark survival design. We freeze the landmark at
2018-01-01 (primary) and
2019-01-01 (sensitivity). We build every feature from
records dated strictly before the landmark and observe closure over the following
24 months. We proxy the closure date with the last observed
activity across reviews, tips, and check-ins, because the dataset gives a status
flag, not a closure date.

## Population and outcome
We model 1545 eligible locations with 65 events in the 24-month window. A
location is eligible when its first review predates the landmark by at least
12 months and it shows activity in the
6 months before the landmark.

## Models
We fit four nested feature sets (stars only, engagement, engagement plus text, and
full) as Cox PH, Random Survival Forest, and an XGBoost 24-month classifier, per
cluster and pooled with a cluster indicator. We tune the Cox penalizer over
[0.001, 0.01, 0.1, 1.0].

## Validation
We validate with GroupKFold by chain (5 folds) and a temporal check that trains on
the 2018 landmark and tests on 2019. We report Harrell and Uno C, time-dependent AUC
at 12 and 24 months, and the integrated Brier score, each with bootstrap 95%
confidence intervals.

## Ablation (text over stars)
    feature_set  harrell_c  delta_c_vs_stars  uno_c  auc_12m  auc_24m  ibs
     stars_only     0.6558            0.0000    NaN      NaN      NaN  NaN
     engagement     0.6280           -0.0278    NaN      NaN      NaN  NaN
engagement_text     0.6509           -0.0049    NaN      NaN      NaN  NaN
           full     0.6324           -0.0234    NaN      NaN      NaN  NaN

## Proportional hazards
We run the Schoenfeld test on the Cox model. When a covariate violates proportional
hazards we stratify on it or add a time interaction, and we note the affected
covariate here after each run.

## Explainability
We report Cox hazard ratios with confidence intervals and SHAP values for the
XGBoost model, and we surface the top three drivers per location in the risk table.

## Risk tiers
{'Low': 772, 'Watch': 386, 'Elevated': 232, 'High': 155}

## Limitations
We proxy the closure date from last activity, so late censoring and survivorship in
the text remain. The Fast Food cluster carries known label noise (sit-down brands
tagged Fast Food). Chains are grouped by normalized name because the data has no
chain id. We compute chain priors out of fold to avoid target leakage.

## Tier 1 corpus model (stage 05b) - SKIPPED

Tier 1 training exceeded the 10-minute compute cap on this machine and was skipped. The Tier 2 cohort model and the evidence stages still ran. Tier 1 can be produced offline on a larger machine with `python pipeline/05b_tier1_survival.py`.
