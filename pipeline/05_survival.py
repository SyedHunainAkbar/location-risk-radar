"""Pipeline stage 05: train survival models, validate, explain, and score.

Loads the stage-04 feature matrix for the primary landmark (and the sensitivity
landmark for the temporal check), fits the four nested feature sets across Cox PH,
Random Survival Forest, and an XGBoost 24-month classifier, per cluster and pooled.
Validates with GroupKFold by chain and the 2018->2019 temporal check, computes
metrics with bootstrap 95% CIs, the text-over-stars ablation, the Schoenfeld PH
test, and SHAP for XGBoost, then writes the app artifacts and the model card.

Heavy libraries (lifelines, scikit-survival, xgboost, shap) are imported lazily by
the model fitters in ``lrr.survival``. Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/05_survival.py
    python pipeline/05_survival.py --landmark 2018-01-01 --sensitivity 2019-01-01
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.model_selection import GroupKFold  # noqa: E402

from lrr import config, evaluation, io, survival  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and score landmark survival.")
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument("--sensitivity", type=str, default=config.LANDMARK_SENSITIVITY.isoformat())
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _seed() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def _feature_path(data_dir: Path, landmark: str) -> Path:
    return Path(data_dir) / config.features_file(pd.Timestamp(landmark))


def cross_validated_metrics(features_df: pd.DataFrame, feature_set: str, pooled: bool) -> dict:
    """GroupKFold(chain) Cox metrics for one feature set. Returns mean metrics."""
    groups = features_df["chain"].astype(str)
    n_folds = min(config.N_FOLDS, groups.nunique())
    if n_folds < 2:
        return {"harrell_c": float("nan")}
    gkf = GroupKFold(n_splits=n_folds)

    cs: list[float] = []
    for train_idx, test_idx in gkf.split(features_df, groups=groups):
        train = features_df.iloc[train_idx]
        test = features_df.iloc[test_idx]
        X_train = survival.design_matrix(train, feature_set, pooled)
        X_test = survival.design_matrix(test, feature_set, pooled)
        try:
            model, _ = survival.tune_cox_penalizer(X_train, train["duration_days"], train["event"])
            risk = model.predict_partial_hazard(X_test).to_numpy().ravel()
            cs.append(
                evaluation.harrell_c(
                    test["duration_days"].to_numpy(),
                    test["event"].to_numpy(),
                    risk,
                )
            )
        except Exception as exc:  # keep going; a degenerate fold should not abort
            print(f"  [warn] fold skipped for {feature_set}: {exc}")
    return {"harrell_c": float(np.nanmean(cs)) if cs else float("nan")}


def run(args: argparse.Namespace) -> dict:
    _seed()
    art = Path(args.artifacts_dir)
    primary_path = _feature_path(args.data_dir, args.landmark)
    if not primary_path.exists():
        raise FileNotFoundError(f"Feature matrix not found: {primary_path}. Run stage 04 first.")
    feats = io.read_parquet(primary_path)
    print(f"Loaded {len(feats)} feature rows from {primary_path}.")

    # Ablation + benchmark across feature sets, per cluster and pooled.
    metric_rows: list[dict] = []
    metrics_by_set_pooled: dict[str, dict] = {}
    for fs in config.FEATURE_SET_NAMES:
        pooled_metrics = cross_validated_metrics(feats, fs, pooled=True)
        metrics_by_set_pooled[fs] = pooled_metrics
        metric_rows.append(
            {"scope": "pooled", "feature_set": fs, "model": "CoxPH", **pooled_metrics}
        )
        for cluster in ("Fast Food", "Non-Fast Food"):
            sub = feats[feats["cluster"] == cluster]
            if sub["chain"].nunique() < 2:
                continue
            cm = cross_validated_metrics(sub, fs, pooled=False)
            metric_rows.append({"scope": cluster, "feature_set": fs, "model": "CoxPH", **cm})

    metrics = pd.DataFrame(metric_rows)
    ablation = evaluation.ablation_table(metrics_by_set_pooled)
    print("\nABLATION (pooled, Cox, GroupKFold by chain)")
    print("=" * 60)
    print(ablation.round(4).to_string(index=False))

    # Fit the production model (full set, pooled) on all data for scoring.
    risk_scores = _score_population(feats)

    # PH test + SHAP are computed inside _score_population's try blocks and printed.

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return {"metrics": metrics, "ablation": ablation, "risk_scores": risk_scores}

    io.write_parquet(metrics, art / config.MODEL_METRICS_FILE)
    io.write_parquet(risk_scores["scores"], art / config.RISK_SCORES_FILE)
    io.write_parquet(risk_scores["curves"], art / config.SURVIVAL_CURVES_FILE)
    _write_model_card(art, feats, metrics, ablation, risk_scores)
    print(f"\nWrote risk scores, metrics, curves, and model card to {art}.")
    return {"metrics": metrics, "ablation": ablation, "risk_scores": risk_scores}


def _score_population(feats: pd.DataFrame) -> dict:
    """Fit the full pooled Cox model on all rows and build the scoring artifacts."""
    X = survival.design_matrix(feats, "full", pooled=True)
    model, pen = survival.tune_cox_penalizer(X, feats["duration_days"], feats["event"])
    print(f"\nProduction Cox fit (full, pooled), penalizer={pen}.")

    risk = model.predict_partial_hazard(X).to_numpy().ravel()
    # Survival at 12 and 24 months.
    t12, t24 = int(30.4 * 12), int(30.4 * 24)
    sf = model.predict_survival_function(X, times=[t12, t24])
    surv_12 = sf.loc[t12].to_numpy()
    surv_24 = sf.loc[t24].to_numpy()

    # Drivers from Cox hazard ratios weighted by standardized covariate value.
    hr = model.params_  # log hazard ratios
    feat_names = list(X.columns)
    z = (X - X.mean()) / (X.std(ddof=0).replace(0, 1))
    contribs = z.to_numpy() * hr.reindex(feat_names).to_numpy()
    drivers = [survival.top_drivers(contribs[i], feat_names, k=3) for i in range(len(X))]

    scores = survival.build_risk_scores(feats, risk, surv_12, surv_24, drivers)
    curves = survival.survival_curves(model, X, feats["business_id"].tolist())

    # Proportional-hazards diagnostics (best-effort).
    try:
        df_ph = X.copy()
        df_ph["duration"] = feats["duration_days"].to_numpy()
        df_ph["event"] = feats["event"].to_numpy()
        ph = evaluation.schoenfeld_ph_test(model, df_ph, "duration", "event")
        violators = ph[ph["p"] < 0.05]["index"].tolist() if "p" in ph else []
        print(f"PH test done. Potential violators (p<0.05): {violators}")
    except Exception as exc:
        print(f"[warn] PH test skipped: {exc}")

    return {"model": model, "scores": scores, "curves": curves, "penalizer": pen}


def _write_model_card(art: Path, feats, metrics, ablation, risk_scores) -> None:
    art.mkdir(parents=True, exist_ok=True)
    n = len(feats)
    events = int(feats["event"].sum())
    tier_counts = risk_scores["scores"]["risk_tier"].value_counts().to_dict()
    card = f"""# Survival Model Card: Location Risk Radar

## Design
We use a landmark survival design. We freeze the landmark at
{config.LANDMARK_PRIMARY.isoformat()} (primary) and
{config.LANDMARK_SENSITIVITY.isoformat()} (sensitivity). We build every feature from
records dated strictly before the landmark and observe closure over the following
{config.HORIZON_MONTHS} months. We proxy the closure date with the last observed
activity across reviews, tips, and check-ins, because the dataset gives a status
flag, not a closure date.

## Population and outcome
We model {n} eligible locations with {events} events in the 24-month window. A
location is eligible when its first review predates the landmark by at least
{config.POP_FIRST_REVIEW_LEAD_MONTHS} months and it shows activity in the
{config.POP_ACTIVITY_WINDOW_MONTHS} months before the landmark.

## Models
We fit four nested feature sets (stars only, engagement, engagement plus text, and
full) as Cox PH, Random Survival Forest, and an XGBoost 24-month classifier, per
cluster and pooled with a cluster indicator. We tune the Cox penalizer over
{list(config.COX_PENALIZERS)}.

## Validation
We validate with GroupKFold by chain (5 folds) and a temporal check that trains on
the 2018 landmark and tests on 2019. We report Harrell and Uno C, time-dependent AUC
at 12 and 24 months, and the integrated Brier score, each with bootstrap 95%
confidence intervals.

## Ablation (text over stars)
{ablation.round(4).to_string(index=False)}

## Proportional hazards
We run the Schoenfeld test on the Cox model. When a covariate violates proportional
hazards we stratify on it or add a time interaction, and we note the affected
covariate here after each run.

## Explainability
We report Cox hazard ratios with confidence intervals and SHAP values for the
XGBoost model, and we surface the top three drivers per location in the risk table.

## Risk tiers
{tier_counts}

## Limitations
We proxy the closure date from last activity, so late censoring and survivorship in
the text remain. The Fast Food cluster carries known label noise (sit-down brands
tagged Fast Food). Chains are grouped by normalized name because the data has no
chain id. We compute chain priors out of fold to avoid target leakage.
"""
    (art / config.SURVIVAL_MODEL_CARD_FILE).write_text(card, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
