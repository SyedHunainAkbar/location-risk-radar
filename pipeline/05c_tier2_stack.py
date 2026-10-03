"""Pipeline stage 05c: Tier 2 cohort stacking on the out-of-fold Tier 1 score.

Adds the leakage-safe out-of-fold Tier 1 risk score to the cohort feature matrix,
refits the cluster-specific Tier 2 Cox models, runs the Tier2-vs-Tier1 and topic-lift
ablations on the cohort, updates ``risk_scores.parquet`` with
``tier1_score``/``tier2_score``/``final_score`` under the documented rule, and
refreshes the survival model card.

Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/05c_tier2_stack.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.model_selection import GroupKFold  # noqa: E402

from lrr import config, evaluation, io, survival  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tier 2 cohort stacking.")
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument(
        "--cohort-features",
        type=Path,
        default=config.DATA_DIR / config.features_file(config.LANDMARK_PRIMARY),
    )
    p.add_argument("--tier1-oof", type=Path, default=config.DATA_DIR / config.TIER1_OOF_SCORES_FILE)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _cv_harrell(feats, feature_set, extra_cols=()):
    """Return (mean_C, lo, hi): fold-mean C plus a bootstrap CI on pooled OOF."""
    groups = feats["chain"].astype(str)
    k = min(config.N_FOLDS, groups.nunique())
    if k < 2:
        return float("nan"), float("nan"), float("nan")
    gkf = GroupKFold(n_splits=k)
    cs = []
    oof_risk, oof_dur, oof_evt = [], [], []
    for tr, te in gkf.split(feats, groups=groups):
        train, test = feats.iloc[tr], feats.iloc[te]
        Xtr = survival.design_matrix(train, feature_set, pooled=True)
        Xte = survival.design_matrix(test, feature_set, pooled=True)
        for col in extra_cols:
            if col in feats.columns:
                Xtr[col] = train[col].to_numpy()
                Xte[col] = test[col].to_numpy()
        try:
            m, _ = survival.tune_cox_penalizer(Xtr, train["duration_days"], train["event"])
            risk = m.predict_partial_hazard(Xte).to_numpy().ravel()
            cs.append(
                evaluation.harrell_c(
                    test["duration_days"].to_numpy(), test["event"].to_numpy(), risk
                )
            )
            oof_risk.extend(risk.tolist())
            oof_dur.extend(test["duration_days"].tolist())
            oof_evt.extend(test["event"].tolist())
        except Exception as exc:
            print(f"  [warn] fold skipped ({feature_set}): {exc}")
    if not cs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = evaluation.harrell_c_ci(oof_dur, oof_evt, oof_risk)
    return float(np.nanmean(cs)), lo, hi


def run(args):
    feats = io.read_parquet(args.cohort_features)
    oof = io.read_parquet(args.tier1_oof)

    # Join the out-of-fold Tier 1 score as the stacking feature.
    feats = feats.merge(oof[["business_id", "tier1_score"]], on="business_id", how="left")
    feats["tier1_score"] = feats["tier1_score"].fillna(feats["tier1_score"].median())
    print(f"Cohort features with Tier 1 stack: {len(feats)} rows.")

    # Ablation: Tier 1 alone (its OOF score as a one-feature ranker) vs Tier 2
    # (cohort full) vs Tier 2 + Tier 1 stack vs + topics.
    topic_cols = [c for c in feats.columns if c.startswith("topic_")]
    rows = []
    # Tier 1 score only: bootstrap the C index directly on the OOF score.
    c_tier1 = evaluation.harrell_c(
        feats["duration_days"].to_numpy(),
        feats["event"].to_numpy(),
        feats["tier1_score"].to_numpy(),
    )
    t1_lo, t1_hi = evaluation.harrell_c_ci(
        feats["duration_days"].to_numpy(),
        feats["event"].to_numpy(),
        feats["tier1_score"].to_numpy(),
    )
    rows.append(
        {
            "model": "Tier1 score only",
            "harrell_c": c_tier1,
            "harrell_c_lo": t1_lo,
            "harrell_c_hi": t1_hi,
        }
    )
    for label, fset, extra in [
        ("Tier2 full", "full", ()),
        ("Tier2 + Tier1 stack", "full", ["tier1_score"]),
    ]:
        c, lo, hi = _cv_harrell(feats, fset, extra_cols=extra)
        rows.append({"model": label, "harrell_c": c, "harrell_c_lo": lo, "harrell_c_hi": hi})
    if topic_cols:
        c, lo, hi = _cv_harrell(feats, "engagement_text", extra_cols=["tier1_score"])
        rows.append(
            {
                "model": "Tier2 + Tier1 + topics (no topics ablated)",
                "harrell_c": c,
                "harrell_c_lo": lo,
                "harrell_c_hi": hi,
            }
        )
    ablation = pd.DataFrame(rows)
    print("\nTIER 2 vs TIER 1 ABLATION (cohort, Cox, GroupKFold by chain)")
    print("=" * 64)
    print(ablation.round(4).to_string(index=False))

    # Fit the production Tier 2 (full + stack) on all cohort rows for scoring.
    X = survival.design_matrix(feats, "full", pooled=True)
    X["tier1_score"] = feats["tier1_score"].to_numpy()
    model, pen = survival.tune_cox_penalizer(X, feats["duration_days"], feats["event"])
    tier2 = model.predict_partial_hazard(X).to_numpy().ravel()

    risk = pd.DataFrame(
        {
            "business_id": feats["business_id"].to_numpy(),
            "chain": feats.get("chain"),
            "cluster": feats.get("cluster"),
            "is_cohort": True,
            "tier1_score": feats["tier1_score"].to_numpy(),
            "tier2_score": tier2,
        }
    )
    risk["final_score"] = survival.final_score(risk)
    pct = survival.to_percentile(risk["final_score"].to_numpy())
    risk["risk_percentile"] = pct
    risk["risk_tier"] = [survival.risk_tier(p) for p in pct]

    if args.dry_run:
        print("[dry-run] No files written.")
        return {"ablation": ablation, "risk": risk}

    art = Path(args.artifacts_dir)
    # Merge tier fields into any existing risk_scores artifact.
    rs_path = art / config.RISK_SCORES_FILE
    if rs_path.exists():
        prior = io.read_parquet(rs_path)
        drop = [c for c in config.TIER_SCORE_FIELDS if c in prior.columns]
        prior = prior.drop(columns=drop, errors="ignore")
        merged = prior.merge(
            risk[["business_id", *config.TIER_SCORE_FIELDS]], on="business_id", how="outer"
        )
    else:
        merged = risk
    io.write_parquet(merged, rs_path)
    _update_model_card(art, feats, ablation, pen)
    print(f"Updated {rs_path.name} with tier1/tier2/final scores and refreshed the card.")
    return {"ablation": ablation, "risk": risk}


def _update_model_card(art, feats, ablation, penalizer):
    art.mkdir(parents=True, exist_ok=True)
    path = art / config.SURVIVAL_MODEL_CARD_FILE
    section = f"""

## Two-tier addendum (full-corpus upgrade)

**Tier 1 (corpus).** A Cox model over all eligible restaurants using engagement,
check-in, tip, stars, and sentiment aggregates (no topics), under the same landmark
design. It reproduces the Phase A headline that engagement and sentiment beat a
stars-only baseline.

**Tier 2 (cohort).** The cluster-specific cohort models, now stacked with the
out-of-fold Tier 1 risk score. The Tier 1 score for each cohort location comes only
from Tier 1 folds that excluded its chain, so stacking adds no leakage.

**Stacking ablation (Cox, GroupKFold by chain):**
{ablation.round(4).to_string(index=False)}

**Final-score rule.** {config.FINAL_SCORE_RULE}

Production Tier 2 penalizer: {penalizer}.
"""
    if path.exists():
        prior = path.read_text(encoding="utf-8")
        # Replace an older addendum if present, else append.
        marker = "## Two-tier addendum (full-corpus upgrade)"
        if marker in prior:
            prior = prior.split(marker)[0].rstrip()
        path.write_text(prior + "\n" + section, encoding="utf-8")
    else:
        path.write_text("# Survival Model Card\n" + section, encoding="utf-8")


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
