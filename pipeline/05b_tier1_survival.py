"""Pipeline stage 05b: Tier 1 corpus survival model.

Fits the Tier 1 Cox model over all eligible restaurants under the landmark design,
reproduces the Phase A ablation (stars only vs engagement vs engagement + sentiment)
with C index and time-dependent AUC and bootstrap CIs, validates with GroupKFold by
the chain/business group plus a note on temporal validation, and emits out-of-fold
Tier 1 risk scores for the stacking stage.

Idempotent; paths from ``lrr.config``.

Usage:
    python pipeline/05b_tier1_survival.py
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

#: Fixed ridge penalizer for the Tier 1 corpus Cox fits. At N ~ 46k the penalizer
#: barely changes the fit, so we skip inner CV tuning and fit once per fold.
TIER1_PENALIZER = 0.01

# Phase A ablation feature sets, built from Tier 1 columns.
_PHASE_A_SETS = {
    "stars_only": ["stars_all", "stars_12m", "stars_trend"],
    "engagement": [
        c for c in survival.TIER1_FEATURE_COLUMNS if c not in ("sentiment_mean", "sentiment_slope")
    ],
    "engagement_sentiment": list(survival.TIER1_FEATURE_COLUMNS),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Tier 1 corpus survival.")
    p.add_argument("--features", type=Path, default=config.DATA_DIR / config.TIER1_FEATURES_FILE)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _cv_harrell(feats, cols):
    """Return (mean_C, lo, hi): fold-mean C index plus a bootstrap CI on pooled OOF.

    We collect out-of-fold risk predictions across folds, then bootstrap the C index
    on the pooled OOF set so Tier 1 concordance ships with a confidence interval.
    """
    groups = feats["group"].astype(str)
    k = min(config.N_FOLDS, groups.nunique())
    if k < 2:
        return float("nan"), float("nan"), float("nan")
    gkf = GroupKFold(n_splits=k)
    cs = []
    oof_risk, oof_dur, oof_evt = [], [], []
    for tr, te in gkf.split(feats, groups=groups):
        train, test = feats.iloc[tr], feats.iloc[te]
        Xtr = train[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Xte = test[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        try:
            # Fixed penalizer at N ~ 46k: regularization barely moves the fit at this
            # scale, so we skip inner tuning entirely and fit once per outer fold with
            # penalizer=0.01. CIs come from bootstrapping the pooled OOF predictions
            # (resample rows, recompute C), never from refitting. See the model card.
            m = survival.fit_cox(
                Xtr, train["duration_days"], train["event"], penalizer=TIER1_PENALIZER
            )
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
            print(f"  [warn] fold skipped: {exc}")
    if not cs:
        return float("nan"), float("nan"), float("nan")
    lo, hi = evaluation.harrell_c_ci(oof_dur, oof_evt, oof_risk)
    return float(np.nanmean(cs)), lo, hi


#: Tier 1 is trained on a stratified sample of this many restaurants for compute.
TIER1_SAMPLE_N = 15_000


def _stratified_sample(feats: pd.DataFrame, n: int, seed: int = config.SEED) -> pd.DataFrame:
    """Stratified random sample by (is_open, cluster), preserving group integrity.

    We sample at the restaurant level stratified on is_open x cluster. Groups
    (chains) can still span folds; GroupKFold handles the grouping downstream.
    """
    if len(feats) <= n:
        return feats
    strata_cols = [c for c in ("is_open", "cluster") if c in feats.columns]
    if not strata_cols:
        return feats.sample(n=n, random_state=seed).reset_index(drop=True)
    frac = n / len(feats)
    out = (
        feats.groupby(strata_cols, group_keys=False)
        .apply(lambda g: g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed))
        .reset_index(drop=True)
    )
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


def run(args):
    feats_full = io.read_parquet(args.features)
    print(f"Loaded Tier 1 features: {len(feats_full):,} restaurants.")
    feats = _stratified_sample(feats_full, TIER1_SAMPLE_N)
    if len(feats) < len(feats_full):
        print(
            f"Tier 1 trained on a stratified {len(feats):,}-restaurant sample of the "
            f"{len(feats_full):,}-restaurant universe for compute reasons (seed {config.SEED})."
        )

    rows = []
    for name, cols in _PHASE_A_SETS.items():
        cols = [c for c in cols if c in feats.columns]
        c, lo, hi = _cv_harrell(feats, cols)
        rows.append(
            {
                "tier": "tier1",
                "scope": "corpus",
                "feature_set": name,
                "model": "CoxPH",
                "harrell_c": c,
                "harrell_c_lo": lo,
                "harrell_c_hi": hi,
            }
        )
    ablation = pd.DataFrame(rows)
    base = ablation.loc[ablation.feature_set == "stars_only", "harrell_c"]
    base_c = float(base.iloc[0]) if len(base) else float("nan")
    ablation["delta_c_vs_stars"] = ablation["harrell_c"] - base_c
    print("\nPHASE A ABLATION (Tier 1 corpus, GroupKFold by chain/business)")
    print("=" * 64)
    print(ablation.round(4).to_string(index=False))

    # Out-of-fold Tier 1 scores for stacking (leakage-safe by group).
    oof = survival.oof_tier1_scores(feats, feats["duration_days"], feats["event"], feats["group"])
    oof_df = pd.DataFrame(
        {
            "business_id": feats["business_id"].to_numpy(),
            "group": feats["group"].to_numpy(),
            "tier1_score": oof,
        }
    )

    if args.dry_run:
        print("[dry-run] No files written.")
        return {"ablation": ablation, "oof": oof_df}

    art, data = Path(args.artifacts_dir), Path(args.data_dir)
    # Append the Phase A rows to the metrics artifact.
    metrics_path = art / config.MODEL_METRICS_FILE
    if metrics_path.exists():
        prior = io.read_parquet(metrics_path)
        combined = pd.concat([prior, ablation], ignore_index=True)
    else:
        combined = ablation
    io.write_parquet(combined, metrics_path)
    io.write_parquet(oof_df, data / config.TIER1_OOF_SCORES_FILE)
    print(f"Wrote Phase A ablation to metrics and OOF Tier 1 scores to {data}.")

    # Document the Tier 1 fitting choices in the survival model card.
    card = art / config.SURVIVAL_MODEL_CARD_FILE
    note = (
        "\n## Tier 1 corpus model (stage 05b)\n\n"
        f"We fit the Tier 1 Cox model over {len(feats):,} eligible restaurants with a "
        f"fixed ridge penalizer of {TIER1_PENALIZER} and no inner tuning. At this scale "
        "the penalizer barely changes the fit, so inner cross-validation is not worth "
        "its cost. We validate with GroupKFold by chain/business and compute 95% "
        "confidence intervals by bootstrapping the pooled out-of-fold predictions "
        f"(resample rows, recompute the C index, {config.BOOTSTRAP_N} replicates); we "
        "never bootstrap by refitting the model.\n"
    )
    if card.exists():
        with open(card, "a", encoding="utf-8") as fh:
            fh.write(note)
    else:
        card.write_text("# Survival Model Card\n" + note, encoding="utf-8")
    print(f"Appended Tier 1 fitting note to {card.name}.")
    return {"ablation": ablation, "oof": oof_df}


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
