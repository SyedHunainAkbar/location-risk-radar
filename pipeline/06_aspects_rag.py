"""Pipeline stage 06: aspect agents and the RAG evidence assistant.

Two phases:

- aspects: build a stratified sample (per chain, the top-10 and bottom-10 locations
  by risk, up to 10 recent pre-landmark reviews each, capped by --sample-size), run
  the food/service/ambience/lead agents with a JSONL checkpoint for resume, write
  the aspect results, the predicted-stars evaluation, and the high-vs-low risk
  contrast.
- rag: build the compact per-location index, precompute answers for the standard
  questions on high-risk locations, and write the cache and an eval summary.

All LLM calls go through lrr.llm (nvidia/voyager/mock). Idempotent; paths from
lrr.config. Use --provider mock for a no-network dry run.

Usage:
    python pipeline/06_aspects_rag.py --phase all --provider mock --sample-size 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import aspects, config, io, rag  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aspect agents and RAG evidence assistant.")
    p.add_argument("--phase", choices=["aspects", "rag", "all"], default="all")
    p.add_argument("--provider", type=str, default=None)
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument(
        "--sample-size",
        type=int,
        default=600,
        help="Max reviews to run through the aspect agents (cost cap).",
    )
    p.add_argument(
        "--per-chain",
        type=int,
        default=0,
        help="If >0, sample this many reviews per chain stratified by star rating "
        "(seed 42) instead of the risk-stratified sample. Keeps live cost bounded.",
    )
    p.add_argument("--reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    p.add_argument("--tips", type=Path, default=config.DATA_DIR / config.COHORT_TIPS_FILE)
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument(
        "--risk-scores", type=Path, default=config.ARTIFACTS_DIR / config.RISK_SCORES_FILE
    )
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--index-dir", type=Path, default=config.RAG_INDEX_DIR)
    p.add_argument(
        "--checkpoint", type=Path, default=config.ARTIFACTS_DIR / config.ASPECT_CHECKPOINT_FILE
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _seed() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def _read_optional(path: Path) -> pd.DataFrame:
    path = Path(path)
    return io.read_parquet(path) if path.exists() else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Aspect sampling
# --------------------------------------------------------------------------- #


def build_aspect_sample(
    reviews: pd.DataFrame,
    risk_scores: pd.DataFrame,
    T: pd.Timestamp,
    sample_size: int,
) -> pd.DataFrame:
    """Stratified sample: per chain top/bottom-k locations by risk, recent pre-T reviews."""
    picks: list[str] = []
    for _, grp in risk_scores.groupby("chain"):
        ordered = grp.sort_values("risk_score", ascending=False)
        top = ordered.head(config.ASPECT_TOP_K_LOCATIONS)["business_id"].tolist()
        bottom = ordered.tail(config.ASPECT_BOTTOM_K_LOCATIONS)["business_id"].tolist()
        picks.extend(top + bottom)
    picks = list(dict.fromkeys(picks))  # dedupe, keep order

    rev = reviews.copy()
    rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
    rev = rev[(rev["business_id"].isin(picks)) & (rev["date"] < T)]
    # Up to N recent pre-landmark reviews per location.
    rev = (
        rev.sort_values("date", ascending=False)
        .groupby("business_id", group_keys=False)
        .head(config.ASPECT_REVIEWS_PER_LOCATION)
    )
    # Attach risk tier for the contrast, then cap total cost.
    tier = dict(zip(risk_scores["business_id"], risk_scores["risk_tier"]))
    rev["risk_tier"] = rev["business_id"].map(tier)
    if sample_size and len(rev) > sample_size:
        rev = rev.sample(n=sample_size, random_state=config.SEED)
    return rev.reset_index(drop=True)


def build_per_chain_sample(
    reviews: pd.DataFrame,
    locations: pd.DataFrame,
    T: pd.Timestamp,
    per_chain: int,
    seed: int = config.SEED,
) -> pd.DataFrame:
    """Sample ``per_chain`` pre-T reviews per chain, stratified by star rating.

    For each chain we allocate the per-chain budget across the star ratings present
    (1..5) as evenly as possible, filling any shortfall from the remaining reviews
    by recency. Deterministic under ``seed``. Keeps the live aspect-agent run small.
    """
    loc = locations[["business_id", "chain"]].drop_duplicates()
    rev = reviews.copy()
    rev["date"] = pd.to_datetime(rev["date"], errors="coerce")
    rev = rev.merge(loc, on="business_id", how="inner")
    rev = rev[rev["date"] < T]
    rev["stars"] = pd.to_numeric(rev["stars"], errors="coerce")

    out = []
    for chain, grp in rev.groupby("chain"):
        stars_present = sorted(s for s in grp["stars"].dropna().unique())
        if not stars_present:
            continue
        base = per_chain // len(stars_present)
        extra = per_chain - base * len(stars_present)
        picks = []
        for i, s in enumerate(stars_present):
            k = base + (1 if i < extra else 0)
            sub = grp[grp["stars"] == s]
            picks.append(sub.sample(n=min(k, len(sub)), random_state=seed))
        taken = pd.concat(picks) if picks else grp.head(0)
        if len(taken) < per_chain:
            rest = grp[~grp.index.isin(taken.index)].sort_values("date", ascending=False)
            taken = pd.concat([taken, rest.head(per_chain - len(taken))])
        out.append(taken.head(per_chain))
    sample = pd.concat(out).reset_index(drop=True) if out else rev.head(0)
    sample["risk_tier"] = "n/a"
    return sample


def _load_checkpoint(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["review_id"]] = rec
    return done


def run_aspects(args: argparse.Namespace) -> dict:
    T = pd.Timestamp(args.landmark)
    reviews = io.read_parquet(args.reviews)
    risk_scores = _read_optional(args.risk_scores)
    if risk_scores.empty:
        print("[warn] risk_scores not found; aspects phase needs stage 05 output.")
        return {}

    if getattr(args, "per_chain", 0):
        locations = io.read_parquet(args.locations)
        sample = build_per_chain_sample(reviews, locations, T, args.per_chain)
        print(
            f"Aspect sample (LIVE): {len(sample)} reviews, {args.per_chain}/chain "
            f"stratified by star rating across {sample['chain'].nunique()} chains."
        )
    else:
        sample = build_aspect_sample(reviews, risk_scores, T, args.sample_size)
        print(
            f"Aspect sample: {len(sample)} reviews across "
            f"{sample['business_id'].nunique()} locations."
        )

    done = _load_checkpoint(args.checkpoint)
    print(f"Resuming from checkpoint with {len(done)} completed reviews.")

    if not args.dry_run:
        Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = list(done.values())
    ckpt = None if args.dry_run else open(args.checkpoint, "a", encoding="utf-8")
    try:
        for _, row in sample.iterrows():
            rid = str(row["review_id"])
            if rid in done:
                continue
            analysis = aspects.analyze_review(str(row["text"]), provider=args.provider)
            rec = {
                "review_id": rid,
                "business_id": row["business_id"],
                "risk_tier": row.get("risk_tier"),
                "actual_stars": float(row["stars"]),
                "predicted_stars": analysis["predicted_stars"],
                "overall_score": analysis["overall_score"],
                "aspect_predicted_stars": analysis.get("aspect_predicted_stars"),
                "schema_failures": analysis.get("schema_failures", 0),
                "dropped_quotes": analysis["dropped_quotes"],
            }
            records.append(rec)
            if ckpt is not None:
                ckpt.write(json.dumps(rec) + "\n")
                ckpt.flush()
    finally:
        if ckpt is not None:
            ckpt.close()

    results = pd.DataFrame(records)
    if results.empty:
        return {"results": results}

    stars_eval = aspects.evaluate_predicted_stars(
        results["predicted_stars"], results["actual_stars"]
    )
    print("\nPREDICTED STARS vs ACTUAL")
    print("=" * 50)
    _ci_key = {
        "accuracy": "accuracy_ci",
        "macro_f1": "macro_f1_ci",
        "mae": "mae_ci",
        "pearson_r": "pearson_ci",
    }
    for k in ("accuracy", "macro_f1", "mae", "pearson_r"):
        lo, hi = stars_eval[_ci_key[k]]
        print(f"  {k:<12}: {stars_eval[k]:.4f}  (95% CI {lo:.4f}, {hi:.4f})")

    # High vs low risk contrast on per-location mean aspect score.
    loc_scores = results.groupby("business_id").agg(
        mean_score=("overall_score", "mean"),
        tier=("risk_tier", "first"),
    )
    high = loc_scores[loc_scores["tier"] == "High"]["mean_score"]
    low = loc_scores[loc_scores["tier"] == "Low"]["mean_score"]
    contrast = aspects.risk_contrast(high.to_numpy(), low.to_numpy())
    print(
        f"\nRisk contrast (high vs low): p={contrast['p_value']:.4g}, "
        f"effect={contrast['effect_size']:.3f}"
    )

    if not args.dry_run:
        io.write_parquet(results, Path(args.artifacts_dir) / config.ASPECT_RESULTS_FILE)
        eval_df = pd.DataFrame(
            [
                {
                    "accuracy": stars_eval["accuracy"],
                    "macro_f1": stars_eval["macro_f1"],
                    "mae": stars_eval["mae"],
                    "pearson_r": stars_eval["pearson_r"],
                    "n": stars_eval["n"],
                    "contrast_p": contrast["p_value"],
                    "contrast_effect": contrast["effect_size"],
                }
            ]
        )
        io.write_parquet(eval_df, Path(args.artifacts_dir) / config.ASPECT_EVAL_FILE)
        print(f"Wrote aspect results and eval to {args.artifacts_dir}.")

    return {"results": results, "stars_eval": stars_eval, "contrast": contrast}


# --------------------------------------------------------------------------- #
# RAG
# --------------------------------------------------------------------------- #


def run_rag(args: argparse.Namespace) -> dict:
    reviews = io.read_parquet(args.reviews)
    tips = _read_optional(args.tips)
    locations = io.read_parquet(args.locations)
    risk_scores = _read_optional(args.risk_scores)

    import os as _os

    _emb = Path(args.index_dir) / config.RAG_EMBEDDINGS_FILE
    _meta = Path(args.index_dir) / config.RAG_METADATA_FILE
    if _os.environ.get("LRR_REUSE_INDEX") == "1" and _emb.exists() and _meta.exists():
        # Reuse an index already built on disk (embedding takes ~30 min on CPU).
        _m = io.read_parquet(_meta)
        meta_info = {"n_chunks": len(_m), "bytes": _emb.stat().st_size + _meta.stat().st_size}
        print("Reusing existing RAG index on disk.")
    else:
        meta_info = rag.build_index(
            reviews, tips, locations, out_dir=args.index_dir, landmark=pd.Timestamp(args.landmark)
        )
    print(f"Built RAG index: {meta_info['n_chunks']} chunks, {meta_info['bytes'] / 1e6:.1f} MB.")

    embeddings, metadata = rag.load_index(args.index_dir)

    high_ids: list[str] = []
    if not risk_scores.empty:
        hi = risk_scores[risk_scores["risk_tier"] == "High"]
        if "risk_score" in hi.columns:
            hi = hi.sort_values("risk_score", ascending=False)
        # Cap live cost: cache answers for the N highest-risk locations (default 30).
        import os as _os

        cap = int(_os.environ.get("LRR_RAG_MAX_LOCATIONS", "30"))
        high_ids = hi["business_id"].tolist()[:cap]
    if not high_ids:
        high_ids = metadata["business_id"].dropna().unique().tolist()[:5]

    if args.dry_run:
        print(f"[dry-run] Would cache answers for {len(high_ids)} locations.")
        return {"index": meta_info, "high_ids": high_ids}

    cache = rag.build_rag_cache(embeddings, metadata, high_ids, provider=args.provider)
    io.write_parquet(cache, Path(args.artifacts_dir) / config.RAG_CACHE_FILE)

    eval_df = pd.DataFrame(
        [
            {
                "n_locations": len(high_ids),
                "n_answers": len(cache),
                "mean_groundedness": float(cache["groundedness_rate"].mean())
                if len(cache)
                else float("nan"),
                "retrieval_nonempty_rate": float((cache["n_citations"] > 0).mean())
                if len(cache)
                else float("nan"),
            }
        ]
    )
    io.write_parquet(eval_df, Path(args.artifacts_dir) / config.RAG_EVAL_FILE)
    print(
        f"Cached {len(cache)} answers; "
        f"mean groundedness {eval_df['mean_groundedness'].iloc[0]:.3f}."
    )
    return {"index": meta_info, "cache": cache, "eval": eval_df}


def run(args: argparse.Namespace) -> dict:
    _seed()
    out: dict = {}
    if args.phase in ("aspects", "all"):
        out["aspects"] = run_aspects(args)
    if args.phase in ("rag", "all"):
        out["rag"] = run_rag(args)
    return out


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
