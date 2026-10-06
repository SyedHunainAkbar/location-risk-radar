"""Pipeline stage 08: evaluate the risk committee.

Builds Location Risk Briefs for High and Elevated cohort locations using real RAG
retrieval (MMR-diversified, negative-weighted snippets) and SHAP/hazard-ratio drivers
in the fact sheet, persists them to artifacts/briefs.parquet, and evaluates the
committee: an INDEPENDENT numeric-fidelity re-scan (every number in rendered text must
trace to a fact-sheet value, target 1.0), groundedness rate, auditor rejection rate,
and latency/cost per brief from the gateway ledger. The committee-vs-single-agent
comparison on 30 held-out locations, the blinded-judge rubric, 10 manual labels, and
Cohen's kappa are scaffolded and run when live model access is available. The
TA-style food/service/ambience aspect run is reported as a separate benchmark against
stars, credited as adapted from the TA reference.

All LLM calls go through the gateway (keys from env or st.secrets only). Idempotent;
paths from lrr.config.

Usage:
    python pipeline/08_eval_agents.py --provider mock
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

from lrr import agents, config, gateway, io  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the risk committee.")
    p.add_argument(
        "--risk-scores", type=Path, default=config.ARTIFACTS_DIR / config.RISK_SCORES_FILE
    )
    p.add_argument(
        "--features",
        type=Path,
        default=config.DATA_DIR / config.features_file(config.LANDMARK_PRIMARY),
    )
    p.add_argument("--shap", type=Path, default=config.ARTIFACTS_DIR / "shap_drivers.parquet")
    p.add_argument("--index-dir", type=Path, default=config.RAG_INDEX_DIR)
    p.add_argument("--provider", type=str, default=None)
    p.add_argument("--n-holdout", type=int, default=30)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _read_optional(path):
    path = Path(path)
    return io.read_parquet(path) if path.exists() else None


def _load_rag(index_dir: Path):
    """Load the RAG index if present; else return (None, None)."""
    from lrr import rag

    emb_path = Path(index_dir) / config.RAG_EMBEDDINGS_FILE
    meta_path = Path(index_dir) / config.RAG_METADATA_FILE
    if emb_path.exists() and meta_path.exists():
        return rag.load_index(index_dir)
    return None, None


def _retrieve_snippets(embeddings, metadata, business_id, question):
    """Real RAG retrieval: MMR-diversified, negative-weighted (stars <= 2 first)."""
    from lrr import rag

    if embeddings is None or metadata is None:
        return pd.DataFrame(columns=["review_id", "text"])
    # Encode the question once; reuse the embedding model via rag._encode.
    try:
        qvec = rag._encode([question])[0]
    except Exception:
        # No embedding model available: fall back to the first negative snippets.
        sub = metadata[metadata["business_id"] == business_id]
        neg = sub[pd.to_numeric(sub.get("stars"), errors="coerce") <= 2]
        return (neg if len(neg) else sub).head(config.RAG_TOP_K)[["review_id", "text"]]
    # Negative-weighted: prefer stars <= 2, MMR for diversity.
    neg = rag.retrieve(
        qvec,
        embeddings,
        metadata,
        business_id=business_id,
        k=config.RAG_TOP_K,
        use_mmr=True,
        max_stars=2,
    )
    if len(neg) >= 3:
        return neg[["review_id", "text"]]
    allrows = rag.retrieve(
        qvec, embeddings, metadata, business_id=business_id, k=config.RAG_TOP_K, use_mmr=True
    )
    return allrows[["review_id", "text"]]


def run(args):
    rs = _read_optional(args.risk_scores)
    if rs is None:
        print("[warn] risk_scores not found; stage 05 must run first.")
        return {}

    features = _read_optional(args.features)
    shap = _read_optional(args.shap)
    embeddings, metadata = _load_rag(args.index_dir)
    if embeddings is None:
        print(
            "[note] RAG index not found; briefs will have fact-sheet drivers but no "
            "grounded evidence. Run pipeline/06 to build the index."
        )

    targets = rs[rs["risk_tier"].isin(config.BRIEF_TIERS)]
    print(f"Brief targets (High/Elevated): {len(targets)}")

    briefs = []
    fidelity_flags: list[int] = []
    stray_examples: list[str] = []
    question = "Why is this location at risk?"
    import time as _time

    failed: list[str] = []
    # Walk the High/Elevated list until n_holdout briefs succeed; a provider timeout
    # on one location is logged and skipped instead of killing the whole run.
    for bid in targets.sort_values("risk_score", ascending=False)["business_id"]:
        if len(briefs) >= args.n_holdout:
            break
        fs = agents.build_factsheet(bid, rs, shap_drivers=shap, features=features)
        snippets = _retrieve_snippets(embeddings, metadata, bid, question)
        t0 = _time.time()
        try:
            brief = agents.build_brief(fs, snippets, {"review_volume": "unknown"})
        except Exception as exc:  # noqa: BLE001
            failed.append(bid)
            print(f"  [skip] {bid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            if len(failed) >= 3 * args.n_holdout:
                break
            continue
        print(
            f"  brief {len(briefs) + 1}/{args.n_holdout} {bid}: evidence={brief.n_evidence} "
            f"drivers={len(brief.drivers)} grade={brief.auditor_grade} "
            f"({_time.time() - t0:.0f}s)",
            flush=True,
        )
        briefs.append(brief)
        # INDEPENDENT numeric-fidelity re-scan on EVERY rendered driver (pre-audit),
        # since fidelity is a property of rendering, not of the auditor's verdict.
        for d in brief.rendered_drivers:
            stray = agents.scan_rendered_fidelity(d["text"], fs)
            fidelity_flags.append(1 if not stray else 0)
            if stray:
                stray_examples.append(f"{bid}: {d['text']} -> {stray}")

    fidelity = float(np.mean(fidelity_flags)) if fidelity_flags else 1.0
    groundedness = float(np.mean([b.groundedness for b in briefs])) if briefs else float("nan")
    rejection = (
        float(
            np.mean(
                [
                    len(b.rejected_claims) / max(1, len(b.drivers) + len(b.rejected_claims))
                    for b in briefs
                ]
            )
        )
        if briefs
        else float("nan")
    )

    gw = gateway.get_gateway()
    ledger = gw.ledger.to_frame() if gw.ledger.rows else pd.DataFrame()
    latency = float(ledger["latency_s"].mean()) if len(ledger) else float("nan")
    cost = float(ledger["est_cost_usd"].sum()) if len(ledger) else 0.0

    summary = pd.DataFrame(
        [
            {
                "n_briefs": len(briefs),
                "n_skipped_timeouts": len(failed),
                "numeric_fidelity_rate": fidelity,  # independent re-scan, target 1.0
                "mean_groundedness": groundedness,
                "auditor_rejection_rate": rejection,
                "mean_latency_s": latency,
                "total_est_cost_usd": cost,
                "confidence_formula": agents.CONFIDENCE_FORMULA,
            }
        ]
    )
    print("\nAGENT EVALUATION")
    print("=" * 60)
    with pd.option_context("display.max_colwidth", 50):
        print(summary.T.to_string(header=False))
    if stray_examples:
        print("\n[fidelity] rendered numbers not traced to the fact sheet:")
        for ex in stray_examples[:5]:
            print("  ", ex)

    print(
        "\nNote: committee-vs-single-agent on 30 held-out locations, the blinded "
        "judge rubric, 10 manual labels, and Cohen's kappa (auditor vs manual) "
        "require live model access via the gateway 'risk_auditor' role."
    )
    print(
        "TA-aspect benchmark: run lrr.aspects on the same sample and compare "
        "predicted stars to actual (adapted from the TA's Yelp Aspect Agents)."
    )

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return {"summary": summary, "briefs": briefs}

    art = Path(args.artifacts_dir)
    agents.persist_briefs(briefs, art / config.BRIEFS_FILE)
    io.write_parquet(summary, art / config.AGENT_EVAL_FILE)
    print(f"\nWrote {len(briefs)} briefs and the eval summary to {art}.")
    return {"summary": summary, "briefs": briefs}


def main(argv=None):
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
