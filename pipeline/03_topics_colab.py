"""Offline (Colab) complaint-topic modeling (LA / stage 03).

For each cluster we take the 1-2 star cohort reviews, embed them with MiniLM
(cached), fit three HDBSCAN settings under the locked UMAP config, reduce outliers,
and pick the setting with the best coherence/diversity. We then fit a zero-shot
model seeded with the operational complaint themes and compare. Topics are labeled
via the LLM (temperature 0) with human overrides from a CSV. We write per-cluster
topic tables, topics-over-time by quarter, topics for closed vs open, per-business
pre-landmark topic shares for the survival model, and small HTML plots.

OFFLINE ONLY. BERTopic, UMAP, HDBSCAN, and gensim run here, never in the app.
Idempotent; all paths and constants come from ``lrr.config``.

Usage (Colab):
    !python pipeline/03_topics_colab.py --provider mock
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

from lrr import config, io  # noqa: E402
from lrr import topics as T  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline complaint topic modeling.")
    p.add_argument("--reviews", type=Path, default=config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    p.add_argument(
        "--locations", type=Path, default=config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE
    )
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--cache-dir", type=Path, default=config.EMBEDDING_CACHE_DIR)
    p.add_argument("--landmark", type=str, default=config.LANDMARK_PRIMARY.isoformat())
    p.add_argument(
        "--provider",
        type=str,
        default=None,
        help="LLM provider for labeling: nvidia | voyager | mock.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def _seed() -> None:
    random.seed(config.SEED)
    np.random.seed(config.SEED)


def load_negative_reviews(reviews_path: Path, locations_path: Path) -> pd.DataFrame:
    """Return 1-2 star cohort reviews with cluster and closed/open status."""
    reviews = io.read_parquet(reviews_path)
    locations = io.read_parquet(locations_path)
    loc = locations[["business_id", "cluster", "is_open"]].copy()
    loc["closed_open"] = np.where(loc["is_open"] == 0, "closed", "open")
    merged = reviews.merge(
        loc[["business_id", "cluster", "closed_open"]], on="business_id", how="inner"
    )
    merged["stars"] = pd.to_numeric(merged["stars"], errors="coerce")
    negative = merged[merged["stars"].isin(config.NEGATIVE_STARS)].copy()
    negative["date"] = pd.to_datetime(negative["date"], errors="coerce")
    return negative.reset_index(drop=True)


def _topic_words(model, topic_ids) -> dict[int, list[str]]:
    words: dict[int, list[str]] = {}
    for tid in topic_ids:
        pairs = model.get_topic(tid) or []
        words[tid] = [w for w, _ in pairs]
    return words


def fit_best_unsupervised(texts, embeddings):
    """Fit 3 HDBSCAN settings, reduce outliers, choose by coherence+diversity."""
    results = []
    for mcs in config.HDBSCAN_MIN_CLUSTER_SIZES:
        model = T.build_bertopic(min_cluster_size=mcs)
        topic_ids, _ = model.fit_transform(texts, embeddings)
        topic_ids = model.reduce_outliers(texts, topic_ids)
        model.update_topics(texts, topics=topic_ids, vectorizer_model=T.build_vectorizer())
        info = model.get_topic_info()
        valid = [t for t in info["Topic"].tolist() if t != -1]
        tw = _topic_words(model, valid)
        diversity = T.topic_diversity(list(tw.values()))
        coherence = T.topic_coherence(list(tw.values()), texts)
        score = np.nanmean([coherence, diversity])
        results.append(
            {
                "min_cluster_size": mcs,
                "coherence": coherence,
                "diversity": diversity,
                "score": score,
                "model": model,
                "assignments": topic_ids,
            }
        )
        print(
            f"  mcs={mcs}: coherence={coherence:.4f} diversity={diversity:.4f} topics={len(valid)}"
        )
    best = max(results, key=lambda r: np.nan_to_num(r["score"], nan=-1))
    print(f"  -> chose min_cluster_size={best['min_cluster_size']}")
    return best, pd.DataFrame(
        [
            {k: r[k] for k in ("min_cluster_size", "coherence", "diversity", "score")}
            for r in results
        ]
    )


def fit_zero_shot(texts, embeddings):
    """Fit the seeded zero-shot model and score it."""
    model = T.build_bertopic(min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZES[0], zero_shot=True)
    topic_ids, _ = model.fit_transform(texts, embeddings)
    topic_ids = model.reduce_outliers(texts, topic_ids)
    info = model.get_topic_info()
    valid = [t for t in info["Topic"].tolist() if t != -1]
    tw = _topic_words(model, valid)
    return {
        "model": model,
        "assignments": topic_ids,
        "coherence": T.topic_coherence(list(tw.values()), texts),
        "diversity": T.topic_diversity(list(tw.values())),
    }


def build_topics_table(model, provider, overrides_csv: Path) -> pd.DataFrame:
    """Assemble topic, label, top_words, size with LLM labels + overrides."""
    info = model.get_topic_info()
    rows = []
    for _, r in info.iterrows():
        tid = int(r["Topic"])
        pairs = model.get_topic(tid) or []
        top_words = [w for w, _ in pairs][: config.TOPIC_TOP_K]
        reps = r.get("Representative_Docs", []) or []
        rows.append(
            {
                "topic": tid,
                "top_words": top_words,
                "rep_docs": list(reps)[: config.TOPIC_LABEL_N_DOCS],
                "size": int(r["Count"]),
            }
        )
    table = pd.DataFrame(rows)
    table = T.label_topics_llm(table, provider=provider)
    table = T.apply_label_overrides(table, overrides_csv)
    table["top_words"] = table["top_words"].map(lambda ws: ", ".join(ws))
    return table[["topic", "label", "top_words", "size"]]


def run(args: argparse.Namespace) -> dict:
    _seed()
    landmark = pd.Timestamp(args.landmark)
    negative = load_negative_reviews(args.reviews, args.locations)
    art = Path(args.artifacts_dir)
    data = Path(args.data_dir)
    out: dict = {}

    all_shares: list[pd.DataFrame] = []
    for cluster in ("Fast Food", "Non-Fast Food"):
        sub = negative[negative["cluster"] == cluster].reset_index(drop=True)
        print(f"\n=== Cluster: {cluster} ({len(sub):,} negative reviews) ===")
        if sub.empty:
            continue
        texts = sub["text"].astype(str).tolist()
        embeddings = T.embed_corpus(texts, cluster, args.cache_dir)

        best, settings = fit_best_unsupervised(texts, embeddings)
        zero = fit_zero_shot(texts, embeddings)
        print(f"  zero-shot: coherence={zero['coherence']:.4f} diversity={zero['diversity']:.4f}")

        chosen = best["model"]
        chosen_assignments = best["assignments"]

        topics_table = build_topics_table(
            chosen, args.provider, art / config.TOPIC_LABELS_OVERRIDE_FILE
        )

        sub = sub.assign(topic=chosen_assignments)
        sub["quarter"] = sub["date"].dt.to_period("Q").astype(str)
        over_time = sub.groupby(["quarter", "topic"]).size().reset_index(name="count")
        closed_open = sub.groupby(["closed_open", "topic"]).size().reset_index(name="count")

        shares = T.compute_topic_shares(sub[["business_id", "date", "topic"]], landmark=landmark)
        shares.insert(1, "cluster", cluster)
        all_shares.append(shares)

        out[cluster] = {
            "topics": topics_table,
            "settings": settings,
            "zero_shot": zero,
            "over_time": over_time,
            "closed_open": closed_open,
        }

        if not args.dry_run:
            io.write_parquet(topics_table, art / config.topics_file(cluster))
            slug = cluster.lower().replace(" ", "_").replace("-", "_")
            io.write_parquet(over_time, art / f"topics_over_time_{slug}.parquet")
            io.write_parquet(closed_open, art / f"topics_closed_open_{slug}.parquet")
            _save_html_plots(chosen, art, slug)

    if all_shares and not args.dry_run:
        combined = pd.concat(all_shares, ignore_index=True).fillna(0.0)
        io.write_parquet(combined, data / config.TOPIC_SHARES_FILE)
        print(
            f"\nWrote per-business pre-landmark topic shares to "
            f"{data / config.TOPIC_SHARES_FILE} ({len(combined)} businesses)."
        )

    if args.dry_run:
        print("\n[dry-run] No files written.")
    return out


def _save_html_plots(model, artifacts_dir: Path, slug: str) -> None:
    """Save small static BERTopic plots as HTML for the notebook."""
    try:
        model.visualize_barchart(top_n_topics=12).write_html(
            str(artifacts_dir / f"topics_barchart_{slug}.html")
        )
        model.visualize_topics().write_html(str(artifacts_dir / f"topics_map_{slug}.html"))
    except Exception as exc:  # plotting is best-effort
        print(f"[warn] plot generation skipped for {slug}: {exc}")


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
