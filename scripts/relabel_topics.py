"""Regenerate BERTopic topic labels with the live NVIDIA gateway.

The labels written by stage 03 came from the offline mock (the gateway was silently
falling back). This script relabels every topic in both cluster tables using the
topic's top words plus 5 representative negative reviews from that cluster, with
strict live mode on (no mock fallback). It also flags brand or menu descriptor
topics so they can be mapped to theme_other.

Run from the project root:
    $env:LRR_STRICT_LIVE="1"; python scripts/relabel_topics.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("LRR_STRICT_LIVE", "1")

from lrr import config, llm, topics  # noqa: E402

CLUSTERS = {"fast_food": "Fast Food", "non_fast_food": "Non-Fast Food"}

BRAND_SYSTEM = (
    "You classify a restaurant-review topic. Answer with exactly one word: "
    '"complaint" if the topic describes a customer problem or experience '
    "(service, wait, order errors, food quality, cleanliness, price, staff, "
    'management), or "brand" if it mainly names a restaurant chain, its menu items, '
    "or products without describing a problem."
)


def _words(v) -> list[str]:
    if isinstance(v, str):
        return [w.strip() for w in v.split(",") if w.strip()]
    return [str(w) for w in list(v)]


def _rep_docs(neg: pd.DataFrame, words: list[str], n: int = 5) -> list[str]:
    top = [w.lower() for w in words[:6]]
    text = neg["text"].astype(str)
    low = text.str.lower()
    score = sum(low.str.contains(re.escape(w), regex=True).astype(int) for w in top)
    idx = score.sort_values(ascending=False).index[:n]
    return text.loc[idx].tolist()


def main() -> None:
    reviews = pd.read_parquet(config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    locs = pd.read_parquet(config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE)
    reviews = reviews.merge(locs[["business_id", "cluster"]], on="business_id", how="left")
    reviews = reviews[pd.to_numeric(reviews["stars"], errors="coerce") <= 2]

    for slug, cluster in CLUSTERS.items():
        path = config.ARTIFACTS_DIR / f"topics_{slug}.parquet"
        df = pd.read_parquet(path)
        neg = reviews[reviews["cluster"] == cluster]
        labels, kinds = [], []
        for _, row in df.iterrows():
            if int(row["topic"]) == -1:
                labels.append("Outliers")
                kinds.append("other")
                continue
            words = _words(row["top_words"])
            docs = _rep_docs(neg, words)
            raw = llm.chat(
                topics._label_prompt(words, docs),
                temperature=0.0,
                max_tokens=16,
                provider="nvidia",
            )
            label = topics._clean_label(raw)
            verdict = llm.chat(
                [
                    {"role": "system", "content": BRAND_SYSTEM},
                    {"role": "user", "content": f"Label: {label}\nTop words: {', '.join(words[:10])}"},
                ],
                temperature=0.0,
                max_tokens=4,
                provider="nvidia",
            )
            kind = "brand" if "brand" in str(verdict).lower() else "complaint"
            labels.append(label)
            kinds.append(kind)
            print(f"[{cluster}] topic {int(row['topic']):>3}: {label:<35} ({kind})")
        df["label"] = labels
        df["topic_kind"] = kinds
        assert not df["label"].astype(str).str.contains(r"\[mock:").any(), "mock label remains"
        df.to_parquet(path, index=False)
        print(f"Wrote {path.name}: {len(df)} topics, {kinds.count('brand')} brand/menu topics.")


if __name__ == "__main__":
    main()
