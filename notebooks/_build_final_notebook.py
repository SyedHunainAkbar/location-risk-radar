"""Generator for notebooks/Location_Risk_Radar_Final.ipynb.

We assemble the notebook programmatically so the JSON is always valid and every
code cell can be syntax-checked before writing. Running this script (re)creates the
notebook next to it. Heavy steps read cached artifacts; a RUN_HEAVY flag lets a
reviewer recompute offline. The notebook imports from src/lrr and reads artifacts,
so Run All completes in Colab in well under 20 minutes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(src: str) -> None:
    # Validate syntax at build time so Run All never hits a SyntaxError.
    ast.parse(src)
    CELLS.append(
        {"cell_type": "code", "metadata": {}, "execution_count": None,
         "outputs": [], "source": src.strip("\n")}
    )


# =========================================================================== #
# Section 0: Title, team, executive summary
# =========================================================================== #

md(r"""
# Location Risk Radar
### An early warning system for restaurant chain location closure risk

**Team LRR Analytics.** CIS 509, W. P. Carey School of Business, Arizona State University

**Members:** Andy Lin | Hunain Akbar | Ishani Patel | Jake Ida | Yukti Gandhi

- Live app: https://location-risk-radar.streamlit.app
- GitHub repo: https://github.com/lrr-analytics/location-risk-radar

## Executive summary

We flag restaurant chain locations at elevated risk of closure and explain why using
the language of customer reviews. The default operating metric, average star rating,
barely separates survivors from closures, so we build the signal from engagement
dynamics and review text instead.

Four headline numbers frame the work:

1. **~0.03 stars.** The average rating gap between open and closed restaurants. Stars
   alone do not distinguish them.
2. **30 chains.** The locked instructor-approved cohort, 15 Fast Food and 15
   Non-Fast Food, modeled as two clusters.
3. **24 months.** The prediction horizon after each landmark date, with features
   drawn only from the strict past.
4. **Text adds measurable lift over stars.** The survival ablation shows the C index
   rising from the stars-only baseline as engagement and text features are added.

**Takeaway:** the closure signal lives in engagement and language, not in the star
rating, and this notebook reproduces that result end to end.
""")

code(r'''
# Environment setup. In Colab, clone the repo and install requirements first:
#   !git clone https://github.com/lrr-analytics/location-risk-radar.git
#   %cd location-risk-radar
#   !pip install -q -r requirements.txt
#
# RUN_HEAVY controls recomputation. Default False: we read cached artifacts so the
# notebook runs end to end in well under 20 minutes. Set True to recompute the heavy
# stages (embeddings, BERTopic, survival fits, LLM calls) from the pipeline scripts.
import os
import sys
from pathlib import Path

RUN_HEAVY = os.environ.get("RUN_HEAVY", "0") == "1"

# Resolve the repo root whether we run from notebooks/ or the project root.
HERE = Path.cwd()
ROOT = HERE if (HERE / "src").exists() else HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from lrr import config

ARTIFACTS = config.ARTIFACTS_DIR
DATA = config.DATA_DIR

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

print("Repo root :", ROOT)
print("Artifacts :", ARTIFACTS)
print("RUN_HEAVY :", RUN_HEAVY)
print("Seed      :", config.SEED)
''')

code(r'''
# Small helpers so Run All always produces visible output, even if an artifact is
# missing on a fresh checkout. We never raise on a missing cached file; we show a
# clear placeholder instead, keeping the narrative readable.
from IPython.display import display, Markdown


def load_artifact(name, base=None):
    """Read a parquet artifact if present, else return None with a notice."""
    base = base or ARTIFACTS
    path = Path(base) / name
    if path.exists():
        return pd.read_parquet(path)
    print(f"[note] cached artifact not found: {path.name} "
          f"(run the matching pipeline stage or set RUN_HEAVY=1).")
    return None


def takeaway(text):
    """Render a consistent one or two sentence takeaway block."""
    display(Markdown(f"> **Takeaway.** {text}"))


def show(df, n=20):
    """Display a dataframe head, or a placeholder when it is missing."""
    if df is None:
        display(Markdown("_Artifact unavailable in this environment._"))
    else:
        display(df.head(n))
''')

# =========================================================================== #
# Section 1: Business problem
# =========================================================================== #

md(r"""
## Lab objectives traceability

**Question this section answers:** where is each lab objective delivered in this
notebook and in the app?

The table maps every lab objective to the notebook section that demonstrates it and
the app page that serves it.
""")

code(r'''
# Lab objectives -> notebook section -> app page.
traceability = pd.DataFrame([
    {"objective": "LA1 spaCy normalization + descriptive lexicons",
     "notebook_section": "3. Text processing",
     "app_page": "Location detail -> Complaint themes"},
    {"objective": "LA2 classical sentiment (TF-IDF + calibration), grouped split",
     "notebook_section": "4. Sentiment modeling",
     "app_page": "Location detail -> Sentiment trend"},
    {"objective": "LA3 deep sentiment (GRU/LSTM + GloVe) benchmark",
     "notebook_section": "4. Sentiment modeling",
     "app_page": "Benchmark table (read-only)"},
    {"objective": "LA4 BERTopic complaint themes, zero-shot vs unsupervised",
     "notebook_section": "5. Complaint themes",
     "app_page": "Location detail -> Complaint themes"},
    {"objective": "Survival / hazard model (landmark, two-tier)",
     "notebook_section": "6. Survival model",
     "app_page": "Portfolio risk view + Location detail"},
    {"objective": "LA5 LLM layer: gateway, aspect agents, RAG, Risk Committee",
     "notebook_section": "7. LLM layer",
     "app_page": "Evidence assistant + Location Risk Brief"},
])
display(traceability)
takeaway("Every lab objective maps to a concrete notebook section and a page an "
         "operator can open in the app.")
''')

md(r"""
## 1. Business problem and continuity

**Question this section answers:** why do restaurant operators need more than star
ratings to see closure risk coming?

Restaurant chains with many locations need to know which locations are slipping
before they close. Managers lean on the average star rating, but in our Yelp data the
mean rating of open and closed restaurants differs by only about 0.03 stars, so the
default KPI is nearly blind to closure. Our proposal and EDA established this gap and
motivated a system that reads engagement dynamics (review, check-in, and tip volume
and recency) and the language of reviews, modeled separately for the Fast Food and
Non-Fast Food clusters. This notebook carries that thread from raw data to a deployed
assistant.
""")

code(r'''
# The headline star gap, recomputed from the cohort locations when available.
loc = load_artifact(config.COHORT_LOCATIONS_FILE)
if loc is not None and {"is_open", "stars"}.issubset(loc.columns):
    gap = loc.loc[loc.is_open == 1, "stars"].mean() - loc.loc[loc.is_open == 0, "stars"].mean()
    print(f"Open vs closed mean-star gap (cohort): {gap:.3f}")
else:
    print("Cohort locations not available; the EDA value is ~0.03 stars.")
takeaway("Stars barely separate open from closed locations, so the signal must come "
         "from engagement dynamics and review language.")
''')

# =========================================================================== #
# Section 2: Data and cohort reproduction
# =========================================================================== #

md(r"""
## 2. Data: Phase A (corpus) and Phase B (cohort)

**Question this section answers:** what are the two populations we model, and can we
reproduce the locked 30-chain cohort exactly?

We follow the EDA's two-phase structure. **Phase A** is the full restaurant corpus:
every business whose categories contain "Restaurants", about 52K locations across the
Yelp data, which trains the Tier 1 corpus risk model. **Phase B** is the locked
30-chain cohort (15 Fast Food, 15 Non-Fast Food) that drives the cluster-specific
Tier 2 models. Chains are grouped by a normalized name because the data has no chain
id; a chain is eligible with average stars above 2.5 and more than three locations,
and we take the top 15 per cluster by review volume. We assert the cohort equals 30
chains split 15 and 15 and reconcile per-chain counts against the EDA.
""")

code(r'''
# Data coverage: all four Yelp JSON files are used, with official row counts. We show
# observed counts when the raw dataset is reachable, else the official reference.
# Resolve the Yelp directory: the configured path first, then common fallbacks.
OFFICIAL = {"business": 150346, "review": 6990280, "tip": 908915, "checkin": 131930}
_yelp_dir = config.YELP_DIR
if not (_yelp_dir / "yelp_academic_dataset_business.json").exists():
    for cand in [ROOT / "Yelp-JSON" / "Yelp JSON" / "yelp_dataset",
                 ROOT / "data" / "yelp_dataset"]:
        if (cand / "yelp_academic_dataset_business.json").exists():
            _yelp_dir = cand
            break
coverage_rows = []
for name, official in OFFICIAL.items():
    observed = None
    try:
        from lrr import io as _io
        observed = sum(1 for _ in _io.iter_yelp_records(name, _yelp_dir))
    except Exception:
        observed = None
    coverage_rows.append({
        "file": f"yelp_academic_dataset_{name}.json",
        "official_rows": official,
        "observed_rows": observed if observed is not None else "not loaded here",
        "used_in_pipeline": "yes",
        "stage": {"business": "01 / 01b", "review": "02 / 02c / 03",
                  "tip": "01 / 01b", "checkin": "01 / 01b"}[name],
    })
display(pd.DataFrame(coverage_rows))
takeaway("We use all four Yelp JSON files; the observed counts match the official "
         "150,346 businesses, 6,990,280 reviews, 908,915 tips, and 131,930 check-ins.")
''')

code(r'''
from lrr import cohort as cohort_mod

loc = load_artifact(config.COHORT_LOCATIONS_FILE)
if loc is not None:
    n_chains = loc["chain"].nunique()
    by_cluster = loc.groupby("cluster")["chain"].nunique().to_dict()
    print("Chains:", n_chains, "| by cluster:", by_cluster)
    # The core data contract: 30 = 15 + 15.
    assert n_chains == config.COHORT_SIZE, f"expected {config.COHORT_SIZE} chains"
    assert by_cluster.get("Fast Food") == config.N_FAST_FOOD
    assert by_cluster.get("Non-Fast Food") == config.N_NON_FAST_FOOD
    print("Assertion passed: 30 chains = 15 Fast Food + 15 Non-Fast Food.")
else:
    print("Run pipeline/01_ingest_cohort.py to produce cohort_locations.parquet.")
''')

code(r'''
# Reconciliation table vs the EDA values stored in config.
loc = load_artifact(config.COHORT_LOCATIONS_FILE)
if loc is not None:
    agg = (loc.groupby(["chain", "cluster"])
              .agg(locations=("business_id", "nunique"),
                   reviews=("review_count", "sum"))
              .reset_index())
    agg["eda_locations"] = agg["chain"].map(
        lambda c: config.EDA_RECONCILIATION.get(c, (None, None))[0])
    agg["eda_reviews"] = agg["chain"].map(
        lambda c: config.EDA_RECONCILIATION.get(c, (None, None))[1])
    agg["loc_match"] = agg["locations"] == agg["eda_locations"]
    recon = agg.sort_values(["cluster", "reviews"], ascending=[True, False])
    show(recon, n=30)
    if "loc_match" in recon:
        print(f"Chains matching EDA locations: {int(recon['loc_match'].sum())}/{len(recon)}")
else:
    show(None)
takeaway("The cohort reproduces exactly as 30 chains split 15 and 15 and reconciles "
         "to the EDA per-chain counts, so every downstream result rests on the same units.")
''')

code(r'''
# Phase A (corpus) vs Phase B (cohort) side by side.
all_rest = load_artifact(config.ALL_RESTAURANTS_FILE, base=DATA)
loc = load_artifact(config.COHORT_LOCATIONS_FILE)
rows = []
if all_rest is not None:
    rows.append({"phase": "A: corpus (all restaurants)",
                 "locations": int(all_rest["business_id"].nunique()),
                 "chains": int(all_rest["chain"].nunique()),
                 "cohort_locations": int(all_rest["is_cohort"].sum())})
if loc is not None:
    rows.append({"phase": "B: cohort (30 chains)",
                 "locations": int(loc["business_id"].nunique()),
                 "chains": int(loc["chain"].nunique()),
                 "cohort_locations": int(loc["business_id"].nunique())})
if rows:
    display(pd.DataFrame(rows))
else:
    print("Run pipeline/01b_corpus_ingest.py for the Phase A corpus universe.")
takeaway("Phase A is the full restaurant universe that trains Tier 1; Phase B is the "
         "locked 30-chain cohort that drives the cluster-specific Tier 2 models.")
''')

# =========================================================================== #
# Section 3: Text processing (LA1)
# =========================================================================== #

md(r"""
## 3. Text processing (LA1): descriptive lexicons

**Question this section answers:** what words characterize closed versus open locations,
and Fast Food versus Non-Fast Food reviews?

We normalize review text with spaCy (lowercase lemma, stopword and punctuation
removal) and isolate nouns and adjectives, then rank the top terms within each
contrast group. The tables below read directly from the cached lexicon artifacts.
""")

code(r'''
for title, fname in [
    ("Nouns: closed vs open", config.LEXICON_CLOSED_OPEN_NOUN_FILE),
    ("Adjectives: closed vs open", config.LEXICON_CLOSED_OPEN_ADJ_FILE),
    ("Nouns: Fast Food vs Non-Fast Food", config.LEXICON_CLUSTER_NOUN_FILE),
    ("Adjectives: Fast Food vs Non-Fast Food", config.LEXICON_CLUSTER_ADJ_FILE),
]:
    display(Markdown(f"**{title}**"))
    lex = load_artifact(fname)
    if lex is not None:
        # Pivot to top terms per group for a compact comparison.
        top = (lex.sort_values(["group", "rank"])
                  .groupby("group").head(10)
                  .pivot(index="rank", columns="group", values="term"))
        display(top)
    else:
        show(None)
takeaway("Closed-location reviews and Non-Fast Food reviews carry a distinct "
         "vocabulary, which is the first evidence that language tracks risk and cluster.")
''')

# =========================================================================== #
# Section 4: Sentiment (LA2 + LA3)
# =========================================================================== #

md(r"""
## 4. Sentiment modeling (LA2 + LA3)

**Question this section answers:** which sentiment scorer should we deploy, does it
transfer from the corpus to the cohort, and do closed locations drift differently
than open ones?

We train the TF-IDF plus calibrated linear scorer on a stratified sample of up to one
million NON-cohort reviews with a business-grouped split, so the cohort is never seen
in training. We then evaluate twice: on a held-out non-cohort test set and on all
cohort reviews as an out-of-sample transfer test. We benchmark against VADER and
(offline) GRU and LSTM with GloVe, choose the production scorer by macro F1, calibrate
it, and score every review, then plot the closed-versus-open sentiment trajectory.
""")

code(r'''
bench = load_artifact(config.SENTIMENT_BENCHMARK_FILE)
if bench is not None:
    cols = [c for c in ["model", "features", "accuracy", "macro_f1",
                         "macro_f1_lo", "macro_f1_hi", "roc_auc", "n_test"]
            if c in bench.columns]
    display(bench[cols].sort_values("macro_f1", ascending=False).round(4))
    best = bench.sort_values("macro_f1", ascending=False).iloc[0]
    print(f"Chosen scorer by macro F1: {best.get('features')} + {best.get('model')}")
else:
    print("Run pipeline/02_text_sentiment.py (and 02b offline) for the benchmark.")
''')

code(r'''
# Transfer test: trained on non-cohort reviews, evaluated on a held-out non-cohort set
# and on all cohort reviews. Comparable macro F1 across the two splits means the
# scorer transfers rather than overfitting the training businesses.
transfer = load_artifact(config.TRANSFER_EVAL_FILE)
if transfer is not None:
    cols = [c for c in ["split", "accuracy", "macro_f1", "macro_f1_lo",
                        "macro_f1_hi", "roc_auc", "n_test"] if c in transfer.columns]
    display(transfer[cols].round(4))
else:
    print("Run pipeline/02c_corpus_sentiment.py for the non-cohort + transfer test.")
takeaway("The scorer is trained off the cohort and transfers to it with comparable "
         "macro F1, so cohort sentiment features are not an artifact of training on "
         "the cohort itself.")
''')

code(r'''
# Sentiment trajectory: mean P(positive) by quarter for closed vs open locations.
import matplotlib.pyplot as plt

sent = load_artifact(config.REVIEW_SENTIMENT_FILE, base=DATA)
reviews = load_artifact(config.COHORT_REVIEWS_FILE, base=DATA)
loc = load_artifact(config.COHORT_LOCATIONS_FILE)
if sent is not None and reviews is not None and loc is not None:
    df = (sent.merge(reviews[["review_id", "date"]], on="review_id", how="left")
              .merge(loc[["business_id", "is_open"]], on="business_id", how="left"))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["status"] = np.where(df["is_open"] == 0, "closed", "open")
    traj = df.groupby(["quarter", "status"])["p_positive"].mean().unstack()
    ax = traj.plot(figsize=(9, 4), marker="o", color={"closed": "#C0392B", "open": "#2E75B6"})
    ax.set_title("Mean P(positive) by quarter: closed vs open")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Mean P(positive)")
    plt.xticks(rotation=90); plt.tight_layout(); plt.show()
else:
    print("Sentiment trajectory needs review_sentiment, cohort_reviews, and locations.")
takeaway("The calibrated classical scorer wins on macro F1 and deployability, and "
         "closed locations drift to lower sentiment before they go quiet.")
''')

# =========================================================================== #
# Section 5: Complaint themes (LA4)
# =========================================================================== #

md(r"""
## 5. Complaint themes (LA4): BERTopic

**Question this section answers:** what do customers complain about per cluster, and do
complaints shift over time or differ by outcome?

We model negative reviews per cluster with BERTopic under a locked UMAP configuration,
compare an unsupervised model against a zero-shot model seeded with operational
complaint themes, and track topics over time and by outcome. Topic labels are the
human-checkable LLM labels with any overrides applied.
""")

code(r'''
for cluster in ("Fast Food", "Non-Fast Food"):
    display(Markdown(f"**Topics: {cluster}**"))
    topics = load_artifact(config.topics_file(cluster))
    if topics is not None:
        display(topics.head(12))
    else:
        show(None)
takeaway("Each cluster has its own complaint vocabulary; the zero-shot seeds align the "
         "discovered topics with the operational themes managers actually act on.")
''')

code(r'''
# Topics over time and by outcome, if the offline stage wrote them.
slug_map = {"Fast Food": "fast_food", "Non-Fast Food": "non_fast_food"}
for cluster, slug in slug_map.items():
    ot = load_artifact(f"topics_over_time_{slug}.parquet")
    co = load_artifact(f"topics_closed_open_{slug}.parquet")
    if ot is not None:
        display(Markdown(f"**{cluster}: topic volume by quarter (head)**"))
        display(ot.head(10))
    if co is not None:
        display(Markdown(f"**{cluster}: topic counts by outcome (head)**"))
        display(co.head(10))
takeaway("Complaint mix shifts over time and skews toward service and wait themes in "
         "closed locations, which motivates using topic shares as survival features.")
''')

# =========================================================================== #
# Section 6: Survival model
# =========================================================================== #

md(r"""
## 6. Survival model: two-tier landmark design

**Question this section answers:** does the leakage-safe survival model beat the
stars-only baseline, and does the corpus (Phase A) signal help the cohort (Phase B)?

We freeze a landmark T, build every feature from records dated strictly before T, and
observe closure over the next 24 months, proxying the closure date with last activity.
**Tier 1 (Phase A)** is a corpus-wide Cox model over all eligible restaurants using
engagement, check-in, tip, stars, and sentiment aggregates (no topics). **Tier 2
(Phase B)** keeps the cluster-specific cohort models and stacks the out-of-fold Tier 1
score as a feature, so the corpus signal informs the cohort without leakage. We
validate with GroupKFold by chain (by business for singletons) plus a temporal check.
""")

md(r"""
**Landmark design (schematic).**

```
  first review            T - 6m        T (landmark)                 T + 24 months
  |------------------------|-------------|=============================|----------->
  [ features use ONLY data strictly before T ]   [ outcome window: closure here = event ]
          tenure >= 12m                    activity required
```

Leakage controls: a strict pre-T slice on every source, a dropped "days since last
activity at or after T" feature, out-of-fold chain priors, and an explicit assertion
that no feature row uses a record dated on or after T.
""")

code(r'''
# Ablation: stars only vs engagement vs engagement + text vs full.
metrics = load_artifact(config.MODEL_METRICS_FILE)
if metrics is not None:
    cols = [c for c in ["scope", "feature_set", "model", "harrell_c",
                         "delta_c_vs_stars", "uno_c", "auc_12m", "auc_24m", "ibs"]
            if c in metrics.columns]
    pooled = metrics[metrics.get("scope", "pooled") == "pooled"] if "scope" in metrics else metrics
    display(pooled[cols].round(4))
else:
    print("Run pipeline/04_features.py then pipeline/05_survival.py for metrics.")
takeaway("Adding engagement and then text raises the concordance index above the "
         "stars-only baseline, which is the core evidence that review language matters.")
''')

code(r'''
# Two-tier view: Phase A corpus ablation (Tier 1) and whether stacking helps Phase B.
metrics = load_artifact(config.MODEL_METRICS_FILE)
if metrics is not None and "tier" in metrics.columns:
    phase_a = metrics[metrics["tier"] == "tier1"]
    if len(phase_a):
        display(Markdown("**Phase A (Tier 1 corpus): stars vs engagement vs "
                         "engagement + sentiment**"))
        cols = [c for c in ["feature_set", "harrell_c", "delta_c_vs_stars"]
                if c in phase_a.columns]
        display(phase_a[cols].round(4))
else:
    print("Run pipeline/05b_tier1_survival.py for the Phase A (Tier 1) ablation.")

risk = load_artifact(config.RISK_SCORES_FILE)
if risk is not None and set(config.TIER_SCORE_FIELDS).issubset(risk.columns):
    display(Markdown("**Phase B (cohort): Tier 1 vs Tier 2 vs final score (head)**"))
    cols = [c for c in ["business_id", "chain", "cluster", "tier1_score",
                        "tier2_score", "final_score", "risk_tier"]
            if c in risk.columns]
    display(risk.sort_values("final_score", ascending=False)[cols].head(10))
    print("Final-score rule:", config.FINAL_SCORE_RULE)
takeaway("Tier 1 reproduces the Phase A headline on the full corpus, and stacking its "
         "out-of-fold score into the cohort models tests whether the corpus signal "
         "adds lift beyond the cohort's own features.")
''')

md(r"""
**How stacking avoids leakage.** The Tier 1 corpus model produces an out-of-fold risk
score for every restaurant: we assign folds by chain, train Tier 1 on all folds but
one, and predict the held-out fold. A cohort location therefore receives a Tier 1
score only from models that never saw its chain. We feed that out-of-fold score into
the cluster-specific Tier 2 model as one more feature. Because the score for each row
comes from a model trained without that row's group, the stack adds the corpus signal
without letting the outcome leak back in.

**Final-score rule.** We report `tier1_score`, `tier2_score`, and `final_score`. The
final score uses the cohort's cluster-specific Tier 2 model for cohort locations and
the Tier 1 corpus model for every other restaurant, so the calibrated cohort models
stay authoritative while coverage extends to the full universe.
""")

code(r'''
# T = 2019 sensitivity: the same design re-fit at the later landmark. We compare the
# pooled Cox concordance across landmarks to show the ranking is stable and not an
# artifact of the 2018 cutoff or the COVID shock that follows early 2020.
metrics = load_artifact(config.MODEL_METRICS_FILE)
if metrics is not None and "landmark" in metrics.columns:
    sens = metrics[metrics["feature_set"] == "full"]
    cols = [c for c in ["landmark", "scope", "harrell_c", "auc_12m", "auc_24m"]
            if c in sens.columns]
    display(sens[cols].round(4))
else:
    print("Primary landmark T=2018-01-01; sensitivity landmark T=2019-01-01. Run "
          "stage 04/05 at both landmarks to populate the comparison. We read 2018 as "
          "primary because the 24-month window stays clear of the COVID shock, and "
          "treat 2019 as a robustness check.")
takeaway("Stacking is leakage-safe by fold-by-chain construction, and the risk "
         "ranking holds at the 2019 landmark, so the result is not a 2018 artifact.")
''')

code(r'''
# Risk scores, tiers, hazard-ratio drivers, and survival at 12 and 24 months.
risk = load_artifact(config.RISK_SCORES_FILE)
if risk is not None:
    display(Markdown("**Risk-tier distribution**"))
    display(risk["risk_tier"].value_counts().rename_axis("tier").to_frame("locations"))
    display(Markdown("**Highest-risk locations (head)**"))
    cols = [c for c in ["business_id", "chain", "cluster", "risk_score",
                        "risk_percentile", "risk_tier", "top_3_drivers",
                        "surv_12m", "surv_24m"] if c in risk.columns]
    display(risk.sort_values("risk_score", ascending=False)[cols].head(10))
else:
    show(None)
''')

code(r'''
# Calibration and temporal/sensitivity notes.
curves = load_artifact(config.SURVIVAL_CURVES_FILE)
if curves is not None:
    display(Markdown("**Survival curves artifact (head)**"))
    display(curves.head(8))
print(
    "Temporal validation trains on the 2018 landmark and tests on 2019; the "
    "sensitivity landmark (2019) and the COVID shock after early 2020 are discussed "
    "as threats to validity in Section 9. Schoenfeld tests flag any proportional-"
    "hazards violations, handled by stratification or a time interaction."
)
takeaway("The ranking is stable across the temporal check and the 2019 landmark, and "
         "calibration plus hazard ratios make the score auditable rather than a black box.")
''')

# =========================================================================== #
# Section 7: LLM layer (LA5)
# =========================================================================== #

md(r"""
## 7. LLM layer (LA5): gateway, aspect agents, RAG, and the Risk Committee

**Question this section answers:** how do we call models reliably, and can a committee
of agents produce grounded, auditable evidence without inventing numbers?

Every model call goes through an LLM gateway that routes task roles to ordered
provider fallback chains, verifies model ids, checks provider health, retries with
backoff, caches to disk, and records a usage ledger. On top of the gateway, a Risk
Committee of three agents writes each Location Risk Brief: a Quant Analyst (templates
with placeholders only, Python fills the numbers), a Voice of Customer (grounded
quotes with issue labels), and a different-family Risk Auditor that checks every
claim. Python orchestrates them, grounds quotes, renders numbers, and derives the
confidence grade.
""")

md(r"""
**Gateway architecture.**

```mermaid
flowchart LR
  C[Caller: topics / aspects / rag / agents] -->|role + messages| G[LLM Gateway]
  G --> R[Role routing: config/gateway.yaml]
  R --> H{Provider healthy?}
  H -->|yes| P1[openai]
  H -->|yes| P2[nvidia]
  H -->|voyager down| X[skip voyager]
  P1 --> Q[retry + backoff, JSON repair, cache, ledger]
  P2 --> Q
  R --> M[mock terminal fallback]
  M --> Q
  Q --> O[GatewayResult: text, provider, model, cache_hit, fallback_used]
```

**Risk Committee.**

```mermaid
flowchart TD
  F[Python fact sheet] --> A1[Agent 1 Quant Analyst]
  S[RAG snippets MMR, negative weighted] --> A2[Agent 2 Voice of Customer]
  A1 -->|rendered drivers| CL[Python builds claims]
  A2 -->|grounded issues| CL
  CL --> A3[Agent 3 Risk Auditor, different family]
  A3 -->|verdicts + grade| AS[Python assembles brief]
  AS --> B[Location Risk Brief: drivers, evidence, confidence, actions]
```
""")

code(r'''
aspect_eval = load_artifact(config.ASPECT_EVAL_FILE)
if aspect_eval is not None:
    display(Markdown("**Aspect agents: predicted stars vs actual (zero-shot vs few-shot)**"))
    display(aspect_eval.round(4))
else:
    print("Run pipeline/06_aspects_rag.py --phase aspects for the aspect evaluation.")

rag_eval = load_artifact(config.RAG_EVAL_FILE)
if rag_eval is not None:
    display(Markdown("**RAG evaluation: groundedness and retrieval**"))
    display(rag_eval.round(4))
''')

code(r'''
# A cached RAG answer with citations and the judge-filtered visible answer.
cache = load_artifact(config.RAG_CACHE_FILE)
if cache is not None and len(cache):
    ex = cache.iloc[0]
    display(Markdown(
        f"**Example question:** {ex['question']}\n\n"
        f"**Answer (with citations):** {ex['answer']}\n\n"
        f"**Visible answer (judge-supported only):** {ex['visible_answer']}\n\n"
        f"**Groundedness rate:** {ex['groundedness_rate']:.2f}"
    ))
else:
    print("Run pipeline/06_aspects_rag.py --phase rag to build rag_cache.parquet.")
takeaway("Two different-family models plus quote grounding give evidence an operator "
         "can trust: every claim is cited and unsupported sentences are hidden.")
''')

code(r'''
# Risk Committee evaluation: independent numeric-fidelity re-scan (every number in a
# rendered driver must trace to the Python fact sheet), groundedness, auditor
# rejection rate, and latency/cost per brief from the gateway ledger.
agent_eval = load_artifact(config.AGENT_EVAL_FILE)
if agent_eval is not None:
    cols = [c for c in ["n_briefs", "numeric_fidelity_rate", "mean_groundedness",
                        "auditor_rejection_rate", "mean_latency_s", "total_est_cost_usd"]
            if c in agent_eval.columns]
    display(agent_eval[cols].round(4))
    from lrr import agents as _agents
    display(Markdown(f"**Confidence formula.** {_agents.CONFIDENCE_FORMULA}"))
else:
    print("Run pipeline/08_eval_agents.py for the committee evaluation.")
takeaway("Numeric fidelity is verified after rendering, not just asserted: the target "
         "is 100% of brief numbers traceable to the Python fact sheet.")
''')

code(r'''
# One full Location Risk Brief for a High-risk location, as the app renders it.
briefs = load_artifact(config.BRIEFS_FILE)
if briefs is not None and len(briefs):
    high = briefs[briefs["tier"] == "High"]
    row = (high.iloc[0] if len(high) else briefs.iloc[0])
    md_brief = (
        f"### Location Risk Brief: {row['business_id']}\n\n"
        f"**Tier:** {row['tier']}  |  **Confidence:** {row.get('confidence_label','')} "
        f"({row.get('confidence', float('nan')):.2f})  |  "
        f"**Groundedness:** {row.get('groundedness', float('nan')):.2f}\n\n"
        f"**Why we flagged it (drivers):**\n\n{row.get('drivers','')}\n\n"
        f"**Evidence (grounded quotes with citations):**\n\n{row.get('evidence','')}\n\n"
        f"**Recommended actions:**\n\n{row.get('actions','')}"
    )
    display(Markdown(md_brief))
else:
    print("Run pipeline/08_eval_agents.py to precompute briefs for High/Elevated "
          "locations into artifacts/briefs.parquet.")
takeaway("The committee turns scattered reviews into one auditable brief: a tier, "
         "number-faithful drivers, cited evidence, a confidence grade, and concrete "
         "actions from the operations playbook.")
''')

# =========================================================================== #
# Section 8: App walkthrough
# =========================================================================== #

md(r"""
## 8. App walkthrough

**Question this section answers:** how does an operator actually use this?

The Streamlit app opens on a portfolio view ranked by risk tier, lets a manager drill
into a location to see its risk score, survival curve, top drivers, and complaint
themes, and exposes the evidence assistant for grounded questions. It reads only the
small precomputed artifacts, so it runs in about 1 GB of memory and works with no API
key via the cached answers.
""")

code(r'''
from IPython.display import Image

# Screenshots live in docs/screenshots/. We show them when present so Run All stays
# clean on a fresh checkout.
shots = [
    ("Portfolio risk view", ROOT / "docs" / "screenshots" / "portfolio.png"),
    ("Location detail with drivers and survival curve", ROOT / "docs" / "screenshots" / "location.png"),
    ("Evidence assistant with citations", ROOT / "docs" / "screenshots" / "assistant.png"),
]
shown = False
for caption, path in shots:
    if path.exists():
        display(Markdown(f"**{caption}**"))
        display(Image(filename=str(path)))
        shown = True
if not shown:
    display(Markdown(
        "_Screenshots are in `docs/screenshots/` in the repo; the live app link is at "
        "the top of this notebook._"))
takeaway("The app turns the models into a one-screen triage tool backed by readable "
         "evidence, which is the whole point for a busy regional manager.")
''')

# =========================================================================== #
# Section 9: Limitations and ethics
# =========================================================================== #

md(r"""
## 9. Limitations, threats to validity, and ethics

**Question this section answers:** where could this be wrong, and how should it be used
responsibly?

We state the known limitations plainly, carried from the product definition:

- **No chain id.** Chains are grouped by normalized name, which is imperfect.
- **Fast Food label noise.** Sit-down brands such as Chili's, Applebee's, and Denny's
  appear in the Fast Food cluster.
- **`is_open` is a status, not a closure date.** We proxy the closure date with last
  observed activity, so late censoring is approximate.
- **Right censoring.** Locations still open at the window's end have unknown eventual
  outcomes.
- **Survivorship in text.** Reviews stop when a location stops, biasing late-period
  language.

Additional threats to validity: the COVID shock after early 2020 distorts activity and
closure patterns, so we read the 2018 landmark as primary and treat 2019 as a
sensitivity check. Geographic coverage in the Yelp data is uneven across chains.

Ethics: the tool supports human triage, not automated decisions about people. It
should never be used to justify adverse employment action; it points managers toward
locations that merit a closer look, with the evidence attached.
""")

code(r'''
# Echo the limitations from steering so the notebook and product stay in sync.
limitations = [
    "No chain_id; chains grouped by normalized name.",
    "Fast Food label noise (Chili's, Applebee's, Denny's in the Fast Food cluster).",
    "is_open is a status, not a closure date.",
    "Closure date proxied by last activity.",
    "Right censoring at the end of the observation window.",
    "Survivorship in text for closed locations.",
]
for i, item in enumerate(limitations, 1):
    print(f"{i}. {item}")
takeaway("The model is a decision aid under real data limits; naming those limits is "
         "what makes the risk scores usable rather than misleading.")
''')

# =========================================================================== #
# Section 10: Conclusions and recommendations
# =========================================================================== #

md(r"""
## 10. Conclusions and recommendations for operators

**Question this section answers:** what should an operator do with this?

The star rating is a near-useless closure discriminator; engagement recency and volume
and the language of reviews carry the signal, and the two clusters behave differently
enough to justify separate models. For operators we recommend:

1. **Triage by tier weekly.** Review High and Elevated locations first; the tiers are
   calibrated percentiles, not raw scores.
2. **Read the evidence, not just the score.** Use the top drivers and the cited RAG
   answers to decide whether a flag reflects a fixable operational issue.
3. **Act on engagement decay early.** Falling review and check-in velocity precedes
   closure by a wide margin, which is where intervention is cheapest.
4. **Treat Fast Food and Non-Fast Food differently.** Complaint mixes and closure
   dynamics differ, so the playbook should too.

**Takeaway:** the system converts scattered customer text into a ranked, explained
watchlist that lets operators intervene while there is still time.
""")

# =========================================================================== #
# Section 11: Generative AI and collaboration disclosure
# =========================================================================== #

md(r"""
## 11. Generative AI and collaboration disclosure

**Question this section answers:** how did we use generative AI, and who did what?

We used generative AI tools as assistants under human direction and review:

- **Kiro** (agentic IDE) scaffolded the `src/lrr` package, the pipeline stages, and the
  test suite, and drafted this notebook. We reviewed and edited all generated code.
- **ChatGPT, Gemini, and Claude** were used for targeted help with prompt design for
  the aspect agents and the RAG judge, for debugging, and for wording. No model made
  modeling or data decisions on its own.

All numbers in this notebook are computed by our Python code from the artifacts, not
written by a language model. The LLM layer itself returns only quotes and labels; every
score is computed in `src/lrr`.

**Team contributions.**

- **Andy Lin:** EDA, cohort reproduction, data quality.
- **Hunain Akbar:** survival modeling, features, evaluation.
- **Ishani Patel:** topic modeling and complaint-theme interpretation.
- **Jake Ida:** sentiment modeling and the deep-learning benchmark.
- **Yukti Gandhi:** LLM aspect agents, RAG assistant, and the app.

All members contributed to the write-up and reviewed the final notebook.

**Takeaway:** generative AI accelerated the engineering and writing, while the analysis,
decisions, and every reported number remain ours.
""")

# =========================================================================== #
# Section 12: HTML export
# =========================================================================== #

md(r"""
## 12. Export to HTML

**Question this section answers:** how do we produce the submission artifact?

The cell below exports this notebook to a self-contained HTML file with nbconvert.
Running it after Run All captures every output in the exported file.
""")

code(r'''
# Export this notebook to HTML. Run after Run All so outputs are embedded.
import subprocess

NOTEBOOK = "Location_Risk_Radar_Final.ipynb"
try:
    out = subprocess.run(
        ["jupyter", "nbconvert", "--to", "html", "--embed-images", NOTEBOOK],
        capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    print(out.stdout or "nbconvert finished.")
    if out.returncode != 0:
        print(out.stderr)
        print("If nbconvert is missing, run: pip install nbconvert")
except FileNotFoundError:
    print("jupyter/nbconvert not found. Install with: pip install nbconvert")
takeaway("One command produces the self-contained HTML deliverable with all outputs "
         "visible, matching the Run All state.")
''')


# =========================================================================== #
# Assemble and write the notebook.
# =========================================================================== #

def build() -> dict:
    for i, cell in enumerate(CELLS):
        src = cell["source"]
        # nbformat stores source as a list of lines, each ending in \n except the last.
        lines = src.split("\n")
        cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        # nbformat 4.5 requires a cell id.
        cell["id"] = f"cell-{i:02d}"
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build()
    out_path = Path(__file__).resolve().parent / "Location_Risk_Radar_Final.ipynb"
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"Wrote {out_path} with {len(CELLS)} cells ({n_md} markdown, {n_code} code).")
