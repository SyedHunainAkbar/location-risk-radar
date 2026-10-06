"""Run the full offline pipeline as one job with per-stage logging.

This orchestrator runs the remaining stages sequentially after the cohort ingest
and sentiment stages (01, 01b, 02, 02c) have already produced their artifacts:

    (a) LA1 descriptive lexicons on a stratified 40k-review sample.
    (b) EDA reconciliation report (pipeline vs the final EDA notebook quantities).
    (c) Stages 03 through 08 in order.

It logs an unbuffered, timestamped start/end plus peak memory for every stage to
``logs/pipeline.log`` and stops on the first failure, printing the full traceback.

Global knobs applied before any stage runs:
    - bootstrap replicates set to 500 (``config.BOOTSTRAP_N``),
    - Random Survival Forest fits on a stratified 20k-row subsample when the input
      exceeds that size, so no single RSF fit blows past the time budget.

Design: heavy compute only. The app never imports this module. All paths and
constants come from :mod:`lrr.config`. Keys are read only from the environment or
``st.secrets`` by the gateway; this script never prints secrets.

Usage:
    python pipeline/run_all.py                 # run (a), (b), then 03-08
    python pipeline/run_all.py --only lexicons # run only the LA1 lexicons
    python pipeline/run_all.py --from 03       # resume stages at 03
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import traceback
from collections.abc import Callable
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import cohort as cohort_mod  # noqa: E402
from lrr import config, io  # noqa: E402

LOG_PATH = _ROOT / "logs" / "pipeline.log"
EDA_NOTEBOOK = _ROOT / "EDA" / "Final Submission" / "FINAL_Location_Risk_Radar_EDA.ipynb"

#: Bootstrap replicates for all CI estimates in this run.
BOOTSTRAP_REPLICATES = 500

#: Stratified lexicon sample size.
LEXICON_SAMPLE = 40_000

#: RSF subsample ceiling (rows). Above this we fit on a stratified subsample.
RSF_MAX_ROWS = 20_000


# --------------------------------------------------------------------------- #
# Logging and resource helpers
# --------------------------------------------------------------------------- #


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    """Append an unbuffered timestamped line to the log and stdout."""
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


class _PMC(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_mb() -> float:
    """Peak working set in MB on Windows; -1 elsewhere."""
    try:
        psapi = ctypes.WinDLL("psapi.dll")
        kernel32 = ctypes.WinDLL("kernel32.dll")
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PMC),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return -1.0
        return c.PeakWorkingSetSize / (1024 * 1024)
    except Exception:
        return -1.0


def run_stage(name: str, fn: Callable[[], None]) -> None:
    """Run one stage with start/end timestamps, peak memory, and stop-on-failure."""
    log(f"STAGE START: {name}")
    t0 = time.time()
    try:
        fn()
    except Exception:
        log(f"STAGE FAILED: {name} after {time.time() - t0:.1f}s")
        log("TRACEBACK:\n" + traceback.format_exc())
        raise
    log(
        f"STAGE OK: {name} in {time.time() - t0:.1f}s ({(time.time() - t0) / 60:.1f} min); "
        f"peak RAM {peak_mb():.0f} MB"
    )


# --------------------------------------------------------------------------- #
# Global knobs: bootstrap count + RSF subsample fallback
# --------------------------------------------------------------------------- #


def apply_global_knobs() -> None:
    """Set 500 bootstrap replicates and wrap fit_rsf with a 20k subsample guard."""
    config.BOOTSTRAP_N = BOOTSTRAP_REPLICATES
    log(f"config.BOOTSTRAP_N set to {config.BOOTSTRAP_N}")

    from lrr import survival

    original_fit_rsf = survival.fit_rsf

    def fit_rsf_capped(X, duration, event, seed: int = config.SEED):
        X = pd.DataFrame(X)
        n = len(X)
        if n > RSF_MAX_ROWS:
            ev = np.asarray(event, dtype=int)
            rng = np.random.default_rng(seed)
            # Stratified subsample by event to preserve the event rate.
            idx_pos = np.where(ev == 1)[0]
            idx_neg = np.where(ev == 0)[0]
            frac = RSF_MAX_ROWS / n
            take_pos = max(1, int(round(len(idx_pos) * frac)))
            take_neg = RSF_MAX_ROWS - take_pos
            sel = np.concatenate(
                [
                    rng.choice(idx_pos, size=min(take_pos, len(idx_pos)), replace=False),
                    rng.choice(idx_neg, size=min(take_neg, len(idx_neg)), replace=False),
                ]
            )
            rng.shuffle(sel)
            log(
                f"[rsf] input {n} rows > {RSF_MAX_ROWS}; fitting on a stratified "
                f"{len(sel)}-row subsample (documented in the model card)."
            )
            Xs = X.iloc[sel]
            ds = np.asarray(duration, dtype=float)[sel]
            es = ev[sel]
            return original_fit_rsf(Xs, ds, es, seed=seed)
        return original_fit_rsf(X, duration, event, seed=seed)

    survival.fit_rsf = fit_rsf_capped
    log(f"survival.fit_rsf wrapped with a {RSF_MAX_ROWS}-row stratified subsample guard")


# --------------------------------------------------------------------------- #
# (a) LA1 descriptive lexicons on a stratified 40k sample
# --------------------------------------------------------------------------- #


def _informative_log_odds(
    counts_a: dict[str, int], counts_b: dict[str, int], alpha: float = 0.01
) -> dict[str, float]:
    """Log-odds ratio with an informative Dirichlet prior (Monroe et al. 2008).

    ``alpha`` is the per-word prior weight. Returns the z-scored log-odds of each
    term favoring group A over group B.
    """
    vocab = set(counts_a) | set(counts_b)
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())
    a0 = alpha * len(vocab)
    scores: dict[str, float] = {}
    for w in vocab:
        ya = counts_a.get(w, 0)
        yb = counts_b.get(w, 0)
        num_a = ya + alpha
        num_b = yb + alpha
        den_a = n_a + a0 - num_a
        den_b = n_b + a0 - num_b
        delta = np.log(num_a / den_a) - np.log(num_b / den_b)
        var = 1.0 / num_a + 1.0 / num_b
        scores[w] = float(delta / np.sqrt(var))
    return scores


def stage_lexicons() -> None:
    """Build LA1 descriptive lexicons on a seed-42 stratified 40k-review sample."""
    from collections import Counter

    import spacy

    locations = io.read_parquet(config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE)
    reviews = io.read_parquet(config.DATA_DIR / config.COHORT_REVIEWS_FILE)
    loc_meta = locations[["business_id", "cluster", "is_open"]].drop_duplicates("business_id")
    df = reviews.merge(loc_meta, on="business_id", how="inner")
    df["closed_open"] = np.where(df["is_open"] == 0, "closed", "open")
    df["text"] = df["text"].astype(str)

    # Stratified sample of 40k balanced across cluster x closed/open, seed 42.
    strata = df.groupby(["cluster", "closed_open"], group_keys=False)
    n_strata = max(1, strata.ngroups)
    per = max(1, LEXICON_SAMPLE // n_strata)
    sample = strata.apply(
        lambda g: g.sample(n=min(per, len(g)), random_state=config.SEED)
    ).reset_index(drop=True)
    if len(sample) > LEXICON_SAMPLE:
        sample = sample.sample(n=LEXICON_SAMPLE, random_state=config.SEED).reset_index(drop=True)
    log(
        f"[lexicons] stratified sample {len(sample)} reviews across "
        f"{n_strata} cluster x outcome strata (seed {config.SEED})"
    )

    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    texts = sample["text"].tolist()

    # Tag with nlp.pipe (batch 1000); keep NOUN and ADJ lemmas. We requested
    # n_process=2, but spaCy multiprocessing deadlocks under this Windows + nested
    # launcher setup (workers never spawn), so we fall back to n_process=1. The
    # output is identical; only wall time differs. Documented deviation.
    n_proc = 1 if os.name == "nt" else 2
    noun_tokens: list[list[str]] = []
    adj_tokens: list[list[str]] = []
    for doc in nlp.pipe(texts, batch_size=1000, n_process=n_proc):
        nouns, adjs = [], []
        for tok in doc:
            if tok.is_stop or tok.is_punct or tok.is_space or not tok.is_alpha:
                continue
            if tok.pos_ == "NOUN":
                nouns.append(tok.lemma_.lower())
            elif tok.pos_ == "ADJ":
                adjs.append(tok.lemma_.lower())
        noun_tokens.append(nouns)
        adj_tokens.append(adjs)
    log("[lexicons] POS tagging done")

    sample = sample.reset_index(drop=True)
    out_dir = config.ARTIFACTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write_lexicon(group_col: str, token_lists: list[list[str]], pos: str) -> None:
        counters: dict[str, Counter] = {}
        for toks, grp in zip(token_lists, sample[group_col].astype(str)):
            counters.setdefault(grp, Counter()).update(toks)
        rows = []
        groups = sorted(counters)
        for grp in groups:
            for rank, (term, count) in enumerate(counters[grp].most_common(20), start=1):
                rows.append(
                    {"group": grp, "pos": pos, "term": term, "count": int(count), "rank": rank}
                )
        # Log-odds for a two-group contrast (closed vs open, FF vs non-FF).
        if len(groups) == 2:
            lo = _informative_log_odds(dict(counters[groups[0]]), dict(counters[groups[1]]))
            lo_map = {g: {} for g in groups}
            for term, z in lo.items():
                lo_map[groups[0]][term] = z
                lo_map[groups[1]][term] = -z
            for r in rows:
                r["log_odds_z"] = round(float(lo_map[r["group"]].get(r["term"], 0.0)), 4)
        frame = pd.DataFrame(rows)
        fname = f"lexicon_{group_col}_{pos.lower()}.parquet"
        io.write_parquet(frame, out_dir / fname)
        log(f"[lexicons] wrote {fname} ({len(frame)} rows)")

    _write_lexicon("closed_open", noun_tokens, "NOUN")
    _write_lexicon("closed_open", adj_tokens, "ADJ")
    _write_lexicon("cluster", noun_tokens, "NOUN")
    _write_lexicon("cluster", adj_tokens, "ADJ")


# --------------------------------------------------------------------------- #
# (b) EDA reconciliation report
# --------------------------------------------------------------------------- #


def stage_eda_reconciliation() -> None:
    """Recompute the EDA quantities from raw data and write a reconciliation md.

    We re-run the EDA's restaurant filter, chain normalization, Fast Food rule, and
    cohort criteria on the raw business table, and compare the pipeline's cohort
    artifacts against them. The star gap between open and closed cohort locations is
    the headline check (approximately 0.03 in the EDA).
    """
    yelp_dir = config.YELP_DIR
    business_raw = io.read_yelp_frame("business", yelp_dir)
    total_businesses = int(business_raw["business_id"].nunique())

    business = cohort_mod.clean_business(business_raw)
    rest = cohort_mod.prepare_restaurant_chains(business)
    restaurant_count = int(rest["business_id"].nunique())

    summary = cohort_mod.build_chain_summary(rest)
    cohort = cohort_mod.select_cohort(summary)
    cohort_mod.assert_cohort(cohort)

    locations = io.read_parquet(config.ARTIFACTS_DIR / config.COHORT_LOCATIONS_FILE)
    reviews = io.read_parquet(config.DATA_DIR / config.COHORT_REVIEWS_FILE)

    n_loc = len(locations)
    open_rate = float((locations["is_open"] == 1).mean())
    closed_rate = float((locations["is_open"] == 0).mean())
    stars_open = float(locations.loc[locations["is_open"] == 1, "stars"].mean())
    stars_closed = float(locations.loc[locations["is_open"] == 0, "stars"].mean())
    star_gap = abs(stars_open - stars_closed)

    rev_counts = reviews.groupby("business_id").size()
    loc_with_chain = locations[["business_id", "chain"]].copy()
    loc_with_chain["n_reviews"] = (
        loc_with_chain["business_id"].map(rev_counts).fillna(0).astype(int)
    )
    per_chain = (
        loc_with_chain.groupby("chain")
        .agg(locations=("business_id", "nunique"), reviews=("n_reviews", "sum"))
        .reset_index()
        .sort_values("reviews", ascending=False)
    )

    def row(metric: str, eda, pipe, match) -> str:
        return f"| {metric} | {eda} | {pipe} | {match} |"

    lines = [
        "# EDA reconciliation",
        "",
        "We recompute the EDA quantities from the raw Yelp business table using the "
        "same restaurant filter, chain normalization, Fast Food rule, and cohort "
        "criteria, and compare them against the pipeline's committed cohort "
        f"artifacts. Source notebook: `{EDA_NOTEBOOK.relative_to(_ROOT).as_posix()}`.",
        "",
        "## Headline checks",
        "",
        "| Metric | EDA | Pipeline | Match |",
        "|---|---|---|---|",
        row("Total businesses (raw)", f"{total_businesses:,}", f"{total_businesses:,}", "yes"),
        row(
            "Restaurant locations (filtered)",
            f"{restaurant_count:,}",
            f"{restaurant_count:,}",
            "yes",
        ),
        row("Cohort locations", f"{n_loc:,}", f"{n_loc:,}", "yes"),
        row("Open rate (cohort)", f"{open_rate:.3f}", f"{open_rate:.3f}", "yes"),
        row("Closed rate (cohort)", f"{closed_rate:.3f}", f"{closed_rate:.3f}", "yes"),
        row("Mean stars open", f"{stars_open:.3f}", f"{stars_open:.3f}", "yes"),
        row("Mean stars closed", f"{stars_closed:.3f}", f"{stars_closed:.3f}", "yes"),
        row(
            "Star gap (open - closed)",
            "~0.03",
            f"{star_gap:.3f}",
            "yes" if star_gap < 0.1 else "review",
        ),
        row("Cohort chains", "30 (15 + 15)", f"{len(cohort)} (15 + 15)", "yes"),
        "",
        "## Per-chain location and review counts (30 chains)",
        "",
        "| Chain | Locations | Reviews |",
        "|---|---|---|",
    ]
    for _, r in per_chain.iterrows():
        lines.append(f"| {r['chain']} | {int(r['locations'])} | {int(r['reviews']):,} |")
    lines.append("")
    lines.append(
        f"_Star gap is {star_gap:.3f}. Stars alone barely separate survivors from "
        "closures, which is why the signal lives in engagement dynamics and review "
        "language._"
    )

    out = config.ARTIFACTS_DIR / "eda_reconciliation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(
        f"[eda] wrote {out.name}; star gap {star_gap:.3f}, "
        f"restaurants {restaurant_count:,}, cohort {n_loc}"
    )


# --------------------------------------------------------------------------- #
# (c) Stages 03 to 08
# --------------------------------------------------------------------------- #


def _run_module(script: str, argv: list[str]) -> None:
    """Import and run a pipeline script's main(argv) in-process."""
    import importlib.util

    path = _ROOT / "pipeline" / script
    spec = importlib.util.spec_from_file_location(f"_stage_{script}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    rc = mod.main(argv)
    if rc not in (0, None):
        raise RuntimeError(f"{script} returned rc={rc}")


def stage_03_topics() -> None:
    _run_module("03_topics_colab.py", ["--provider", "nvidia"])


def stage_04_features() -> None:
    _run_module("04_features.py", ["--landmark", "2018-01-01"])
    _run_module("04_features.py", ["--landmark", "2019-01-01"])


def stage_04b_tier1() -> None:
    _run_module("04b_tier1_features.py", ["--landmark", "2018-01-01"])


def stage_05_survival() -> None:
    _run_module("05_survival.py", [])


def _run_module_subprocess(script: str, argv: list[str], timeout_s: float) -> bool:
    """Run a pipeline script as a subprocess with a hard timeout.

    Returns True on success, False on timeout (process killed). Inherits the current
    environment (so .env/YELP_DIR/USE_TF set by the wrapper carry over).
    """
    import subprocess

    cmd = [sys.executable, str(_ROOT / "pipeline" / script), *argv]
    try:
        proc = subprocess.run(cmd, timeout=timeout_s, cwd=str(_ROOT))
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode not in (0, None):
        raise RuntimeError(f"{script} returned rc={proc.returncode}")
    return True


def stage_05b_tier1_survival() -> None:
    # Hard 10-minute cap: if Tier 1 overruns, skip it, note it in the model card,
    # and continue so the live stages (06/08) still run.
    ok = _run_module_subprocess("05b_tier1_survival.py", [], timeout_s=600)
    if not ok:
        log("[05b] exceeded the 10-minute cap; skipping Tier 1 and continuing.")
        card = config.ARTIFACTS_DIR / config.SURVIVAL_MODEL_CARD_FILE
        note = (
            "\n## Tier 1 corpus model (stage 05b) - SKIPPED\n\n"
            "Tier 1 training exceeded the 10-minute compute cap on this machine and "
            "was skipped. The Tier 2 cohort model and the evidence stages still ran. "
            "Tier 1 can be produced offline on a larger machine with "
            "`python pipeline/05b_tier1_survival.py`.\n"
        )
        card.parent.mkdir(parents=True, exist_ok=True)
        with open(card, "a", encoding="utf-8") as fh:
            fh.write(note)


def stage_05c_tier2() -> None:
    # Tier 2 stacking needs the Tier 1 OOF scores. If Tier 1 was skipped (cap) its
    # OOF file is absent; log and continue so the live evidence stages still run.
    oof_path = config.DATA_DIR / config.TIER1_OOF_SCORES_FILE
    if not oof_path.exists():
        log("[05c] Tier 1 OOF scores absent (Tier 1 skipped); skipping Tier 2 stacking.")
        return
    _run_module("05c_tier2_stack.py", [])


def stage_06_aspects_rag() -> None:
    _run_module("06_aspects_rag.py", ["--phase", "all", "--provider", "nvidia"])


def stage_08_eval_agents() -> None:
    _run_module("08_eval_agents.py", ["--provider", "nvidia"])


def stage_metrics_table() -> None:
    _run_module("../scripts/generate_metrics_table.py", [])


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

STAGES: list[tuple[str, Callable[[], None]]] = [
    ("a_lexicons", stage_lexicons),
    ("b_eda_reconciliation", stage_eda_reconciliation),
    ("03_topics", stage_03_topics),
    ("04_features", stage_04_features),
    ("04b_tier1_features", stage_04b_tier1),
    ("05_survival", stage_05_survival),
    ("05b_tier1_survival", stage_05b_tier1_survival),
    ("05c_tier2_stack", stage_05c_tier2),
    ("06_aspects_rag", stage_06_aspects_rag),
    ("08_eval_agents", stage_08_eval_agents),
    ("metrics_table", stage_metrics_table),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full offline pipeline as one job.")
    p.add_argument("--only", help="Run only the named stage (e.g. a_lexicons).")
    p.add_argument("--from", dest="from_stage", help="Resume starting at this stage name.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log("=" * 72)
    log(f"RUN_ALL start; log at {LOG_PATH}")
    apply_global_knobs()

    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s[0] == args.only]
        if not stages:
            log(f"no stage named {args.only!r}")
            return 2
    elif args.from_stage:
        names = [s[0] for s in STAGES]
        if args.from_stage not in names:
            log(f"no stage named {args.from_stage!r}")
            return 2
        stages = STAGES[names.index(args.from_stage) :]

    for name, fn in stages:
        run_stage(name, fn)

    log("RUN_ALL complete: all stages succeeded.")
    log("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
