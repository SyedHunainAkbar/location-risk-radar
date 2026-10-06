"""Survival models, risk scoring, and explainability (stage 05).

Defines the four nested feature sets, the design-matrix builder (with a pooled
cluster indicator), Cox PH / Random Survival Forest / XGBoost fitters, and the
risk-score, tiering, driver, and survival-curve helpers. Heavy libraries
(lifelines, scikit-survival, xgboost, shap) are imported lazily so the pure logic
(feature sets, tiering, drivers, design matrix) is unit-testable without them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from lrr import config

# --------------------------------------------------------------------------- #
# Feature sets (nested, for the ablation)
# --------------------------------------------------------------------------- #

_STARS = ["stars_all", "stars_12m", "stars_trend"]
_ENGAGEMENT = [
    "review_vol_12m",
    "review_vol_6m",
    "review_vol_3m",
    "review_velocity_slope",
    "review_velocity_ratio",
    "checkin_vol",
    "checkin_slope",
    "tip_vol",
    "tip_coverage",
    "review_length",
    "location_age_days",
    "chain_prior",
]
_TEXT = [
    "sentiment_mean",
    "sentiment_slope",
    "share_negative",
    "lexicon_complaint_rate",
]


def _topic_cols(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith("topic_")]


def _theme_cols(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c.startswith("theme_")]


#: Nested feature sets. ``full`` = engagement + text + topic shares + stars.
FEATURE_SETS: dict[str, list[str]] = {
    "stars_only": list(_STARS),
    "engagement": list(_ENGAGEMENT),
    "engagement_text": list(_ENGAGEMENT) + list(_TEXT),
    "full": list(_STARS) + list(_ENGAGEMENT) + list(_TEXT),
}


def feature_columns(features: pd.DataFrame, feature_set: str) -> list[str]:
    """Resolve a feature-set name to the columns present in ``features``.

    Topic-share columns are appended to the ``full`` set when present.
    """
    cols = list(FEATURE_SETS[feature_set])
    if feature_set == "full":
        # Prefer collapsed complaint-theme shares when present; otherwise fall back
        # to the raw per-topic shares.
        theme = _theme_cols(features)
        cols = cols + (theme if theme else _topic_cols(features))
    return [c for c in cols if c in features.columns]


#: Threshold above which a pairwise absolute correlation is treated as redundant.
COLLINEAR_THRESHOLD: float = 0.95

#: Records the most recent design-matrix cleaning decisions (for the model card /
#: feature_decisions.md). Each call to :func:`design_matrix` with ``record=True``
#: refreshes this. A list of (feature, action, reason) tuples.
DESIGN_DECISIONS: list[tuple[str, str, str]] = []


def _median_impute_with_indicator(
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Median-impute each column; add a 0/1 missingness indicator where any gap exists.

    Returns the imputed frame and the list of (feature, action, reason) decisions.
    """
    decisions: list[tuple[str, str, str]] = []
    out = X.copy()
    medians = out.median(numeric_only=True)
    for col in list(out.columns):
        n_missing = int(out[col].isna().sum())
        if n_missing:
            out[f"{col}_missing"] = out[col].isna().astype(int)
            out[col] = out[col].fillna(medians.get(col, 0.0))
            decisions.append(
                (
                    col,
                    "impute+indicator",
                    f"{n_missing} missing filled with train median, added {col}_missing",
                )
            )
    out = out.fillna(0.0)
    return out, decisions


def _drop_zero_variance(X: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Drop columns with zero variance (constant), which make the Hessian singular."""
    decisions: list[tuple[str, str, str]] = []
    std = X.std(ddof=0)
    drop = [c for c in X.columns if float(std.get(c, 0.0)) == 0.0]
    for c in drop:
        decisions.append(
            (c, "drop", "zero variance (constant column); singular in the Cox Hessian")
        )
    return X.drop(columns=drop), decisions


def _drop_collinear(
    X: pd.DataFrame, keep_priority: Sequence[str]
) -> tuple[pd.DataFrame, list[tuple[str, str, str]]]:
    """Drop one member of each pairwise |r| > threshold, keeping the more interpretable.

    ``keep_priority`` lists columns in order of interpretability; when two columns
    are highly correlated we drop the one that appears later in the priority order.
    """
    decisions: list[tuple[str, str, str]] = []
    cols = list(X.columns)
    if len(cols) < 2:
        return X, decisions
    corr = X.corr().abs()
    rank = {c: i for i, c in enumerate(keep_priority)}

    def interp_rank(c: str) -> int:
        return rank.get(c, len(keep_priority) + cols.index(c))

    dropped: set[str] = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            if a in dropped or b in dropped:
                continue
            r = corr.iloc[i, j]
            if pd.notna(r) and r > COLLINEAR_THRESHOLD:
                # Keep the more interpretable (lower rank); drop the other.
                drop_c = b if interp_rank(a) <= interp_rank(b) else a
                keep_c = a if drop_c == b else b
                dropped.add(drop_c)
                reason = (
                    f"|r|={float(r):.3f} with {keep_c} (> {COLLINEAR_THRESHOLD}); "
                    f"kept more interpretable {keep_c}"
                )
                decisions.append((drop_c, "drop", reason))
    return X.drop(columns=list(dropped)), decisions


#: Descriptive phrase for each VOC complaint theme, used to embed and match topics.
THEME_PHRASES: dict[str, str] = {
    "service_speed": "slow service, long wait times, slow to be served",
    "staff_attitude": "rude, unfriendly, or inattentive staff and servers",
    "order_accuracy": "wrong, missing, or incorrect order",
    "food_quality": "cold, bland, or poor quality food",
    "cleanliness": "dirty, unclean, poor hygiene, messy tables and restrooms",
    "value": "price, value for money, too expensive",
    "management_response": "management and how complaints are handled",
    "drive_thru_takeout": "drive-thru, takeout, delivery, and pickup",
    "other": "general comments not about a specific complaint theme",
}


def collapse_topics_to_themes(
    features: pd.DataFrame,
    topic_labels: pd.DataFrame,
    encode_fn=None,
    map_out_path=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse raw ``topic_*`` share columns into ~9 complaint-theme shares.

    Each BERTopic topic is matched to its nearest complaint theme by cosine
    similarity between the MiniLM embedding of the topic's ``top_words`` and the
    embedding of each theme phrase (:data:`THEME_PHRASES`). Per-business topic
    shares are then summed within each assigned theme to produce ``theme_<name>``
    columns, replacing the raw ``topic_*`` columns.

    Returns ``(features_with_theme_cols, mapping_df)`` where ``mapping_df`` has
    columns ``topic, top_words, theme, similarity``. ``encode_fn`` can be injected
    in tests; by default we use the shared RAG MiniLM encoder.
    """
    topic_cols = _topic_cols(features)
    if not topic_cols:
        return features, pd.DataFrame(columns=["topic", "top_words", "theme", "similarity"])

    themes = list(THEME_PHRASES.keys())
    theme_texts = [THEME_PHRASES[t] for t in themes]

    # Topic -> representative text (top_words), from the per-cluster label tables.
    labels = topic_labels.drop_duplicates(subset="topic").set_index("topic")
    topic_ids = [int(c.split("_")[1]) for c in topic_cols]
    topic_texts = [str(labels["top_words"].get(tid, "")) for tid in topic_ids]

    if encode_fn is None:
        from lrr import rag

        encode_fn = rag._encode

    emb_topics = np.asarray(encode_fn(topic_texts), dtype=float)
    emb_themes = np.asarray(encode_fn(theme_texts), dtype=float)

    def _norm(m):
        return m / np.clip(np.linalg.norm(m, axis=-1, keepdims=True), 1e-9, None)

    sim = _norm(emb_topics) @ _norm(emb_themes).T  # (n_topics, n_themes)
    best = sim.argmax(axis=1)

    rows = []
    theme_of_col: dict[str, str] = {}
    for col, tid, bi, texts in zip(topic_cols, topic_ids, best, topic_texts):
        theme = themes[int(bi)]
        theme_of_col[col] = theme
        rows.append(
            {
                "topic": tid,
                "top_words": texts,
                "theme": theme,
                "similarity": round(float(sim[topic_ids.index(tid), int(bi)]), 4),
            }
        )
    mapping = pd.DataFrame(rows).sort_values("topic").reset_index(drop=True)

    out = features.copy()
    for theme in themes:
        cols = [c for c in topic_cols if theme_of_col[c] == theme]
        out[f"theme_{theme}"] = out[cols].sum(axis=1) if cols else 0.0
    out = out.drop(columns=topic_cols)

    if map_out_path is not None:
        from pathlib import Path

        Path(map_out_path).parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(map_out_path, index=False)

    return out, mapping


#: Interpretability priority for collinear-pair resolution (earlier = keep).
_INTERP_PRIORITY: list[str] = (
    _STARS
    + ["review_vol_12m", "review_vol_6m", "review_vol_3m"]
    + ["review_velocity_slope", "review_velocity_ratio"]
    + [
        "checkin_vol",
        "checkin_slope",
        "tip_vol",
        "tip_coverage",
        "review_length",
        "location_age_days",
        "chain_prior",
    ]
    + _TEXT
    + ["cluster_ff"]
)


def design_matrix(
    features: pd.DataFrame,
    feature_set: str,
    pooled: bool = False,
    *,
    clean: bool = False,
    keep_cols: Sequence[str] | None = None,
    standardize: bool = False,
    record: bool = False,
) -> pd.DataFrame:
    """Build the Cox design matrix.

    By default (``clean=False``) this is the simple dense matrix: selected columns,
    numeric-coerced, median-imputed. Set ``clean=True`` for the robust production
    path that median-imputes with a missingness indicator, drops zero-variance
    columns, drops one of each highly collinear pair (keeping the more
    interpretable), and optionally standardizes continuous columns, so the
    Newton-Raphson Cox fit stays well conditioned.

    For cross-validation, fit the column schema on the training fold and pass the
    resulting column list via ``keep_cols`` so test folds share the exact columns.
    Set ``record=True`` to refresh :data:`DESIGN_DECISIONS` for the model card.
    """
    cols = feature_columns(features, feature_set)
    X = features[cols].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    if pooled and "cluster" in features.columns:
        X["cluster_ff"] = (features["cluster"] == "Fast Food").astype(int)

    if not clean:
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
        return X if keep_cols is None else X.reindex(columns=list(keep_cols), fill_value=0.0)

    decisions: list[tuple[str, str, str]] = []
    X, d1 = _median_impute_with_indicator(X)
    decisions += d1
    X, d2 = _drop_zero_variance(X)
    decisions += d2
    X, d3 = _drop_collinear(X, _INTERP_PRIORITY)
    decisions += d3

    if keep_cols is not None:
        # Enforce the training-fold schema on this (test) fold.
        X = X.reindex(columns=list(keep_cols), fill_value=0.0)

    if standardize:
        cont = [c for c in X.columns if not c.endswith("_missing") and c != "cluster_ff"]
        mu = X[cont].mean()
        sd = X[cont].std(ddof=0).replace(0, 1.0)
        X[cont] = (X[cont] - mu) / sd

    if record:
        DESIGN_DECISIONS.clear()
        DESIGN_DECISIONS.extend(decisions)
    return X


# --------------------------------------------------------------------------- #
# Model fitters (lazy imports)
# --------------------------------------------------------------------------- #


#: Penalizer grid tuned by cross-validated C index (ridge; l1_ratio=0).
COX_PENALIZER_GRID: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5)


def fit_cox(
    X: pd.DataFrame,
    duration: pd.Series,
    event: pd.Series,
    penalizer: float = 0.1,
    l1_ratio: float = 0.0,
):
    """Fit a ridge-penalized lifelines CoxPHFitter, robust to convergence trouble.

    Uses ``l1_ratio=0`` (pure ridge) for a stable Hessian. If Newton-Raphson
    produces a NaN step we retry with a smaller ``step_size`` before giving up.
    """
    from lifelines import CoxPHFitter
    from lifelines.exceptions import ConvergenceError

    df = X.copy()
    df["duration"] = np.asarray(duration, dtype=float)
    df["event"] = np.asarray(event, dtype=int)
    model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    try:
        model.fit(df, duration_col="duration", event_col="event")
        return model
    except ConvergenceError:
        # Smaller, damped Newton steps recover from a NaN delta on stiff matrices.
        model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        model.fit(df, duration_col="duration", event_col="event", fit_options={"step_size": 0.5})
        return model


def tune_cox_penalizer(X, duration, event, penalizers: Sequence[float] = COX_PENALIZER_GRID):
    """Pick the penalizer with the best cross-validated C index (ridge).

    We hold out a simple 5-fold split on rows (grouping is handled upstream by the
    stage driver) and score each penalizer by mean out-of-fold concordance, so the
    choice is not an in-sample optimism artifact. Returns (best_model, best_pen).
    """
    from sklearn.model_selection import KFold

    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    Xr = X.reset_index(drop=True)
    n_splits = min(5, max(2, len(Xr) // 50)) if len(Xr) >= 100 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)

    best_pen = penalizers[0]
    best_c = -np.inf
    for pen in penalizers:
        cs: list[float] = []
        for tr, te in kf.split(Xr):
            try:
                m = fit_cox(Xr.iloc[tr], duration[tr], event[tr], penalizer=pen)
                risk = m.predict_partial_hazard(Xr.iloc[te]).to_numpy().ravel()
                from lrr import evaluation

                cs.append(evaluation.harrell_c(duration[te], event[te], risk))
            except Exception:
                continue
        mean_c = float(np.nanmean(cs)) if cs else -np.inf
        if mean_c > best_c:
            best_c, best_pen = mean_c, pen

    # Refit the chosen penalizer on all rows for the returned model.
    best = fit_cox(X, duration, event, penalizer=best_pen)
    return best, best_pen


def select_penalizer_subsampled(
    X: pd.DataFrame,
    duration,
    event,
    penalizers: Sequence[float] = COX_PENALIZER_GRID,
    subsample: int = 10_000,
    inner_folds: int = 3,
    seed: int = config.SEED,
) -> float:
    """Pick a Cox penalizer with ONE inner CV on a stratified subsample.

    Designed for large-N tiers (for example the ~46k Tier 1 corpus) where a full
    nested CV is far too expensive for negligible benefit. We take a stratified
    (by event) subsample of at most ``subsample`` rows, run a single ``inner_folds``
    KFold, and return the penalizer with the best mean out-of-fold C index. The
    caller refits on the full training fold with the chosen penalizer.
    """
    from sklearn.model_selection import KFold

    from lrr import evaluation

    Xr = X.reset_index(drop=True)
    dur = np.asarray(duration, dtype=float)
    evt = np.asarray(event, dtype=int)
    n = len(Xr)
    if n > subsample:
        rng = np.random.default_rng(seed)
        pos = np.where(evt == 1)[0]
        neg = np.where(evt == 0)[0]
        frac = subsample / n
        k_pos = max(1, int(round(len(pos) * frac)))
        k_neg = subsample - k_pos
        sel = np.concatenate(
            [
                rng.choice(pos, size=min(k_pos, len(pos)), replace=False),
                rng.choice(neg, size=min(k_neg, len(neg)), replace=False),
            ]
        )
        rng.shuffle(sel)
        Xr, dur, evt = Xr.iloc[sel].reset_index(drop=True), dur[sel], evt[sel]

    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_pen, best_c = penalizers[0], -np.inf
    for pen in penalizers:
        cs: list[float] = []
        for tr, te in kf.split(Xr):
            try:
                m = fit_cox(Xr.iloc[tr], dur[tr], evt[tr], penalizer=pen)
                risk = m.predict_partial_hazard(Xr.iloc[te]).to_numpy().ravel()
                cs.append(evaluation.harrell_c(dur[te], evt[te], risk))
            except Exception:
                continue
        mean_c = float(np.nanmean(cs)) if cs else -np.inf
        if mean_c > best_c:
            best_c, best_pen = mean_c, pen
    return best_pen


def fit_rsf(X: pd.DataFrame, duration, event, seed: int = config.SEED):
    """Fit a scikit-survival RandomSurvivalForest."""
    from sksurv.ensemble import RandomSurvivalForest

    y = np.array(
        list(zip(np.asarray(event, dtype=bool), np.asarray(duration, dtype=float))),
        dtype=[("event", "bool"), ("time", "float64")],
    )
    model = RandomSurvivalForest(
        n_estimators=300, min_samples_leaf=10, random_state=seed, n_jobs=-1
    )
    model.fit(X.to_numpy(), y)
    return model


def fit_xgb(X: pd.DataFrame, event, seed: int = config.SEED):
    """Fit an XGBoost classifier on the 24-month event (for comparison)."""
    from xgboost import XGBClassifier

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(X.to_numpy(), np.asarray(event, dtype=int))
    return model


# --------------------------------------------------------------------------- #
# Risk scoring, tiering, drivers
# --------------------------------------------------------------------------- #


def risk_tier(percentile: float, cuts: dict = config.RISK_TIER_CUTS) -> str:
    """Map a 0-100 risk percentile to a tier label.

    High >= 90, Elevated >= 75, Watch >= 50, else Low.
    """
    if percentile >= cuts["High"]:
        return "High"
    if percentile >= cuts["Elevated"]:
        return "Elevated"
    if percentile >= cuts["Watch"]:
        return "Watch"
    return "Low"


def to_percentile(risk_scores: np.ndarray) -> np.ndarray:
    """Convert risk scores to 0-100 percentiles (higher score = higher percentile)."""
    s = pd.Series(np.asarray(risk_scores, dtype=float))
    return (s.rank(pct=True) * 100.0).to_numpy()


def top_drivers(contributions: np.ndarray, feature_names: Sequence[str], k: int = 3) -> list[str]:
    """Return the ``k`` feature names with the largest absolute contribution."""
    contributions = np.asarray(contributions, dtype=float)
    order = np.argsort(-np.abs(contributions))[:k]
    return [str(feature_names[i]) for i in order]


def build_risk_scores(
    features: pd.DataFrame,
    risk_scores: np.ndarray,
    surv_12m: np.ndarray,
    surv_24m: np.ndarray,
    drivers: list[list[str]],
) -> pd.DataFrame:
    """Assemble the risk_scores artifact rows.

    Columns: business_id, chain, cluster, risk_score, risk_percentile, risk_tier,
    top_3_drivers, surv_12m, surv_24m.
    """
    pct = to_percentile(risk_scores)
    return pd.DataFrame(
        {
            "business_id": features["business_id"].to_numpy(),
            "chain": features.get("chain"),
            "cluster": features.get("cluster"),
            "risk_score": np.asarray(risk_scores, dtype=float),
            "risk_percentile": pct,
            "risk_tier": [risk_tier(p) for p in pct],
            "top_3_drivers": [", ".join(d) for d in drivers],
            "surv_12m": np.asarray(surv_12m, dtype=float),
            "surv_24m": np.asarray(surv_24m, dtype=float),
        }
    )


def survival_curves(
    cox_model,
    X: pd.DataFrame,
    business_ids: Sequence[str],
    times_days: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per-location survival curves (long form) for the app.

    Returns columns business_id, t_days, survival. Uses the Cox model's predicted
    survival function sampled at ``times_days`` (default monthly to 24 months).
    """
    if times_days is None:
        times_days = [int(30.4 * m) for m in range(0, config.HORIZON_MONTHS + 1)]
    sf = cox_model.predict_survival_function(X, times=times_days)
    rows = []
    for col, bid in zip(sf.columns, business_ids):
        for t in times_days:
            rows.append({"business_id": bid, "t_days": int(t), "survival": float(sf.loc[t, col])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Full-corpus two-tier additions (Tier 1 fit, out-of-fold stacking, final score)
# --------------------------------------------------------------------------- #

#: Tier 1 feature columns (engagement + stars + sentiment aggregates; no topics).
TIER1_FEATURE_COLUMNS: tuple[str, ...] = (
    "review_vol_12m",
    "review_vol_6m",
    "review_vol_3m",
    "tip_vol_12m",
    "tip_vol_6m",
    "tip_vol_3m",
    "checkin_vol_12m",
    "checkin_vol_6m",
    "checkin_vol_3m",
    "review_velocity_slope",
    "stars_all",
    "stars_12m",
    "stars_trend",
    "tip_vol",
    "checkin_vol",
    "sentiment_mean",
    "sentiment_slope",
)

#: Re-exported so callers write one documented rule.
FINAL_SCORE_RULE: str = config.FINAL_SCORE_RULE


def tier1_design_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Dense Tier 1 design matrix (median-imputed) over the Tier 1 columns present."""
    cols = [c for c in TIER1_FEATURE_COLUMNS if c in features.columns]
    X = features[cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X


def fit_tier1(
    features: pd.DataFrame,
    duration,
    event,
    penalizer: float = 0.01,
):
    """Fit the Tier 1 corpus Cox model with a fixed ridge penalizer.

    At the ~46k-restaurant corpus scale the penalizer barely moves the fit, so we
    use a fixed small value (0.01) rather than an expensive inner CV tune.
    """
    X = tier1_design_matrix(features)
    return fit_cox(X, duration, event, penalizer=penalizer)


def oof_tier1_scores(
    features: pd.DataFrame,
    duration,
    event,
    group: pd.Series,
    n_folds: int = config.N_FOLDS,
    seed: int = config.SEED,
) -> np.ndarray:
    """Out-of-fold Tier 1 risk scores (the leakage-safe stacking feature).

    Rows sharing a ``group`` always fall in the same held-out fold, and each fold's
    Tier 1 model is trained on the complement. A row's score therefore never comes
    from a model that saw its group. Returns an array aligned to ``features``.
    """
    from sklearn.model_selection import GroupKFold

    features = features.reset_index(drop=True)
    duration = np.asarray(duration, dtype=float)
    event = np.asarray(event, dtype=int)
    group = pd.Series(group).reset_index(drop=True).astype(str)

    scores = np.full(len(features), np.nan, dtype=float)
    k = min(n_folds, group.nunique())
    if k < 2:
        model = fit_tier1(features, duration, event)
        return model.predict_partial_hazard(tier1_design_matrix(features)).to_numpy().ravel()

    gkf = GroupKFold(n_splits=k)
    for train_idx, test_idx in gkf.split(features, groups=group):
        model = fit_tier1(features.iloc[train_idx], duration[train_idx], event[train_idx])
        X_test = tier1_design_matrix(features.iloc[test_idx])
        scores[test_idx] = model.predict_partial_hazard(X_test).to_numpy().ravel()
    return scores


def assert_oof_no_group_leak(group: pd.Series, fold_of_row: pd.Series) -> None:
    """Assert each group maps to exactly one fold (the stacking invariant).

    If any group appears in more than one fold, a row could have received a Tier 1
    score from a model trained on its own group. Raises on violation.
    """
    g = pd.DataFrame(
        {
            "group": pd.Series(group).astype(str).to_numpy(),
            "fold": pd.Series(fold_of_row).to_numpy(),
        }
    )
    bad = g.groupby("group")["fold"].nunique()
    offenders = bad[bad > 1].index.tolist()
    assert not offenders, f"Group(s) split across folds (leakage): {offenders}"


def final_score(
    risk_df: pd.DataFrame,
    tier1_col: str = "tier1_score",
    tier2_col: str = "tier2_score",
    cohort_col: str = "is_cohort",
) -> np.ndarray:
    """Apply the documented final-score rule.

    Cohort locations use ``tier2_score`` (the cluster-specific model that already
    stacks the out-of-fold Tier 1 signal); non-cohort restaurants use
    ``tier1_score``. See :data:`FINAL_SCORE_RULE`.
    """
    t1 = risk_df[tier1_col].to_numpy(dtype=float)
    if tier2_col in risk_df.columns and cohort_col in risk_df.columns:
        t2 = risk_df[tier2_col].to_numpy(dtype=float)
        is_cohort = risk_df[cohort_col].to_numpy(dtype=bool)
        return np.where(is_cohort & ~np.isnan(t2), t2, t1)
    return t1
