"""Static recommended-action playbook keyed by Voice-of-Customer issue.

Loads ``config/actions.yaml`` (pyyaml, with an in-code fallback) and maps the issues
present in a brief to ordered action lists, prioritized by issue prevalence.
"""

from __future__ import annotations

from pathlib import Path

from lrr import config

_FALLBACK = {
    issue: [f"Review {issue.replace('_', ' ')} concerns with the manager."]
    for issue in config.VOC_ISSUES
}


def load_actions(path: Path = config.ACTIONS_YAML) -> dict[str, list[str]]:
    """Load the issue -> actions playbook. Falls back if yaml/file is absent."""
    try:
        import yaml
    except ImportError:
        return dict(_FALLBACK)
    path = Path(path)
    if not path.exists():
        return dict(_FALLBACK)
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return dict(_FALLBACK)
    return {k: list(v) for k, v in doc.items()} or dict(_FALLBACK)


def recommend(
    issue_prevalence: dict,
    actions: dict | None = None,
    max_actions: int = 6,
) -> list[str]:
    """Return recommended actions for the present issues, ordered by prevalence.

    Issues are ranked by prevalence (desc); their actions are concatenated and
    de-duplicated, capped at ``max_actions``.
    """
    actions = actions or load_actions()
    ordered_issues = sorted(issue_prevalence.items(), key=lambda kv: -kv[1])
    out: list[str] = []
    seen: set[str] = set()
    for issue, _ in ordered_issues:
        for action in actions.get(issue, []):
            if action not in seen:
                out.append(action)
                seen.add(action)
            if len(out) >= max_actions:
                return out
    return out
