"""Agent 1: Quant Analyst. Placeholder-only driver explanations.

The model proposes driver explanation templates that reference fact-sheet values
only through ``{placeholder}`` tokens. Python enforces that no literal digit appears
outside a placeholder (the numeric-fidelity guarantee), retries on violation, and
renders the final text from the fact sheet so every number is Python-computed.
"""

from __future__ import annotations

import json
import re

from lrr import config, gateway
from lrr.agents.schemas import QuantOutput

PLACEHOLDER_RE = re.compile(r"\{[a-z0-9_]+\}")
_DIGIT_RE = re.compile(r"\d")


def enforce_placeholders(template: str) -> bool:
    """True if ``template`` has no literal digit outside a {placeholder}.

    We strip every placeholder, then any remaining digit is a violation.
    """
    stripped = PLACEHOLDER_RE.sub("", str(template))
    return _DIGIT_RE.search(stripped) is None


def _system_prompt(available_keys: list[str]) -> str:
    keys = ", ".join("{" + k + "}" for k in available_keys)
    return (
        "You are a quantitative risk analyst. For each important driver, write a short "
        "explanation TEMPLATE. The template MUST reference values only through these "
        f"placeholders: {keys}. Never write a literal number; always use a placeholder. "
        'Return JSON {"drivers": [{"driver_key": str, "direction": "up"|"down", '
        '"explanation_template": str}]}. No prose outside JSON.'
    )


def _placeholder_keys(factsheet: dict) -> list[str]:
    """Fact-sheet keys usable as placeholders (scalars rendered to strings)."""
    return [k for k, v in factsheet.items() if isinstance(v, str)]


def build_prompt(factsheet: dict) -> list[dict]:
    keys = _placeholder_keys(factsheet)
    drivers = factsheet.get("top_drivers", [])
    user = (
        "Fact sheet (values are already formatted strings; reference them by key as "
        "placeholders, do not copy the numbers):\n"
        + json.dumps({k: v for k, v in factsheet.items() if k != "top_drivers"}, ensure_ascii=False)
        + "\n\nTop drivers (by absolute SHAP contribution): "
        + json.dumps([d["driver_key"] for d in drivers])
        + "\n\nWrite one template per driver."
    )
    return [{"role": "system", "content": _system_prompt(keys)}, {"role": "user", "content": user}]


def run_quant(
    factsheet: dict,
    retries: int = config.QUANT_PLACEHOLDER_RETRIES,
    completion_fn=None,
) -> QuantOutput:
    """Call the Quant Analyst and enforce the placeholder-only rule.

    Retries up to ``retries`` times when any template contains a literal digit,
    feeding the offending template back. Raises ``ValueError`` if it never complies.
    """
    messages = build_prompt(factsheet)
    last_bad: str | None = None
    for _ in range(retries + 1):
        result = gateway.chat(
            messages, role="quant_analyst", schema=QuantOutput, completion_fn=completion_fn
        )
        parsed = result.parsed
        if parsed is None:
            parsed = _coerce(result.text, factsheet)
        violations = [
            d.explanation_template
            for d in parsed.drivers
            if not enforce_placeholders(d.explanation_template)
        ]
        if not violations:
            return parsed
        last_bad = violations[0]
        messages = messages + [
            {"role": "assistant", "content": result.text},
            {
                "role": "user",
                "content": f"This template has a literal number: {last_bad!r}. "
                "Rewrite ALL templates using only {placeholder} tokens, no digits.",
            },
        ]
    raise ValueError(f"Quant Analyst kept emitting literal digits (last: {last_bad!r}).")


def _coerce(text: str, factsheet: dict) -> QuantOutput:
    """Best-effort parse when the gateway did not attach a parsed schema.

    Falls back to a single safe placeholder-only driver so the committee can proceed
    offline (mock provider) without violating the digit rule.
    """
    try:
        start, end = text.find("{"), text.rfind("}")
        obj = json.loads(text[start : end + 1])
        return QuantOutput.model_validate(obj)
    except Exception:
        drivers = factsheet.get("top_drivers", [])
        key = drivers[0]["driver_key"] if drivers else "risk_percentile"
        direction = "up" if drivers and drivers[0]["contribution"] >= 0 else "down"
        tmpl = (
            "This location sits at the {risk_percentile} percentile with a review "
            "velocity ratio of {velocity_ratio}."
        )
        return QuantOutput(
            drivers=[{"driver_key": key, "direction": direction, "explanation_template": tmpl}]
        )


def render(template: str, factsheet: dict) -> str:
    """Render a placeholder-only template from the fact sheet.

    Unknown placeholders render as 'n/a' so a stray key never leaks a brace.
    """

    def sub(match: re.Match) -> str:
        key = match.group(0)[1:-1]
        value = factsheet.get(key, "n/a")
        return str(value) if not isinstance(value, (list, dict)) else "n/a"

    return PLACEHOLDER_RE.sub(sub, str(template))


def render_drivers(parsed: QuantOutput, factsheet: dict) -> list[dict]:
    """Render every driver template to final text (Python supplies the numbers)."""
    out = []
    for d in parsed.drivers:
        out.append(
            {
                "driver_key": d.driver_key,
                "direction": d.direction,
                "text": render(d.explanation_template, factsheet),
            }
        )
    return out


# Numbers in rendered text: integers, decimals, and percentages.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?%?")


def _factsheet_number_strings(factsheet: dict) -> set[str]:
    """Collect every numeric substring that appears in the fact-sheet values.

    These are the only numbers allowed to appear in rendered driver text.
    """
    allowed: set[str] = set()
    for value in factsheet.values():
        if isinstance(value, str):
            allowed.update(_NUMBER_RE.findall(value))
    return allowed


def scan_rendered_fidelity(rendered_text: str, factsheet: dict) -> list[str]:
    """Independent check: every number in rendered text must trace to the fact sheet.

    Returns the list of numeric tokens in ``rendered_text`` that are NOT present in
    any fact-sheet value. An empty list means the rendering invented no numbers. This
    verifies numeric fidelity AFTER rendering, not just the pre-render placeholder
    guard.
    """
    allowed = _factsheet_number_strings(factsheet)
    found = _NUMBER_RE.findall(str(rendered_text))
    return [tok for tok in found if tok not in allowed]


def numeric_fidelity_rate(driver_texts, factsheet: dict) -> float:
    """Fraction of rendered driver texts whose numbers all trace to the fact sheet."""
    texts = list(driver_texts)
    if not texts:
        return 1.0
    ok = sum(1 for t in texts if not scan_rendered_fidelity(t, factsheet))
    return ok / len(texts)
