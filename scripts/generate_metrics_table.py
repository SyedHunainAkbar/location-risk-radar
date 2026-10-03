"""Generate the headline-results table from artifacts/model_metrics.parquet.

Numbers are never hand-typed. This reads the metrics artifact produced by the
survival pipelines and writes a Markdown table to docs/metrics_table.md, then injects
it into README.md between the HEADLINE-METRICS markers. If the artifact is absent
(pipelines not yet run), it writes a clear placeholder instead of fabricating values.

Usage:
    python scripts/generate_metrics_table.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from lrr import config  # noqa: E402

START = "<!-- HEADLINE-METRICS:START -->"
END = "<!-- HEADLINE-METRICS:END -->"
DOCS_TABLE = _ROOT / "docs" / "metrics_table.md"
README = _ROOT / "README.md"


def _fmt_ci(row, point, lo, hi) -> str:
    p = row.get(point)
    if p is None or pd.isna(p):
        return "n/a"
    lo_v, hi_v = row.get(lo), row.get(hi)
    if lo_v is not None and hi_v is not None and not pd.isna(lo_v) and not pd.isna(hi_v):
        return f"{p:.3f} ({lo_v:.3f}, {hi_v:.3f})"
    return f"{p:.3f}"


def build_table() -> str:
    path = config.ARTIFACTS_DIR / config.MODEL_METRICS_FILE
    if not path.exists():
        return (
            "_Headline metrics are generated from "
            "`artifacts/model_metrics.parquet`._\n\n"
            "Run the pipeline (`pipeline/04_features.py` then `05_survival.py` and "
            "`05b_tier1_survival.py`) and re-run "
            "`python scripts/generate_metrics_table.py` to populate this table with "
            "the C index, time-dependent AUC, and integrated Brier score, each with "
            "95% confidence intervals. We never hand-type these numbers.\n"
        )
    df = pd.read_parquet(path)
    lines = [
        "| Tier | Scope | Feature set | Model | C index (95% CI) | "
        "Uno C | AUC 12m | AUC 24m | IBS |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r.get('tier', '')} | {r.get('scope', '')} | "
            f"{r.get('feature_set', '')} | {r.get('model', '')} | "
            f"{_fmt_ci(r, 'harrell_c', 'harrell_c_lo', 'harrell_c_hi')} | "
            f"{_num(r.get('uno_c'))} | {_num(r.get('auc_12m'))} | "
            f"{_num(r.get('auc_24m'))} | {_num(r.get('ibs'))} |"
        )
    note = (
        "\n_Generated from `artifacts/model_metrics.parquet` by "
        "`scripts/generate_metrics_table.py`. Numbers are not hand-typed._\n"
    )
    return "\n".join(lines) + "\n" + note


def _num(v) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{float(v):.3f}"


def main() -> int:
    table = build_table()
    DOCS_TABLE.parent.mkdir(parents=True, exist_ok=True)
    DOCS_TABLE.write_text("# Headline results\n\n" + table, encoding="utf-8")
    print(f"Wrote {DOCS_TABLE}")

    if README.exists():
        text = README.read_text(encoding="utf-8")
        if START in text and END in text:
            pre = text.split(START)[0]
            post = text.split(END)[1]
            text = f"{pre}{START}\n{table}\n{END}{post}"
            README.write_text(text, encoding="utf-8")
            print("Injected headline metrics into README.md")
        else:
            print("README markers not found; skipped README injection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
