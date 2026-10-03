"""Usage ledger: one row per gateway call.

Records role, provider, model, latency, token counts, estimated cost, cache-hit, and
whether a fallback was used. Offline runs flush to ``artifacts/llm_ledger.parquet``;
the app keeps the ledger in memory (session state). No secrets are recorded.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from lrr import config


@dataclass
class LedgerRow:
    role: str
    provider: str
    model: str | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    est_cost_usd: float
    cache_hit: bool
    fallback_used: bool


def estimate_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from the per-1K price table; 0.0 for unknown/mock."""
    price = config.MODEL_PRICES_PER_1K.get(str(model), (0.0, 0.0))
    return round(prompt_tokens / 1000 * price[0] + completion_tokens / 1000 * price[1], 6)


class Ledger:
    """Accumulates ledger rows; flushes to parquet or exposes a DataFrame."""

    def __init__(self) -> None:
        self.rows: list[LedgerRow] = []

    def append(self, row: LedgerRow) -> None:
        self.rows.append(row)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([asdict(r) for r in self.rows])

    def flush(self, path: Path = None) -> Path | None:
        """Append the ledger to the parquet artifact (offline). Returns the path."""
        if not self.rows:
            return None
        import pandas as pd

        path = Path(path or (config.ARTIFACTS_DIR / config.LLM_LEDGER_FILE))
        path.parent.mkdir(parents=True, exist_ok=True)
        new = self.to_frame()
        if path.exists():
            prior = pd.read_parquet(path)
            new = pd.concat([prior, new], ignore_index=True)
        new.to_parquet(path, index=False)
        return path
