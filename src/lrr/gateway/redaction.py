"""Key redaction for logs. Masks sk-, sk_, and nvapi- patterns everywhere.

Keys must never reach a log sink. We install a logging filter on the gateway logger
that scrubs the record message and args before emission, and expose a reusable
``redact`` helper for tests and ad hoc use.
"""

from __future__ import annotations

import logging
import re

# Match common secret shapes without being so greedy they swallow surrounding text.
_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"sk_[A-Za-z0-9_\-]{6,}"),
    re.compile(r"nvapi-[A-Za-z0-9_\-]{6,}"),
]

_MASK = "[REDACTED]"


def redact(text: str) -> str:
    """Return ``text`` with any sk-/sk_/nvapi- secrets masked."""
    out = str(text)
    for pat in _PATTERNS:
        out = pat.sub(_MASK, out)
    return out


class RedactionFilter(logging.Filter):
    """Logging filter that masks secrets in the message and args."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(redact(a) if isinstance(a, str) else a for a in record.args)
        return True


def get_logger(name: str = "lrr.gateway") -> logging.Logger:
    """Return the gateway logger with the redaction filter installed once."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, RedactionFilter) for f in logger.filters):
        logger.addFilter(RedactionFilter())
    return logger
