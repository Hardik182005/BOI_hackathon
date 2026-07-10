"""Structured logging for MuleGuard.

Plain-text console + optional JSON-lines file. Secrets are never logged:
the formatter redacts values of keys matching SENSITIVE_PATTERNS.
"""
from __future__ import annotations

import json
import logging as _stdlog
import re
import sys
import time
from pathlib import Path

SENSITIVE_PATTERNS = re.compile(r"(api[_-]?key|secret|password|token|authorization)", re.I)

_CONFIGURED = False


class _JsonFormatter(_stdlog.Formatter):
    def format(self, record: _stdlog.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(_redact(payload), default=str)


def _redact(obj):
    if isinstance(obj, dict):
        return {
            k: ("***REDACTED***" if SENSITIVE_PATTERNS.search(str(k)) else _redact(v))
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_redact(v) for v in obj]
    return obj


def configure(level: int = _stdlog.INFO, json_file: Path | None = None) -> None:
    """Idempotent root configuration: console always, JSON file optional."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    root = _stdlog.getLogger("muleguard")
    root.setLevel(level)
    console = _stdlog.StreamHandler(sys.stderr)
    console.setFormatter(_stdlog.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root.addHandler(console)
    if json_file is not None:
        json_file.parent.mkdir(parents=True, exist_ok=True)
        fh = _stdlog.FileHandler(json_file, encoding="utf-8")
        fh.setFormatter(_JsonFormatter())
        root.addHandler(fh)
    _CONFIGURED = True


def get_logger(name: str) -> _stdlog.Logger:
    configure()
    return _stdlog.getLogger(f"muleguard.{name}")
