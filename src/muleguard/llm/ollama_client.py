"""Guarded Ollama client.

Flow: build prompt from verified facts -> call local Ollama with temperature 0
and a JSON format constraint -> validate against schema + hallucination rules
-> on ANY failure (network, timeout, invalid, hallucination) fall back to the
deterministic template. A circuit breaker skips calls after repeated failures.

The scoring pipeline never depends on this module succeeding.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

from muleguard import settings
from muleguard.llm.deterministic_fallback import deterministic_narrative
from muleguard.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from muleguard.llm.schemas import NarratorInput, NarratorOutput
from muleguard.llm.validator import validate_llm_output
from muleguard.logging import get_logger

log = get_logger("llm.ollama")


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_seconds: float):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.consecutive_failures = 0
        self.opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if time.monotonic() - self.opened_at >= self.cooldown_seconds:
            self.opened_at = None  # half-open: allow a probe
            self.consecutive_failures = 0
            return False
        return True

    def record(self, success: bool) -> None:
        if success:
            self.consecutive_failures = 0
            self.opened_at = None
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.failure_threshold:
                self.opened_at = time.monotonic()


class OllamaNarrator:
    def __init__(self) -> None:
        cfg = settings.load_config("ollama")
        self.cfg = cfg
        self.breaker = CircuitBreaker(
            int(cfg["circuit_breaker"]["failure_threshold"]),
            float(cfg["circuit_breaker"]["cooldown_seconds"]),
        )
        self._model: str | None = None

    # ---- availability -------------------------------------------------
    def available_model(self) -> str | None:
        """First preferred model present on the local Ollama, else None."""
        if self._model:
            return self._model
        try:
            r = httpx.get(f"{self.cfg['base_url']}/api/tags", timeout=3.0)
            r.raise_for_status()
            local = {m["name"] for m in r.json().get("models", [])}
            for want in self.cfg["model_preference"]:
                if want in local:
                    self._model = want
                    return want
        except Exception:
            return None
        return None

    # ---- narration ----------------------------------------------------
    def narrate(self, ctx: NarratorInput) -> dict[str, Any]:
        """Always returns a narrative. source: 'ollama' | 'deterministic'."""
        result = {
            "source": "deterministic",
            "narrative": None,
            "llm_rejected_reasons": [],
            "model": None,
        }
        if not self.cfg.get("enabled", True) or self.breaker.is_open:
            result["narrative"] = deterministic_narrative(ctx).model_dump()
            return result

        model = self.available_model()
        if model is None:
            result["narrative"] = deterministic_narrative(ctx).model_dump()
            return result

        attempts = 1 + int(self.cfg.get("retries", 1))
        for attempt in range(attempts):
            raw = self._call(model, ctx)
            if raw is None:
                self.breaker.record(False)
                continue
            out, reasons = validate_llm_output(raw, ctx)
            if out is not None:
                self.breaker.record(True)
                result.update(source="ollama", narrative=out.model_dump(), model=model)
                return result
            result["llm_rejected_reasons"].extend(reasons)
            log.warning("LLM output rejected (attempt %d): %s", attempt + 1, reasons)
            self.breaker.record(False)

        result["narrative"] = deterministic_narrative(ctx).model_dump()
        return result

    def _call(self, model: str, ctx: NarratorInput) -> str | None:
        try:
            r = httpx.post(
                f"{self.cfg['base_url']}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(ctx)},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": float(self.cfg["temperature"]),
                        "num_predict": int(self.cfg["max_tokens"]),
                    },
                },
                timeout=float(self.cfg["timeout_seconds"]),
            )
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:  # network, timeout, HTTP, JSON structure
            log.warning("Ollama call failed: %s", type(e).__name__)
            return None
