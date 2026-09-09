"""Thin LLM client. Two modes: live (Anthropic) and stub (no key, no network).

Stub mode exists so the talk demo is deterministic on conference wifi. Live
mode exists so the repo is honest about what the real thing costs you.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_TEMPERATURE = 0.0


class StubModeError(RuntimeError):
    """Raised when live-only code is reached in stub mode."""


@dataclass
class LLMClient:
    mode: str = "stub"  # "stub" | "live"
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls, mode: str | None = None) -> "LLMClient":
        return cls(
            mode=mode or os.environ.get("LLM_MODE", "stub"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            temperature=float(os.environ.get("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)),
        )

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def _client(self) -> Any:
        import anthropic  # imported lazily so stub mode needs no dependency

        return anthropic.Anthropic()

    def complete_text(self, system: str, user: str, max_tokens: int = 400) -> str:
        if not self.is_live:
            raise StubModeError("complete_text called in stub mode")
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text").strip()

    def complete_json(self, system: str, user: str, schema: dict[str, Any], max_tokens: int = 1000) -> dict[str, Any]:
        """Strict JSON via a single forced tool call. No parsing heroics."""
        if not self.is_live:
            raise StubModeError("complete_json called in stub mode")
        tool = {"name": "emit", "description": "Emit the structured result.", "input_schema": schema}
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError(f"model returned no tool_use block: {json.dumps([b.type for b in msg.content])}")
