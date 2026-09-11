"""Thin LLM client. Two modes: live (Anthropic) and stub (no key, no network).

Stub mode exists so the talk demo is deterministic on conference wifi. Live
mode exists so the repo is honest about what the real thing costs you.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000  # adaptive thinking is on by default and spends this budget too


class StubModeError(RuntimeError):
    """Raised when live-only code is reached in stub mode."""


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader. Not worth a dependency for one file of KEY=value."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass
class LLMClient:
    mode: str = "stub"  # "stub" | "live"
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls, mode: str | None = None) -> "LLMClient":
        load_dotenv()
        return cls(
            mode=mode or os.environ.get("LLM_MODE", "stub"),
            model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )

    @property
    def is_live(self) -> bool:
        return self.mode == "live"

    def _client(self) -> Any:
        import anthropic  # imported lazily so stub mode needs no dependency

        return anthropic.Anthropic()

    # Note: no temperature. Current models reject sampling parameters, so both
    # agents are held to the same settings by having no knobs at all.
    def complete_text(self, system: str, user: str) -> str:
        if not self.is_live:
            raise StubModeError("complete_text called in stub mode")
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        _check_refusal(msg)
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Strict JSON via a single forced tool call. No parsing heroics."""
        if not self.is_live:
            raise StubModeError("complete_json called in stub mode")
        tool = {"name": "emit", "description": "Emit the structured result.", "input_schema": schema}
        msg = self._client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": "emit"},
            messages=[{"role": "user", "content": user}],
        )
        _check_refusal(msg)
        for block in msg.content:
            if block.type == "tool_use":
                return dict(block.input)
        raise RuntimeError(f"no tool_use block: {json.dumps([b.type for b in msg.content])}")


def _check_refusal(msg: Any) -> None:
    if msg.stop_reason == "refusal":
        raise RuntimeError(f"model declined: {getattr(msg.stop_details, 'explanation', '')}")
