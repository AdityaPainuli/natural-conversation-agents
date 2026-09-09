"""Layer 3: delivery. Minimal on purpose.

Two things only, because they are the two that people actually notice:
match the participant's register, and do not write three paragraphs when
they wrote four words.
"""

from __future__ import annotations

import re
from typing import Literal

Register = Literal["standard", "brief"]

BRIEF_WORD_LIMIT = 8
FORMAL_LEAD_INS = ("That is useful. ", "That's useful, and ", "Thank you. ", "I appreciate that. ")


def detect_register(text: str) -> Register:
    return "brief" if len(text.split()) <= BRIEF_WORD_LIMIT else "standard"


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s.strip()]


def match_register(response: str, register: Register) -> str:
    """Short turn in, short turn out."""
    if register != "brief":
        return response
    for lead in FORMAL_LEAD_INS:
        if response.startswith(lead):
            response = response[len(lead):]
    parts = sentences(response)
    return " ".join(parts[-2:]) if len(parts) > 2 else response


def cap_length(response: str, max_sentences: int = 3) -> str:
    parts = sentences(response)
    return " ".join(parts[:max_sentences]) if len(parts) > max_sentences else response
