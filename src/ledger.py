"""Layer 1: the conversation ledger.

The ledger is the whole idea. An agent that keeps one can tell the difference
between "I have not asked this yet" and "they already told me, sideways."
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Coverage(Enum):
    UNASKED = "unasked"
    PARTIAL = "partial"
    ANSWERED = "answered"
    IMPLICITLY_ANSWERED = "implicitly_answered"


COVERED = (Coverage.ANSWERED, Coverage.IMPLICITLY_ANSWERED)


@dataclass(frozen=True)
class Objective:
    id: str
    question: str
    label: str  # short noun phrase, used when the agent talks *about* the objective


@dataclass
class ObjectiveState:
    objective: Objective
    coverage: Coverage = Coverage.UNASKED
    asked: bool = False
    evidence: list[str] = field(default_factory=list)  # turn ids that answered it


@dataclass(frozen=True)
class Preference:
    polarity: Literal["likes", "dislikes"]
    topic: str
    turn_id: str


@dataclass
class Commitment:
    """Something the agent promised itself it would come back to."""

    objective_id: str
    reason: str
    created_turn: str
    max_digression_turns: int = 2


@dataclass
class Turn:
    id: str
    speaker: Literal["participant", "agent"]
    text: str


@dataclass
class TurnExtraction:
    turn: Turn
    answers_objectives: list[str] = field(default_factory=list)
    implies_objectives: list[str] = field(default_factory=list)
    preferences: list[Preference] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    summary: str = ""
    digression: bool = False
    digression_value: Literal["valuable", "noise"] | None = None
    ambiguous_objective: str | None = None

    @property
    def notable(self) -> bool:
        """Did the participant volunteer something worth stopping for?"""
        return bool(self.preferences or self.implies_objectives)


@dataclass
class Ledger:
    objectives: dict[str, ObjectiveState]
    order: list[str]
    cursor: int = 0
    preferences: list[Preference] = field(default_factory=list)
    commitments: deque[Commitment] = field(default_factory=deque)
    history: list[Turn] = field(default_factory=list)
    extractions: list[TurnExtraction] = field(default_factory=list)
    acknowledged: set[str] = field(default_factory=set)

    @classmethod
    def from_objectives(cls, objectives: list[Objective]) -> "Ledger":
        return cls(
            objectives={o.id: ObjectiveState(o) for o in objectives},
            order=[o.id for o in objectives],
        )

    # --- slide 9: this is the whole update rule ---------------------------
    def update(self, ex: TurnExtraction) -> None:
        self.history.append(ex.turn)
        self.extractions.append(ex)
        for oid in ex.answers_objectives:
            self._cover(oid, Coverage.ANSWERED, ex.turn.id)
        for oid in ex.implies_objectives:
            if self.objectives[oid].coverage not in COVERED:
                self._cover(oid, Coverage.IMPLICITLY_ANSWERED, ex.turn.id)  # <- the point
        if ex.ambiguous_objective and self.objectives[ex.ambiguous_objective].coverage is Coverage.UNASKED:
            self.objectives[ex.ambiguous_objective].coverage = Coverage.PARTIAL
        self.preferences.extend(ex.preferences)

    def _cover(self, oid: str, coverage: Coverage, turn_id: str) -> None:
        state = self.objectives[oid]
        state.coverage = coverage
        state.evidence.append(turn_id)
    # ----------------------------------------------------------------------

    def record_agent_turn(self, turn: Turn) -> None:
        self.history.append(turn)

    def state(self, oid: str) -> ObjectiveState:
        return self.objectives[oid]

    def cursor_objective_id(self) -> str | None:
        return self.order[self.cursor] if self.cursor < len(self.order) else None

    def mark_asked(self, oid: str) -> None:
        self.objectives[oid].asked = True

    def advance_cursor_past_covered(self) -> list[str]:
        """Move the cursor past anything already covered. Returns the ids we
        skipped without ever having asked them, which is the interesting case."""
        skipped: list[str] = []
        while (oid := self.cursor_objective_id()) is not None:
            state = self.objectives[oid]
            if state.coverage not in COVERED:
                break
            if not state.asked:
                skipped.append(oid)
            self.cursor += 1
        return skipped

    def active_objective_id(self) -> str | None:
        """Last objective we asked about that is still open."""
        for oid in reversed(self.order):
            state = self.objectives[oid]
            if state.asked and state.coverage not in COVERED:
                return oid
        return self.cursor_objective_id()

    def last_extraction(self) -> TurnExtraction | None:
        return self.extractions[-1] if self.extractions else None

    def trailing_digression_turns(self) -> int:
        n = 0
        for ex in reversed(self.extractions):
            if not ex.digression:
                break
            n += 1
        return n

    def preferences_for(self, polarity: Literal["likes", "dislikes"] | None = None) -> list[Preference]:
        return [p for p in self.preferences if polarity is None or p.polarity == polarity]

    def summary_lines(self) -> list[str]:
        """Compact ledger view, used in prompts and in the terminal panel."""
        lines = [
            f"{oid}: {self.objectives[oid].coverage.value}"
            + (f" (evidence: {', '.join(self.objectives[oid].evidence)})" if self.objectives[oid].evidence else "")
            for oid in self.order
        ]
        for p in self.preferences:
            lines.append(f"preference: {p.polarity} {p.topic} ({p.turn_id})")
        for c in self.commitments:
            lines.append(f"commitment: return to {c.objective_id} ({c.reason})")
        return lines
