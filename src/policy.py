"""Layer 2: the next-move policy.

Ordered guard clauses, deliberately. The order *is* the conversational
etiquette: finish what you promised, follow a good tangent, clear up
ambiguity, never re-ask, notice what was volunteered, and only then
carry on down your list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .ledger import Commitment, Ledger


class Move(Enum):
    ACKNOWLEDGE_AND_DEEPEN = auto()
    CLARIFY = auto()
    SKIP_ANSWERED = auto()
    ALLOW_DIGRESSION = auto()
    RETURN_TO_OBJECTIVE = auto()
    CLOSE_TOPIC = auto()
    ASK_OBJECTIVE = auto()  # the default: nothing special happened, ask the next one


@dataclass
class Decision:
    move: Move
    objective_id: str | None = None
    skipped: list[str] = field(default_factory=list)
    reason: str = ""


def next_move(ledger: Ledger) -> Decision:
    last = ledger.last_extraction()

    if ledger.commitments and _commitment_due(ledger):
        c = ledger.commitments.popleft()
        return Decision(Move.RETURN_TO_OBJECTIVE, c.objective_id, reason=f"commitment from {c.created_turn} is due")

    if last and last.digression and last.digression_value == "valuable":
        open_id = ledger.active_objective_id()
        if open_id:
            ledger.commitments.append(Commitment(open_id, "digression", last.turn.id))
        return Decision(Move.ALLOW_DIGRESSION, reason="valuable digression, return scheduled")

    if last and last.ambiguous_objective:
        return Decision(Move.CLARIFY, last.ambiguous_objective, reason="last turn was ambiguous")

    skipped = ledger.advance_cursor_past_covered()
    if skipped:
        return Decision(Move.SKIP_ANSWERED, ledger.cursor_objective_id(), skipped, reason="already covered")

    if last and last.notable and last.turn.id not in ledger.acknowledged:
        ledger.acknowledged.add(last.turn.id)
        return Decision(Move.ACKNOWLEDGE_AND_DEEPEN, reason="participant volunteered something notable")

    oid = ledger.cursor_objective_id()
    if oid is None:
        return Decision(Move.CLOSE_TOPIC, reason="every objective is covered")
    ledger.mark_asked(oid)
    ledger.cursor += 1
    return Decision(Move.ASK_OBJECTIVE, oid, reason="next unasked objective")


def _commitment_due(ledger: Ledger) -> bool:
    """A scheduled return comes due once the digression has run its course."""
    c = ledger.commitments[0]
    trailing = ledger.trailing_digression_turns()
    return trailing == 0 or trailing >= c.max_digression_turns
