"""Layer 2: the next-move policy.

Ordered guard clauses, deliberately. The order *is* the conversational
etiquette: finish what you promised, follow a good tangent, clear up
ambiguity, never re-ask, notice what was volunteered, and only then
carry on down your list.

Acknowledgment is not one of the guards. It composes with whichever guard
fires, because "reflect what they just gave you" and "skip the question you
no longer need to ask" are not competing choices, they are the same turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .ledger import Commitment, Ledger, Reference


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
    acknowledge: list[Reference] = field(default_factory=list)


def next_move(ledger: Ledger) -> Decision:
    last = ledger.last_extraction()
    acknowledge = pending_references(ledger)

    def decide(move: Move, objective_id: str | None = None, skipped: list[str] | None = None,
               reason: str = "") -> Decision:
        return Decision(move, objective_id, skipped or [], reason, acknowledge)

    if ledger.commitments and _commitment_due(ledger):
        c = ledger.commitments.popleft()
        return decide(Move.RETURN_TO_OBJECTIVE, c.objective_id, reason=f"commitment from {c.created_turn} is due")

    if last and last.digression and last.digression_value == "valuable":
        open_id = ledger.active_objective_id()
        if open_id:
            ledger.commitments.append(Commitment(open_id, "digression", last.turn.id))
        return decide(Move.ALLOW_DIGRESSION, reason="valuable digression, return scheduled")

    if last and last.ambiguous_objective:
        return decide(Move.CLARIFY, last.ambiguous_objective, reason="last turn was ambiguous")

    skipped = ledger.advance_cursor_past_covered()
    if skipped:
        return decide(Move.SKIP_ANSWERED, ledger.cursor_objective_id(), skipped, reason="already covered")

    if last and last.notable and last.turn.id not in ledger.acknowledged:
        ledger.acknowledged.add(last.turn.id)
        return decide(Move.ACKNOWLEDGE_AND_DEEPEN, reason="participant volunteered something notable")

    oid = ledger.cursor_objective_id()
    if oid is None:
        return decide(Move.CLOSE_TOPIC, reason="every objective is covered")
    ledger.mark_asked(oid)
    ledger.cursor += 1
    return decide(Move.ASK_OBJECTIVE, oid, reason="next unasked objective")


def pending_references(ledger: Ledger) -> list[Reference]:
    """What the last turn volunteered, phrased so a reply can reflect it.

    Populated whenever the participant gave something away, whichever guard
    ends up firing.
    """
    last = ledger.last_extraction()
    if last is None or not last.notable:
        return []
    refs = [
        Reference("implicit_answer", ledger.state(oid).objective.label, last.turn.id)
        for oid in last.implies_objectives
    ]
    refs += [
        Reference("preference", _preference_phrase(p.polarity, p.topic), last.turn.id)
        for p in last.preferences
    ]
    return refs


def _preference_phrase(polarity: str, topic: str) -> str:
    return f"can't stand {topic}" if polarity == "dislikes" else f"like {topic}"


def _commitment_due(ledger: Ledger) -> bool:
    """A scheduled return comes due once the digression has run its course."""
    c = ledger.commitments[0]
    trailing = ledger.trailing_digression_turns()
    return trailing == 0 or trailing >= c.max_digression_turns
