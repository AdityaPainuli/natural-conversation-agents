"""Per-turn extraction: turn text -> structured facts the ledger can merge.

Live: one LLM call with a forced JSON schema.
Stub: read the ground-truth annotations off the scripted turn.

Both paths return the same dataclass, so nothing downstream knows the difference.
"""

from __future__ import annotations

from typing import Any

from .ledger import Ledger, Objective, Preference, Turn, TurnExtraction
from .llm import LLMClient
from .participant import ScriptedTurn

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answers_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objective ids the turn answers directly, in response to what was asked.",
        },
        "implies_objectives": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Objective ids the turn answers incidentally, while talking about something else.",
        },
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "polarity": {"type": "string", "enum": ["likes", "dislikes"]},
                    "topic": {"type": "string"},
                },
                "required": ["polarity", "topic"],
            },
        },
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Short verbatim phrases from the turn that a human would call back to later.",
        },
        "summary": {"type": "string", "description": "The turn in one noun phrase, e.g. 'the order that never arrived'."},
        "digression": {"type": "boolean"},
        "digression_value": {"type": "string", "enum": ["valuable", "noise"]},
        "ambiguous_objective": {
            "type": ["string", "null"],
            "description": "Objective id the turn gestures at without actually answering.",
        },
    },
    "required": ["answers_objectives", "implies_objectives", "preferences", "topics", "summary", "digression"],
}

SYSTEM = """You are the extraction layer of an interview agent.
Given the participant's latest turn, report what it covered.

Rules:
- answers_objectives: what they answered because it was asked.
- implies_objectives: what they answered while talking about something else. This is the
  one that matters. Be generous here: if a human interviewer would not re-ask it, list it.
- Only use objective ids from the list you are given.
- topics: short phrases lifted verbatim, the ones a good interviewer would echo back.
- digression: true if the turn wanders off the objective. valuable if it is the kind of
  wandering that produces the best material, noise if it is genuinely off-task."""


def extract(
    turn: ScriptedTurn | str,
    objectives: list[Objective],
    ledger: Ledger,
    client: LLMClient | None = None,
    turn_id: str | None = None,
) -> TurnExtraction:
    client = client or LLMClient.from_env()
    text = turn.text if isinstance(turn, ScriptedTurn) else turn
    tid = turn_id or (turn.id if isinstance(turn, ScriptedTurn) else f"p{len(ledger.extractions) + 1}")
    record = Turn(id=tid, speaker="participant", text=text)

    if not client.is_live:
        if not isinstance(turn, ScriptedTurn):
            raise ValueError("stub-mode extraction needs an annotated ScriptedTurn")
        return _from_annotations(record, turn)
    return _from_llm(record, objectives, ledger, client)


def _from_annotations(record: Turn, turn: ScriptedTurn) -> TurnExtraction:
    a = turn.annotations
    return TurnExtraction(
        turn=record,
        answers_objectives=list(a.answers_objectives),
        implies_objectives=list(a.implies_objectives),
        preferences=[Preference(polarity=p, topic=t, turn_id=record.id) for p, t in a.preferences],
        topics=list(a.topics),
        summary=a.summary,
        digression=a.digression,
        digression_value=a.digression_value,
        ambiguous_objective=a.ambiguous_objective,
    )


def _from_llm(record: Turn, objectives: list[Objective], ledger: Ledger, client: LLMClient) -> TurnExtraction:
    catalogue = "\n".join(f"- {o.id}: {o.question}" for o in objectives)
    covered = "\n".join(ledger.summary_lines())
    user = (
        f"Objectives:\n{catalogue}\n\n"
        f"Ledger so far:\n{covered or '(empty)'}\n\n"
        f"Participant turn:\n{record.text}"
    )
    raw = client.complete_json(SYSTEM, user, EXTRACTION_SCHEMA)
    valid = {o.id for o in objectives}
    return TurnExtraction(
        turn=record,
        answers_objectives=[o for o in raw.get("answers_objectives", []) if o in valid],
        implies_objectives=[o for o in raw.get("implies_objectives", []) if o in valid],
        preferences=[
            Preference(polarity=p["polarity"], topic=p["topic"], turn_id=record.id)
            for p in raw.get("preferences", [])
        ],
        topics=list(raw.get("topics", [])),
        summary=raw.get("summary", ""),
        digression=bool(raw.get("digression", False)),
        digression_value=raw.get("digression_value") if raw.get("digression") else None,
        ambiguous_objective=raw.get("ambiguous_objective") if raw.get("ambiguous_objective") in valid else None,
    )
