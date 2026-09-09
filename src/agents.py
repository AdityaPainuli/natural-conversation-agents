"""The two agents.

They share a model, a temperature, a scenario, and full conversation history.
The only difference is what sits between "read the turn" and "write the reply":
the naive agent has an index, the ledger agent has a ledger and a policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .delivery import cap_length, detect_register, match_register
from .extraction import extract
from .ledger import Ledger, Objective, Turn
from .llm import LLMClient
from .participant import ScriptedTurn
from .policy import Decision, Move, next_move

WRAP_UP = "Is there anything else you'd like to add?"

SYSTEM_NAIVE = """You are conducting a research interview. You have a fixed list of questions.
Ask the next one on the list. Keep it to two sentences."""

SYSTEM_LEDGER = """You are conducting a research interview.

You are given a ledger of what the participant has already covered and the single
move you have decided to make this turn. Perform that move and nothing else.

Always acknowledge, by name, the specific thing the participant said that you are
building on. Never re-ask something the ledger marks answered or implicitly_answered.
Two or three sentences, plain spoken, no interviewer boilerplate."""


@dataclass
class AgentTurn:
    id: str
    text: str
    move: str
    asks_objective: str | None = None
    reason: str = ""


@dataclass
class NaiveAgent:
    """Fair baseline: same model, same history, no memory model.

    It fails because of its architecture, not because we made it stupid.
    """

    objectives: list[Objective]
    client: LLMClient = field(default_factory=LLMClient.from_env)
    index: int = 0
    history: list[Turn] = field(default_factory=list)
    name: str = "naive"

    def observe(self, turn: ScriptedTurn) -> None:
        self.history.append(Turn(id=turn.id, speaker="participant", text=turn.text))

    def respond(self) -> AgentTurn:
        tid = f"a{sum(1 for t in self.history if t.speaker == 'agent') + 1}"
        objective = self.objectives[self.index] if self.index < len(self.objectives) else None
        self.index += 1
        text = self._generate(objective)
        self.history.append(Turn(id=tid, speaker="agent", text=text))
        return AgentTurn(tid, text, "ASK_NEXT_IN_LIST", objective.id if objective else None)

    def _generate(self, objective: Objective | None) -> str:
        question = objective.question if objective else WRAP_UP
        if not self.client.is_live:
            return f"Got it, thanks. {question}"
        return self.client.complete_text(SYSTEM_NAIVE, f"{_render(self.history)}\n\nNext question to ask: {question}")


@dataclass
class LedgerAgent:
    """extraction -> ledger.update -> next_move -> generation."""

    objectives: list[Objective]
    client: LLMClient = field(default_factory=LLMClient.from_env)
    ledger: Ledger = field(init=False)
    decisions: list[Decision] = field(default_factory=list)
    register: str = "standard"
    name: str = "ledger"

    def __post_init__(self) -> None:
        self.ledger = Ledger.from_objectives(self.objectives)

    def observe(self, turn: ScriptedTurn) -> None:
        self.ledger.update(extract(turn, self.objectives, self.ledger, self.client))
        self.register = detect_register(turn.text)

    def respond(self) -> AgentTurn:
        tid = f"a{sum(1 for t in self.ledger.history if t.speaker == 'agent') + 1}"
        decision = next_move(self.ledger)
        self.decisions.append(decision)
        text = match_register(cap_length(self._generate(decision)), self.register)
        self.ledger.record_agent_turn(Turn(id=tid, speaker="agent", text=text))
        asks = decision.objective_id if decision.move in _QUESTION_MOVES else None
        return AgentTurn(tid, text, decision.move.name, asks, decision.reason)

    # --- generation -------------------------------------------------------
    def _generate(self, decision: Decision) -> str:
        if self.client.is_live:
            return self.client.complete_text(SYSTEM_LEDGER, self._live_prompt(decision))
        return _template(decision, self.ledger, self.objectives)

    def _live_prompt(self, decision: Decision) -> str:
        target = _objective(self.objectives, decision.objective_id)
        return (
            f"{_render(self.ledger.history)}\n\n"
            f"Ledger:\n" + "\n".join(self.ledger.summary_lines()) + "\n\n"
            f"Move: {decision.move.name} ({decision.reason})\n"
            + (f"Objective in play: {target.question}\n" if target else "")
            + (f"Objectives to skip, say so out loud: {', '.join(decision.skipped)}\n" if decision.skipped else "")
        )


_QUESTION_MOVES = (Move.ASK_OBJECTIVE, Move.CLARIFY, Move.RETURN_TO_OBJECTIVE)


def _objective(objectives: list[Objective], oid: str | None) -> Objective | None:
    return next((o for o in objectives if o.id == oid), None)


def _render(history: list[Turn]) -> str:
    return "\n".join(f"{t.speaker}: {t.text}" for t in history) or "(no turns yet)"


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _template(decision: Decision, ledger: Ledger, objectives: list[Objective]) -> str:
    """Stub-mode generation. The phrasing is fixed so the transcript is stable;
    every fact in it is pulled from the ledger, not from the script."""
    last = ledger.last_extraction()
    target = _objective(objectives, decision.objective_id)

    if decision.move is Move.ACKNOWLEDGE_AND_DEEPEN:
        pref = ledger.preferences[-1]
        verb = "clearly can't stand" if pref.polarity == "dislikes" else "clearly like"
        implied = _join([_objective(objectives, o).label for o in last.implies_objectives])
        return (
            f"Hold on, two things there. You just told me {implied} without me asking, "
            f"and you {verb} {pref.topic}. What would have to change about {pref.topic} "
            f"for it to be worth opening?"
        )

    if decision.move is Move.ALLOW_DIGRESSION:
        return f"No, go on. {last.summary.capitalize()} is exactly the kind of thing I want to hear. What happened next?"

    if decision.move is Move.RETURN_TO_OBJECTIVE:
        return (
            f"{last.summary.capitalize()} is the part I'll be quoting back. "
            f"Coming back to where we were, though: {target.question}"
        )

    if decision.move is Move.CLARIFY:
        return f"When you say {last.summary}, which is it most of the time? Take the last two orders you placed."

    if decision.move is Move.SKIP_ANSWERED:
        labels = _join([_objective(objectives, o).label for o in decision.skipped])
        evidence = _join([_evidence_summary(ledger, o) for o in decision.skipped])
        tail = f" So, last one: {target.question}" if target else " So I won't make you repeat yourself."
        return f"I still had {labels} on my list, but you covered them already, back when you were talking about {evidence}.{tail}"

    if decision.move is Move.CLOSE_TOPIC:
        return (
            "That's everything I had on my list, and you gave me more than I came for. "
            "Thanks for the time. I'll stop there."
        )

    if last is None:
        return target.question
    return f"Makes sense, {last.summary}. {target.question}"


def _evidence_summary(ledger: Ledger, oid: str) -> str:
    turn_id = ledger.state(oid).evidence[-1]
    ex = next(e for e in ledger.extractions if e.turn.id == turn_id)
    return ex.summary
