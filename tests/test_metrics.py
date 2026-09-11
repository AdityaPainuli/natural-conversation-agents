"""Metrics on a tiny fixture, with the expected values worked out by hand.

Fixture: two objectives, three participant turns.
  p1 answers 'x' and implies 'y', and uses the phrase "pancakes".
  p2 uses the phrase "mondays".
  p3 says nothing new.
"""

from dataclasses import dataclass, field

from demo.harness import run_agent
from src.agents import AgentTurn
from src.participant import Annotations, ScriptedParticipant, ScriptedTurn

SCRIPT = [
    ScriptedTurn("p1", "I order at night, and never pancakes.",
                 Annotations(answers_objectives=("x",), implies_objectives=("y",), topics=("pancakes",))),
    ScriptedTurn("p2", "Mostly mondays.", Annotations(topics=("mondays",))),
    ScriptedTurn("p3", "That's it.", Annotations()),
]
PARTICIPANT = ScriptedParticipant(SCRIPT)


@dataclass
class FakeAgent:
    """Replays a fixed list of agent turns, so the metric code is what's under test."""

    turns: list[AgentTurn]
    name: str = "fake"
    seen: list[ScriptedTurn] = field(default_factory=list)

    def observe(self, turn: ScriptedTurn) -> None:
        self.seen.append(turn)

    def respond(self) -> AgentTurn:
        return self.turns[len(self.seen)]


def test_forgetful_agent_scores_as_hand_checked():
    run = run_agent(
        FakeAgent([
            AgentTurn("a1", "When do you order?", "ASK", "x"),
            AgentTurn("a2", "And do you ever order pancakes?", "ASK", "y"),
            AgentTurn("a3", "Anything else?", "ASK", None),
        ]),
        PARTICIPANT,
    )
    # a2 asks 'y' after p1 already implied it -> 1 repeat out of 2 objective questions
    assert run.counts["repeat_question_rate"] == [1, 2]
    # 'y' was implied at p1 and re-asked afterwards -> nothing captured
    assert run.counts["implicit_answer_capture_rate"] == [0, 1]
    # only a2 echoes an earlier phrase ("pancakes"), out of 3 agent turns
    assert run.counts["acknowledgment_rate"] == [1, 3]
    assert run.metrics["repeat_question_rate"] == 0.5
    assert run.metrics["implicit_answer_capture_rate"] == 0.0


def test_attentive_agent_scores_as_hand_checked():
    run = run_agent(
        FakeAgent([
            AgentTurn("a1", "When do you order?", "ASK", "x"),
            AgentTurn("a2", "You mentioned pancakes, say more.", "DEEPEN", None),
            AgentTurn("a3", "Anything else?", "ASK", None),
        ]),
        PARTICIPANT,
    )
    assert run.counts["repeat_question_rate"] == [0, 1]
    assert run.counts["implicit_answer_capture_rate"] == [1, 1]
    assert run.counts["acknowledgment_rate"] == [1, 3]
    assert run.metrics["implicit_answer_capture_rate"] == 1.0


def test_an_agent_turn_before_any_participant_turn_cannot_acknowledge():
    run = run_agent(
        FakeAgent([
            AgentTurn("a1", "Tell me about pancakes.", "ASK", "x"),
            AgentTurn("a2", "Right.", "ASK", None),
            AgentTurn("a3", "Right.", "ASK", None),
        ]),
        PARTICIPANT,
    )
    assert run.counts["acknowledgment_rate"] == [0, 3]


def test_coverage_efficiency_counts_questions_and_covered_objectives():
    run = run_agent(
        FakeAgent([
            AgentTurn("a1", "When do you order?", "ASK", "x"),
            AgentTurn("a2", "You mentioned pancakes, say more.", "DEEPEN", None),
            AgentTurn("a3", "Anything else?", "ASK", None),
        ]),
        PARTICIPANT,
    )
    # one objective question; p1 covered both 'x' (answered) and 'y' (implied)
    assert run.coverage["questions_asked"] == 1
    assert run.coverage["objectives_covered"] == 2
    assert run.coverage["questions_per_objective"] == 0.5
