"""One test per guard clause, on hand-built ledger states."""

from src.ledger import Commitment, Coverage, Ledger, Objective, Preference, Turn, TurnExtraction
from src.policy import Move, next_move

OBJECTIVES = [Objective("a", "A?", "a"), Objective("b", "B?", "b"), Objective("c", "C?", "c")]


def ledger() -> Ledger:
    return Ledger.from_objectives(OBJECTIVES)


def ex(turn_id: str, **kwargs) -> TurnExtraction:
    return TurnExtraction(turn=Turn(turn_id, "participant", "..."), **kwargs)


def test_guard_1_due_commitment_returns_to_objective():
    led = ledger()
    led.update(ex("p1"))  # digression is over
    led.commitments.append(Commitment("b", "digression", "p0"))
    d = next_move(led)
    assert d.move is Move.RETURN_TO_OBJECTIVE and d.objective_id == "b"
    assert not led.commitments


def test_guard_1_commitment_not_due_while_digression_still_running():
    led = ledger()
    led.mark_asked("a")
    led.update(ex("p1", digression=True, digression_value="valuable"))
    led.commitments.append(Commitment("a", "digression", "p0", max_digression_turns=2))
    assert next_move(led).move is Move.ALLOW_DIGRESSION


def test_guard_2_valuable_digression_schedules_a_return():
    led = ledger()
    led.mark_asked("a")
    led.update(ex("p1", digression=True, digression_value="valuable"))
    d = next_move(led)
    assert d.move is Move.ALLOW_DIGRESSION
    assert [c.objective_id for c in led.commitments] == ["a"]


def test_guard_2_noise_digression_is_not_indulged():
    led = ledger()
    led.update(ex("p1", digression=True, digression_value="noise"))
    assert next_move(led).move is Move.ASK_OBJECTIVE
    assert not led.commitments


def test_guard_3_ambiguous_turn_clarifies():
    led = ledger()
    led.update(ex("p1", ambiguous_objective="a"))
    d = next_move(led)
    assert d.move is Move.CLARIFY and d.objective_id == "a"


def test_guard_4_skips_an_objective_answered_sideways():
    led = ledger()
    led.update(ex("p1", implies_objectives=["a"]))
    d = next_move(led)
    assert d.move is Move.SKIP_ANSWERED
    assert d.skipped == ["a"] and d.objective_id == "b"
    assert led.state("a").coverage is Coverage.IMPLICITLY_ANSWERED


def test_guard_5_acknowledges_a_volunteered_preference_once():
    led = ledger()
    led.update(ex("p1", preferences=[Preference("dislikes", "morning notifications", "p1")]))
    assert next_move(led).move is Move.ACKNOWLEDGE_AND_DEEPEN
    assert next_move(led).move is Move.ASK_OBJECTIVE  # not twice for the same turn


def test_guard_6_default_asks_the_next_objective_and_advances():
    led = ledger()
    d = next_move(led)
    assert d.move is Move.ASK_OBJECTIVE and d.objective_id == "a"
    assert led.state("a").asked and led.cursor == 1
    assert next_move(led).objective_id == "b"


def test_close_topic_when_nothing_is_left():
    led = ledger()
    for oid in ("a", "b", "c"):
        led.mark_asked(oid)
    led.update(ex("p1", answers_objectives=["a", "b", "c"]))
    assert next_move(led).move is Move.CLOSE_TOPIC


# --- acknowledgment composes with the move, it does not compete with it -------

def notable_turn(turn_id: str = "p1") -> TurnExtraction:
    return ex(
        turn_id,
        implies_objectives=["a"],
        preferences=[Preference("dislikes", "morning notifications", turn_id)],
    )


def test_skip_answered_after_a_preference_bearing_turn_still_acknowledges():
    led = ledger()
    led.update(notable_turn())
    d = next_move(led)
    assert d.move is Move.SKIP_ANSWERED          # the guard order is untouched
    assert d.skipped == ["a"]
    assert [(r.kind, r.text) for r in d.acknowledge] == [
        ("implicit_answer", "a"),
        ("preference", "can't stand morning notifications"),
    ]


def test_every_guard_carries_the_acknowledgment_when_there_is_one():
    # digression that also volunteers something: ALLOW_DIGRESSION, still acknowledges
    led = ledger()
    led.mark_asked("b")
    led.update(ex(
        "p1",
        implies_objectives=["a"],
        preferences=[Preference("likes", "coupons", "p1")],
        digression=True,
        digression_value="valuable",
    ))
    d = next_move(led)
    assert d.move is Move.ALLOW_DIGRESSION
    assert [r.text for r in d.acknowledge] == ["a", "like coupons"]


def test_no_acknowledgment_when_nothing_was_volunteered():
    led = ledger()
    led.update(ex("p1", answers_objectives=["a"]))
    assert next_move(led).acknowledge == []


def test_acknowledge_and_deepen_still_fires_when_no_earlier_guard_preempts():
    led = ledger()
    led.update(ex("p1", preferences=[Preference("dislikes", "morning notifications", "p1")]))
    d = next_move(led)
    assert d.move is Move.ACKNOWLEDGE_AND_DEEPEN
    assert d.acknowledge and d.acknowledge[0].kind == "preference"


# --- the pause moment must acknowledge, in both generation paths -------------

def pause_moment_agent_turn():
    """Replay the scenario up to the pause moment and return the agent's next
    decision plus the ledger that produced it."""
    from src.agents import LedgerAgent
    from src.llm import LLMClient
    from demo.scenario import OBJECTIVES, PARTICIPANT

    agent = LedgerAgent(OBJECTIVES, LLMClient(mode="stub"))
    for i in range(PARTICIPANT.pause_index + 1):
        agent.respond()
        agent.observe(PARTICIPANT.turn(i))
    return agent


def test_pause_moment_acknowledges_in_the_stub_path():
    from demo.scenario import PARTICIPANT

    agent = pause_moment_agent_turn()
    turn = agent.respond()
    scripted = PARTICIPANT.turn(PARTICIPANT.pause_index)
    _, preference_topic = scripted.annotations.preferences[0]
    assert preference_topic in turn.text
    assert "which meals you order for" in turn.text  # the objective answered sideways
    assert turn.text.index("Before I move on") < turn.text.index("What would have to change")


def test_pause_moment_acknowledges_in_the_live_path():
    """No network: assert the live prompt carries the acknowledgment instruction."""
    from src.policy import next_move

    agent = pause_moment_agent_turn()
    decision = next_move(agent.ledger)
    prompt = agent._live_prompt(decision)
    assert decision.acknowledge
    assert "Acknowledge first, in your own words:" in prompt
    assert "morning notifications" in prompt
    assert "which meals you order for" in prompt
