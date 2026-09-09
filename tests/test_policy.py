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
