from src.ledger import Commitment, Coverage, Ledger, Objective, Preference, Turn, TurnExtraction

OBJECTIVES = [Objective("a", "A?", "a"), Objective("b", "B?", "b"), Objective("c", "C?", "c")]


def ledger() -> Ledger:
    return Ledger.from_objectives(OBJECTIVES)


def ex(turn_id: str, **kwargs) -> TurnExtraction:
    return TurnExtraction(turn=Turn(turn_id, "participant", "..."), **kwargs)


def test_explicit_answer_marks_answered_with_evidence():
    led = ledger()
    led.update(ex("p1", answers_objectives=["a"]))
    assert led.state("a").coverage is Coverage.ANSWERED
    assert led.state("a").evidence == ["p1"]


def test_implicit_transition():
    led = ledger()
    led.update(ex("p1", answers_objectives=["a"], implies_objectives=["c"]))
    assert led.state("c").coverage is Coverage.IMPLICITLY_ANSWERED
    assert led.state("c").evidence == ["p1"]
    assert led.state("b").coverage is Coverage.UNASKED


def test_implicit_does_not_downgrade_an_explicit_answer():
    led = ledger()
    led.update(ex("p1", answers_objectives=["a"]))
    led.update(ex("p2", implies_objectives=["a"]))
    assert led.state("a").coverage is Coverage.ANSWERED
    assert led.state("a").evidence == ["p1"]


def test_ambiguous_turn_marks_partial():
    led = ledger()
    led.update(ex("p1", ambiguous_objective="b"))
    assert led.state("b").coverage is Coverage.PARTIAL


def test_preferences_accumulate_in_order():
    led = ledger()
    led.update(ex("p1", preferences=[Preference("dislikes", "morning notifications", "p1")]))
    led.update(ex("p2", preferences=[Preference("likes", "coupons", "p2")]))
    assert [(p.polarity, p.topic) for p in led.preferences] == [
        ("dislikes", "morning notifications"),
        ("likes", "coupons"),
    ]
    assert [p.topic for p in led.preferences_for("likes")] == ["coupons"]


def test_commitments_pop_in_order():
    led = ledger()
    led.commitments.append(Commitment("a", "digression", "p1"))
    led.commitments.append(Commitment("b", "digression", "p2"))
    assert led.commitments.popleft().objective_id == "a"
    assert led.commitments.popleft().objective_id == "b"
    assert not led.commitments


def test_advance_cursor_reports_only_unasked_skips():
    led = ledger()
    led.mark_asked("a")
    led.update(ex("p1", answers_objectives=["a"], implies_objectives=["b"]))
    assert led.advance_cursor_past_covered() == ["b"]  # 'a' was asked, so skipping it is not news
    assert led.cursor_objective_id() == "c"


def test_trailing_digression_turns():
    led = ledger()
    led.update(ex("p1"))
    led.update(ex("p2", digression=True, digression_value="valuable"))
    led.update(ex("p3", digression=True, digression_value="valuable"))
    assert led.trailing_digression_turns() == 2


def test_pending_references_are_built_from_the_last_turn_only():
    from src.policy import pending_references

    led = ledger()
    led.update(ex("p1", preferences=[Preference("dislikes", "morning notifications", "p1")]))
    led.update(ex("p2", answers_objectives=["a"]))
    assert pending_references(led) == []  # p2 volunteered nothing
