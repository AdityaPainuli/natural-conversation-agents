"""The scenario: a market-research interview about food delivery apps.

Six objectives, one participant, ten turns. The participant is difficult in
the way real participants are difficult: they answer things you have not asked
yet, they wander somewhere more interesting than your script, they hedge, and
by the end they are giving you four words at a time.
"""

from __future__ import annotations

from src.ledger import Objective
from src.participant import Annotations, ScriptedParticipant, ScriptedTurn

OBJECTIVES: list[Objective] = [
    Objective("frequency", "How often do you order food delivery?", "how often you order"),
    Objective("app_choice", "Which delivery app do you use most, and why that one?", "which app you use"),
    Objective("frustration", "What frustrates you most when you're using it?", "what frustrates you"),
    Objective("discovery", "How do you decide what to order?", "how you decide what to order"),
    Objective("occasions", "Which meals do you order for, and at what times of day?", "which meals you order for"),
    Objective("failure", "What happens when an order goes wrong?", "what happens when an order goes wrong"),
]

SCRIPT: list[ScriptedTurn] = [
    ScriptedTurn(
        id="p1",
        text="Two or three times a week. More when work gets busy and I can't be bothered to cook.",
        annotations=Annotations(
            answers_objectives=("frequency",),
            topics=("two or three times a week", "when work gets busy"),
            summary="ordering two or three times a week",
        ),
    ),
    ScriptedTurn(
        id="p2",
        text=(
            "Mostly one app. I've had it for years. Honestly it's habit plus the coupons, "
            "switching over to another one feels like effort I don't want to spend."
        ),
        annotations=Annotations(
            answers_objectives=("app_choice",),
            topics=("habit", "coupons", "feels like effort"),
            summary="staying on one app out of habit",
        ),
    ),
    # --- the pause moment (slides 14-15) ---------------------------------
    ScriptedTurn(
        id="p3",
        text=(
            "The delivery estimates. It says twenty-five minutes and then it's fifty, and there's "
            "no update in between. I only ever order dinner anyway, or lunch on weekends, so a late "
            "dinner is the whole evening gone. And honestly I never order breakfast. I hate morning "
            "notifications. The app pings me at 8am about pancakes and I want to throw my phone."
        ),
        annotations=Annotations(
            answers_objectives=("frustration",),
            implies_objectives=("occasions",),
            preferences=(("dislikes", "morning notifications"),),
            topics=("delivery estimates", "a late dinner", "lunch on weekends", "breakfast", "morning notifications"),
            summary="the delivery estimates being wrong",
        ),
        pause_moment=True,
    ),
    # ----------------------------------------------------------------------
    ScriptedTurn(
        id="p4",
        text=(
            "Yeah, the notifications are the worst part. I turned them off once, and then I stopped "
            "getting the actual delivery updates too, so now they're back on and I just live with it."
        ),
        annotations=Annotations(
            topics=("notifications", "turned them off", "delivery updates"),
            summary="turning notifications off and regretting it",
        ),
    ),
    ScriptedTurn(
        id="p5",
        text=(
            "Which, actually, reminds me. Last month an order just never arrived. The driver marked "
            "it delivered and there was nothing at the door. I spent forty minutes in the chat support "
            "thing and it kept sending me help articles. Eventually I got a refund, but no apology and "
            "no explanation."
        ),
        annotations=Annotations(
            implies_objectives=("failure",),
            topics=("never arrived", "chat support", "help articles", "a refund"),
            summary="the order that never arrived",
            digression=True,
            digression_value="valuable",
        ),
    ),
    ScriptedTurn(
        id="p6",
        text=(
            "The part that actually got me was the driver photo. It was somebody else's door. And "
            "nobody at the app seemed to think that was worth looking into."
        ),
        annotations=Annotations(
            topics=("the driver photo", "somebody else's door"),
            summary="the driver photo showing somebody else's door",
            digression=True,
            digression_value="valuable",
        ),
    ),
    ScriptedTurn(
        id="p7",
        text="I mean, it depends. Sometimes I know what I want, sometimes I just scroll. Kind of both, really.",
        annotations=Annotations(
            topics=("just scroll", "kind of both"),
            summary="sometimes knowing what you want and sometimes just scrolling",
            ambiguous_objective="discovery",
        ),
    ),
    ScriptedTurn(
        id="p8",
        text=(
            "If it's a weeknight it's the same five places and I don't even look. Weekends I'll scroll "
            "and read ratings, but I basically never go past the first screen."
        ),
        annotations=Annotations(
            answers_objectives=("discovery",),
            topics=("the same five places", "read ratings", "the first screen"),
            summary="defaulting to the same five places on weeknights",
        ),
    ),
    ScriptedTurn(
        id="p9",
        text="Yeah. Pretty much it.",
        annotations=Annotations(summary="agreeing", register="brief"),
    ),
    ScriptedTurn(
        id="p10",
        text="Nope, that's everything. Good luck with the app.",
        annotations=Annotations(summary="wrapping up", register="brief"),
    ),
]

PARTICIPANT = ScriptedParticipant(SCRIPT)
