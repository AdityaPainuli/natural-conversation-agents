"""The scripted difficult participant.

The participant does not react to the agent. Both agents face the identical
ten turns, in the identical order, so any difference in the transcript comes
from the agent architecture and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Literal


@dataclass(frozen=True)
class Annotations:
    """Ground truth for one participant turn.

    Powers stub-mode extraction *and* the metrics. Same numbers either way:
    metrics never read the ledger, only these annotations and the transcript.
    """

    answers_objectives: tuple[str, ...] = ()
    implies_objectives: tuple[str, ...] = ()
    preferences: tuple[tuple[Literal["likes", "dislikes"], str], ...] = ()
    topics: tuple[str, ...] = ()
    summary: str = ""
    digression: bool = False
    digression_value: Literal["valuable", "noise"] | None = None
    ambiguous_objective: str | None = None
    register: Literal["standard", "brief"] = "standard"


@dataclass(frozen=True)
class ScriptedTurn:
    id: str
    text: str
    annotations: Annotations = field(default_factory=Annotations)
    pause_moment: bool = False


@dataclass
class ScriptedParticipant:
    script: list[ScriptedTurn]

    def __iter__(self) -> Iterator[ScriptedTurn]:
        return iter(self.script)

    def __len__(self) -> int:
        return len(self.script)

    def turn(self, index: int) -> ScriptedTurn:
        return self.script[index]

    @property
    def pause_index(self) -> int:
        for i, t in enumerate(self.script):
            if t.pause_moment:
                return i
        raise ValueError("script has no pause moment")
