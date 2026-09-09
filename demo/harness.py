"""Runs both agents through the same scenario, computes the metrics, writes
the artifacts the deck needs.

Nothing here is hardcoded. Every number comes from the produced transcript plus
the ground-truth annotations on the participant script.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as `python demo/x.py` or `python -m demo.x`

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from src.agents import AgentTurn, LedgerAgent, NaiveAgent
from src.llm import LLMClient
from src.participant import ScriptedParticipant, ScriptedTurn

from demo.scenario import OBJECTIVES, PARTICIPANT

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

COUNTING_RULES: dict[str, str] = {
    "repeat_question_rate": (
        "An agent turn counts as a repeat if it asks about an objective that the "
        "participant had already answered, explicitly or implicitly, in an earlier turn; "
        "the denominator is every agent turn that asks about a specific objective."
    ),
    "implicit_answer_capture_rate": (
        "An implicitly answered objective counts as captured if the agent never asked "
        "about it again after the turn that answered it sideways; the denominator is "
        "every objective the participant answered without being asked."
    ),
    "acknowledgment_rate": (
        "An agent turn counts as an acknowledgment if it repeats back, verbatim, a topic "
        "phrase the participant used in an earlier turn; the denominator is every agent turn."
    ),
}


class Agent(Protocol):
    name: str

    def observe(self, turn: ScriptedTurn) -> None: ...
    def respond(self) -> AgentTurn: ...


@dataclass
class Round:
    agent_turn: AgentTurn
    participant_turn: ScriptedTurn
    covered_before: list[str]  # objectives already covered per ground truth
    repeat: bool = False          # Goldfish: re-asked something already covered
    acknowledged: bool = False
    deaf: bool = False            # Deaf Interviewer: ignored a turn that volunteered something


@dataclass
class Run:
    name: str
    rounds: list[Round] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, list[int]] = field(default_factory=dict)


def run_agent(agent: Agent, participant: ScriptedParticipant) -> Run:
    run = Run(name=agent.name)
    covered: set[str] = set()
    seen_topics: list[str] = []
    volunteered_topics: list[str] = []  # topics from the previous turn, if it volunteered something
    for scripted in participant:
        turn = agent.respond()
        rnd = Round(turn, scripted, sorted(covered))
        rnd.repeat = turn.asks_objective in covered
        rnd.acknowledged = any(t.lower() in turn.text.lower() for t in seen_topics)
        rnd.deaf = bool(volunteered_topics) and not any(
            t.lower() in turn.text.lower() for t in volunteered_topics
        )
        run.rounds.append(rnd)
        agent.observe(scripted)
        covered |= set(scripted.annotations.answers_objectives) | set(scripted.annotations.implies_objectives)
        seen_topics.extend(scripted.annotations.topics)
        a = scripted.annotations
        volunteered_topics = list(a.topics) if (a.preferences or a.implies_objectives) else []
    _score(run, participant)
    return run


def _score(run: Run, participant: ScriptedParticipant) -> None:
    question_turns = [r for r in run.rounds if r.agent_turn.asks_objective]
    repeats = [r for r in question_turns if r.repeat]
    acks = [r for r in run.rounds if r.acknowledged]

    implicit: dict[str, int] = {}
    for i, scripted in enumerate(participant):
        for oid in scripted.annotations.implies_objectives:
            implicit.setdefault(oid, i)
    captured = [
        oid
        for oid, i in implicit.items()
        if not any(r.agent_turn.asks_objective == oid for r in run.rounds[i + 1 :])
    ]

    run.counts = {
        "repeat_question_rate": [len(repeats), len(question_turns)],
        "implicit_answer_capture_rate": [len(captured), len(implicit)],
        "acknowledgment_rate": [len(acks), len(run.rounds)],
    }
    run.metrics = {k: (n / d if d else 0.0) for k, (n, d) in run.counts.items()}


# --- artifacts -----------------------------------------------------------


def transcript_md(run: Run) -> str:
    lines = [f"# Transcript: {run.name} agent", ""]
    for r in run.rounds:
        flags = []
        if r.repeat:
            flags.append(f"GOLDFISH: re-asked `{r.agent_turn.asks_objective}`")
        if r.deaf:
            flags.append("DEAF INTERVIEWER: ignored what was just volunteered")
        if r.agent_turn.move not in ("ASK_NEXT_IN_LIST", "ASK_OBJECTIVE"):
            flags.append(r.agent_turn.move)
        tag = f"  _[{' | '.join(flags)}]_" if flags else ""
        lines += [f"**Agent ({r.agent_turn.id}):** {r.agent_turn.text}{tag}", ""]
        lines += [f"**Participant ({r.participant_turn.id}):** {r.participant_turn.text}", ""]
    return "\n".join(lines)


def pause_moment_md(naive: Run, ledger: Run, participant: ScriptedParticipant) -> str:
    i = participant.pause_index
    scripted = participant.turn(i)
    return "\n".join(
        [
            "# The pause moment",
            "",
            f"## Participant, turn {i + 1}",
            "",
            f"> {scripted.text}",
            "",
            "What just happened, per the annotations:",
            "",
            f"- answered: `{', '.join(scripted.annotations.answers_objectives)}`",
            f"- answered without being asked: `{', '.join(scripted.annotations.implies_objectives)}`",
            "- volunteered: "
            + ", ".join(f"`{pol} {topic}`" for pol, topic in scripted.annotations.preferences),
            "",
            "## Next turn, naive agent",
            "",
            f"> {naive.rounds[i + 1].agent_turn.text}",
            "",
            "## Next turn, ledger agent",
            "",
            f"> {ledger.rounds[i + 1].agent_turn.text}",
            "",
            f"_Move chosen: `{ledger.rounds[i + 1].agent_turn.move}` "
            f"({ledger.rounds[i + 1].agent_turn.reason})._",
            "",
        ]
    )


def metrics_payload(naive: Run, ledger: Run) -> dict:
    return {
        "counting_rules": COUNTING_RULES,
        "runs": {
            r.name: {
                "metrics": {k: round(v, 4) for k, v in r.metrics.items()},
                "counts": {k: {"numerator": n, "denominator": d} for k, (n, d) in r.counts.items()},
            }
            for r in (naive, ledger)
        },
    }


def write_chart(naive: Run, ledger: Run, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    keys = list(COUNTING_RULES)
    labels = ["repeat questions\n(lower is better)", "implicit answers captured\n(higher is better)",
              "acknowledgment\n(higher is better)"]
    bg, ink, muted = "#ffffff", "#1b1f27", "#9aa3af"
    accent = "#1f7a5e"  # the ledger agent; the naive agent stays deliberately neutral

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=bg)
    ax.set_facecolor(bg)
    x = range(len(keys))
    width = 0.34
    for offset, run, colour in ((-width / 2, naive, muted), (width / 2, ledger, accent)):
        vals = [run.metrics[k] * 100 for k in keys]
        bars = ax.bar([i + offset for i in x], vals, width, label=f"{run.name} agent", color=colour)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 2.5, f"{v:.0f}%", ha="center",
                    color=ink if colour == muted else accent, fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, color=ink, fontsize=10)
    ax.set_ylim(0, 132)
    ax.set_yticks([])
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, labelcolor=ink, loc="upper left", ncols=2, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=bg)
    plt.close(fig)
    return True


def write_artifacts(naive: Run, ledger: Run, participant: ScriptedParticipant, out: Path = ARTIFACTS) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for run in (naive, ledger):
        p = out / f"transcript_{run.name}.md"
        p.write_text(transcript_md(run))
        written.append(p)
    p = out / "pause_moment.md"
    p.write_text(pause_moment_md(naive, ledger, participant))
    written.append(p)
    p = out / "metrics.json"
    p.write_text(json.dumps(metrics_payload(naive, ledger), indent=2) + "\n")
    written.append(p)
    chart = out / "metrics_chart.png"
    if write_chart(naive, ledger, chart):
        written.append(chart)
    return written


def run_both(mode: str = "stub") -> tuple[Run, Run]:
    client = LLMClient.from_env(mode)
    naive = run_agent(NaiveAgent(OBJECTIVES, client), PARTICIPANT)
    ledger = run_agent(LedgerAgent(OBJECTIVES, client), PARTICIPANT)
    return naive, ledger


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run both agents and write the deck artifacts.")
    ap.add_argument("--mode", choices=["stub", "live"], default="stub")
    args = ap.parse_args()

    naive, ledger = run_both(args.mode)
    written = write_artifacts(naive, ledger, PARTICIPANT)

    print("Counting rules")
    for key, rule in COUNTING_RULES.items():
        print(f"  {key}: {rule}")
    print("\nResults")
    for run in (naive, ledger):
        for key, (n, d) in run.counts.items():
            print(f"  {run.name:<7} {key:<30} {run.metrics[key]:6.1%}  ({n}/{d})")
    print("\nArtifacts")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
