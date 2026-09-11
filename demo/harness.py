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
        "An agent turn counts as an acknowledgment if it references a fact or preference the "
        "participant stated in an earlier turn, judged by verbatim topic-phrase match in stub "
        "mode and by an LLM judge in live mode; the denominator is every agent turn."
    ),
    "coverage_efficiency": (
        "Questions asked is every agent turn that asks about a specific objective, and "
        "objectives covered is every objective the participant answered by the end of the "
        "conversation; fewer questions for the same coverage is the point."
    ),
}

# the three rates that go on the chart; coverage_efficiency is a ratio, reported separately
METRIC_KEYS = ["repeat_question_rate", "implicit_answer_capture_rate", "acknowledgment_rate"]

ACK_JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "references_prior": {"type": "boolean"},
        "referenced_item": {"type": "string", "description": "The prior fact or preference referenced. Omit if none."},
    },
    "required": ["references_prior"],
}

ACK_JUDGE_SYSTEM = """You are grading one turn of an interview transcript.

Answer whether the agent's turn references a specific fact or preference the participant
stated in an EARLIER turn.

- Paraphrase counts. The agent does not have to use the participant's exact words.
- Generic politeness does not count: "got it", "thanks", "that's helpful", "understood".
- Restating or building on the participant's own content counts, including when the agent
  says it will skip a question because the participant already answered it."""


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
    acknowledged_item: str = ""   # what the live judge says was referenced
    deaf: bool = False            # Deaf Interviewer: ignored a turn that volunteered something


@dataclass
class Run:
    name: str
    rounds: list[Round] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, list[int]] = field(default_factory=dict)
    coverage: dict[str, float] = field(default_factory=dict)


def run_agent(agent: Agent, participant: ScriptedParticipant, client: LLMClient | None = None) -> Run:
    """Runs the agent and scores it. In live mode acknowledgment is LLM-judged,
    because string matching marks a good paraphrase as a miss."""
    run = Run(name=agent.name)
    judge = client if (client and client.is_live) else None
    prior_turns: list[str] = []
    covered: set[str] = set()
    seen_topics: list[str] = []
    volunteered_topics: list[str] = []  # topics from the previous turn, if it volunteered something
    for scripted in participant:
        turn = agent.respond()
        rnd = Round(turn, scripted, sorted(covered))
        rnd.repeat = turn.asks_objective in covered
        if judge:
            rnd.acknowledged, rnd.acknowledged_item = _judge_acknowledgment(judge, prior_turns, turn.text)
        else:
            rnd.acknowledged = any(t.lower() in turn.text.lower() for t in seen_topics)
        rnd.deaf = bool(volunteered_topics) and not any(
            t.lower() in turn.text.lower() for t in volunteered_topics
        )
        run.rounds.append(rnd)
        agent.observe(scripted)
        covered |= set(scripted.annotations.answers_objectives) | set(scripted.annotations.implies_objectives)
        seen_topics.extend(scripted.annotations.topics)
        prior_turns.append(scripted.text)
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

    covered: set[str] = set()
    for scripted in participant:
        covered |= set(scripted.annotations.answers_objectives) | set(scripted.annotations.implies_objectives)
    run.coverage = {
        "questions_asked": len(question_turns),
        "objectives_covered": len(covered),
        "questions_per_objective": round(len(question_turns) / len(covered), 3) if covered else 0.0,
    }


# --- artifacts -----------------------------------------------------------


def _judge_acknowledgment(client: LLMClient, prior_turns: list[str], agent_text: str) -> tuple[bool, str]:
    if not prior_turns:
        return False, ""
    earlier = "\n".join(f"participant: {t}" for t in prior_turns)
    raw = client.complete_json(
        ACK_JUDGE_SYSTEM,
        f"Earlier participant turns:\n{earlier}\n\nAgent turn to grade:\n{agent_text}",
        ACK_JUDGE_SCHEMA,
    )
    return bool(raw.get("references_prior", False)), str(raw.get("referenced_item", ""))


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
                "coverage": r.coverage,
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

    keys = METRIC_KEYS
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
    naive = run_agent(NaiveAgent(OBJECTIVES, client), PARTICIPANT, client)
    ledger = run_agent(LedgerAgent(OBJECTIVES, client), PARTICIPANT, client)
    return naive, ledger


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Run both agents and write the deck artifacts.")
    ap.add_argument("--mode", choices=["stub", "live"], default="stub")
    ap.add_argument("--out", default=None, help="output directory (default: artifacts/)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else ARTIFACTS
    naive, ledger = run_both(args.mode)
    written = write_artifacts(naive, ledger, PARTICIPANT, out)

    print("Counting rules")
    for key, rule in COUNTING_RULES.items():
        print(f"  {key}: {rule}")
    print("\nResults")
    for run in (naive, ledger):
        for key, (n, d) in run.counts.items():
            print(f"  {run.name:<7} {key:<30} {run.metrics[key]:6.1%}  ({n}/{d})")
        c = run.coverage
        print(f"  {run.name:<7} {'coverage_efficiency':<30} {c['questions_per_objective']:6.2f}  "
              f"({c['questions_asked']} questions asked / {c['objectives_covered']} objectives covered)")
    print("\nArtifacts")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
