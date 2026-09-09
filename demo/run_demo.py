"""Side-by-side terminal demo. Naive agent left, ledger agent right, the same
participant down the middle.

    python demo/run_demo.py --mode stub                  # no API key needed
    python demo/run_demo.py --mode stub --pause-at-moment # the live-talk flow
    python demo/run_demo.py --mode live                   # needs ANTHROPIC_API_KEY

Either invocation form works: `python demo/run_demo.py` or `python -m demo.run_demo`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as `python demo/x.py` or `python -m demo.x`

import argparse

from demo.harness import Round, Run, run_both
from demo.scenario import PARTICIPANT

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text

    RICH = True
except ImportError:  # rich is optional; the demo degrades to plain print
    RICH = False


def _callouts(naive: Round, ledger: Round) -> list[str]:
    out = []
    if naive.repeat:
        out.append(f"[GOLDFISH: re-asked objective '{naive.agent_turn.asks_objective}']")
    if naive.deaf:
        out.append("[DEAF INTERVIEWER: ignored what the participant just volunteered]")
    if ledger.agent_turn.move not in ("ASK_OBJECTIVE",):
        out.append(f"[POLICY: {ledger.agent_turn.move} - {ledger.agent_turn.reason}]")
    return out


def render_rich(naive: Run, ledger: Run, pause_at_moment: bool) -> None:
    console = Console()
    half = max(34, (console.width - 4) // 2)
    pause_index = PARTICIPANT.pause_index
    console.print(Rule("naive question loop  vs  ledger + policy"))

    for i, (n, l) in enumerate(zip(naive.rounds, ledger.rounds)):
        console.print(
            Columns(
                [
                    Panel(n.agent_turn.text, title=f"naive {n.agent_turn.id}",
                          border_style="red" if n.repeat else "grey42", width=half),
                    Panel(l.agent_turn.text, title=f"ledger {l.agent_turn.id}",
                          border_style="green", width=half),
                ]
            )
        )
        for note in _callouts(n, l):
            failure = note.startswith("[GOLDFISH") or note.startswith("[DEAF")
            console.print(Text("  " + note, style="bold yellow" if failure else "cyan"))
        console.print(Panel(n.participant_turn.text, title=f"participant {n.participant_turn.id}",
                            border_style="magenta" if n.participant_turn.pause_moment else "blue"))
        if pause_at_moment and i == pause_index:
            console.print(Rule("[bold magenta]pause moment - press Enter for what each agent said next"))
            input()

    console.print(Rule("metrics"))
    for key in naive.counts:
        n_num, n_den = naive.counts[key]
        l_num, l_den = ledger.counts[key]
        console.print(
            f"{key:<30} naive {naive.metrics[key]:6.1%} ({n_num}/{n_den})   "
            f"ledger {ledger.metrics[key]:6.1%} ({l_num}/{l_den})"
        )


def render_plain(naive: Run, ledger: Run) -> None:
    for n, l in zip(naive.rounds, ledger.rounds):
        print(f"\nNAIVE  {n.agent_turn.id}: {n.agent_turn.text}")
        print(f"LEDGER {l.agent_turn.id}: {l.agent_turn.text}")
        for note in _callouts(n, l):
            print("       " + note)
        print(f"PART.  {n.participant_turn.id}: {n.participant_turn.text}")
    print()
    for key in naive.counts:
        print(f"{key:<30} naive {naive.metrics[key]:6.1%}   ledger {ledger.metrics[key]:6.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["stub", "live"], default="stub")
    ap.add_argument("--pause-at-moment", action="store_true",
                    help="stop at the pause moment and wait for a keypress")
    args = ap.parse_args()

    naive, ledger = run_both(args.mode)
    if RICH:
        render_rich(naive, ledger, args.pause_at_moment)
    else:
        render_plain(naive, ledger)


if __name__ == "__main__":
    main()
