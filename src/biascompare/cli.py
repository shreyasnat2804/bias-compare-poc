"""Command-line entry point. See docs/PLAN.md §7.

    biascompare run CONFIG       score every model, write the results JSON
    biascompare preview CONFIG   build the perturbations only, report flags

`run` writes to the config's `output` path, or to `results/<config name>.json`
next to the config file when none is set.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from biascompare.compare import preview_perturbations
from biascompare.config import load_config, run
from biascompare.results import save_results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="biascompare", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("run", "score every model in the config and write the results JSON"),
        ("preview", "build the perturbation plan without loading any model under test"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", help="path to a .json, .yaml or .yml run config")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ValueError, RuntimeError) as exc:
        print(f"biascompare: {exc}", file=sys.stderr)
        return 2

    if args.command == "preview":
        return _preview(config)

    results = run(config)
    output = config.output or Path(args.config).parent / "results" / f"{Path(args.config).stem}.json"
    save_results(results, output)
    print(results["summary_verdict"])
    for warning in results["warnings"]:
        print(f"warning: {warning}")
    print(f"results written to {output}")
    return 0


def _preview(config) -> int:
    rows = preview_perturbations(
        config.dataset, n_controls=config.n_controls, seed=config.seed, axis=config.axis,
        controls=config.controls,
    )
    flags = Counter(flag for row in rows for flag in row["flags"])
    print(f"{len(rows)} rows, controls: {config.controls.method}")
    for flag, count in sorted(flags.items()):
        print(f"  {flag}: {count}")
    for row in rows[:3]:
        print(f"\n[{row['id']}] {row['original']}\n  swap: {row['gender_swap']['text']}")
        for control in row["controls"][:2]:
            print(f"  control: {control['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
