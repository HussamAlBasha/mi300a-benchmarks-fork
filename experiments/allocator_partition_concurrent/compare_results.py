#!/usr/bin/env python3
"""Compare the latest concurrent full-MI300A HIP STREAM runs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


MODES = ("spx", "tpx", "cpx")
EXPECTED_PROCESSES = {"spx": 1, "tpx": 3, "cpx": 6}
CONFIGURATIONS = (
    ("malloc", 1),
    ("hipMalloc", 0),
    ("hipMalloc", 1),
    ("hipHostMalloc", 0),
    ("hipHostMalloc", 1),
    ("hipMallocManaged", 0),
    ("hipMallocManaged", 1),
)
METRICS = ("copy_gb_s", "scale_gb_s", "add_gb_s", "triad_gb_s")
RUN_PATTERN = re.compile(r"^(spx|tpx|cpx)_paper_(.+)$")


def run_rank(path: Path) -> tuple[int, int]:
    suffix = path.name.rsplit("_", 1)[-1]
    return (int(suffix) if suffix.isdigit() else -1, path.stat().st_mtime_ns)


def latest_runs(runs_dir: Path) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    if not runs_dir.exists():
        return selected
    for path in runs_dir.iterdir():
        match = RUN_PATTERN.match(path.name) if path.is_dir() else None
        if not match:
            continue
        mode = match.group(1)
        if mode not in selected or run_rank(path) > run_rank(selected[mode]):
            selected[mode] = path
    return selected


def read_results(path: Path) -> dict[tuple[str, int], tuple[int, dict[str, float]]]:
    results: dict[tuple[str, int], tuple[int, dict[str, float]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["allocator"], int(row["xnack"]))
            values = {metric: float(row[metric]) for metric in METRICS}
            results[key] = (int(row["processes"]), values)
    return results


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def report(runs: dict[str, Path]) -> tuple[str, list[str]]:
    problems: list[str] = []
    selected_rows: list[list[str]] = []
    aggregate_rows: list[list[str]] = []

    for mode in MODES:
        run = runs.get(mode)
        selected_rows.append([
            mode.upper(), run.name if run else "—", str(EXPECTED_PROCESSES[mode])
        ])
        if run is None:
            problems.append(f"missing {mode} run")
            continue

        result_path = run / "comparison.tsv"
        try:
            results = read_results(result_path)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems.append(f"{result_path}: {exc}")
            continue

        for allocator, xnack in CONFIGURATIONS:
            result = results.get((allocator, xnack))
            if result is None:
                problems.append(f"{run.name}: missing {allocator} XNACK={xnack}")
                aggregate_rows.append([
                    mode.upper(), allocator, str(xnack),
                    str(EXPECTED_PROCESSES[mode]), "—", "—", "—", "—",
                ])
                continue
            processes, values = result
            if processes != EXPECTED_PROCESSES[mode]:
                problems.append(
                    f"{run.name}: {allocator} XNACK={xnack} has {processes} "
                    f"processes, expected {EXPECTED_PROCESSES[mode]}"
                )
            aggregate_rows.append([
                mode.upper(), allocator, str(xnack), str(processes),
                *[f"{values[metric]:.1f}" for metric in METRICS],
            ])

    output = "\n".join([
        "# Concurrent full-MI300A HIP STREAM comparison",
        "",
        "Values are sums across all concurrently running logical devices in decimal GB/s.",
        "",
        "## Selected runs",
        markdown_table(["Mode", "Run", "Processes"], selected_rows),
        "",
        "## Aggregate bandwidth",
        markdown_table(
            ["Mode", "Allocator", "XNACK", "Processes", "Copy", "Scale", "Add", "Triad"],
            aggregate_rows,
        ),
        "",
    ])
    return output, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path(__file__).resolve().parent / "runs"
    )
    parser.add_argument("--md", type=Path, help="write the Markdown report here")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    output, problems = report(latest_runs(args.runs))
    print(output)
    if args.md:
        args.md.write_text(output + "\n")
        print(f"Markdown report written to {args.md}")
    if problems:
        print("Problems:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
    return 1 if args.strict and problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
