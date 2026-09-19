#!/usr/bin/env python3
"""Compare the latest single-logical-partition HIP STREAM runs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


MODES = ("spx", "tpx", "cpx")
EXPECTED_CUS = {"spx": 228, "tpx": 76, "cpx": 38}
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


def read_results(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    results: dict[tuple[str, int], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["allocator"], int(row["xnack"]))
            results[key] = {metric: float(row[metric]) for metric in METRICS}
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
    bandwidth_rows: list[list[str]] = []

    for mode in MODES:
        run = runs.get(mode)
        selected_rows.append([
            mode.upper(), run.name if run else "—", str(EXPECTED_CUS[mode])
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
            values = results.get((allocator, xnack))
            if values is None:
                problems.append(f"{run.name}: missing {allocator} XNACK={xnack}")
                bandwidth_rows.append([
                    mode.upper(), str(EXPECTED_CUS[mode]), allocator, str(xnack),
                    "—", "—", "—", "—",
                ])
                continue
            bandwidth_rows.append([
                mode.upper(), str(EXPECTED_CUS[mode]), allocator, str(xnack),
                *[f"{values[metric]:.1f}" for metric in METRICS],
            ])

    output = "\n".join([
        "# Single logical-partition HIP STREAM comparison",
        "",
        "One process runs on logical device 0 in each mode. Values are decimal GB/s.",
        "",
        "## Selected runs",
        markdown_table(["Mode", "Run", "CUs used"], selected_rows),
        "",
        "## Bandwidth",
        markdown_table(
            ["Mode", "CUs", "Allocator", "XNACK", "Copy", "Scale", "Add", "Triad"],
            bandwidth_rows,
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
