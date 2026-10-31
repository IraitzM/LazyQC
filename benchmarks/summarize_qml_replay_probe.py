#!/usr/bin/env python3
"""Summarize charged structural replay across generated QML trajectories."""

import argparse
import json
import math
import statistics
from pathlib import Path


def geometric_mean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path, nargs="?", default=Path("results/qml_workloads/replay_probe.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/qml_workloads/replay_probe_SUMMARY.md")
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    cases = payload["cases"]
    lines = [
        "# Structural replay across QML parameter trajectories",
        "",
        f"Cases: {len(cases)}; timed rows: {len(payload['runs'])}; "
        f"failures: {len(payload['metadata']['failures'])}; "
        f"maximum oracle error: {payload['metadata']['max_abs_error']:.3g}.",
        "",
        "Lifetime ratios are fresh planning plus collection divided by replay "
        "planning plus collection. Template creation and its first collection "
        "are charged once to replay; later parameter points each build a fresh "
        "state while reusing only the structural plan.",
        "",
        "| Trajectory pattern | Cases | Geometric mean lifetime ratio | "
        "Replay wins | Median fresh/replay planning (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for pattern in sorted({case["pattern"] for case in cases}):
        group = [case for case in cases if case["pattern"] == pattern]
        ratio = geometric_mean(case["fresh_over_replay_lifetime"] for case in group)
        wins = sum(case["fresh_over_replay_lifetime"] > 1 for case in group)
        fresh_plan = statistics.median(case["median_fresh_planning_ns"] for case in group) / 1e6
        replay_plan = statistics.median(case["median_replay_planning_ns"] for case in group) / 1e6
        lines.append(
            f"| `{pattern}` | {len(group)} | {ratio:.3f}× | {wins}/{len(group)} "
            f"| {fresh_plan:.2f} / {replay_plan:.2f} |"
        )
    overall = geometric_mean(case["fresh_over_replay_lifetime"] for case in cases)
    lines.extend(
        [
            "",
            f"Across all cases the geometric mean lifetime ratio is **{overall:.3f}×**; "
            f"the per-case range is {min(c['fresh_over_replay_lifetime'] for c in cases):.3f}–"
            f"{max(c['fresh_over_replay_lifetime'] for c in cases):.3f}×.",
            "",
            "This is evidence for a classical planning-time benefit in these generated "
            "parameter trajectories. It is not evidence of quantum-state reuse, a QPU "
            "sequencing or queue benefit, trained-model performance, or transfer to "
            "unseen circuit families. The benchmark uses one machine and exact "
            "simulation; the case count is small and the timing ratios are descriptive.",
            "",
            f"Raw data: `{args.input}`. Runner: `benchmarks/run_qml_replay_probe.py`.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
