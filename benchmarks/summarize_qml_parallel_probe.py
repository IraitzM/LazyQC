#!/usr/bin/env python3
"""Summarize fixed-graph QML worker scaling and pool setup repayment."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def geomean(values):
    values = tuple(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def summarize(payload):
    groups = defaultdict(lambda: defaultdict(list))
    for row in payload["runs"]:
        groups[(row["case_id"], row["graph_strategy"])][row["workers"]].append(row)

    case_rows = []
    for (case_id, strategy), workers in sorted(groups.items()):
        medians = {
            worker: median(row["collection_ns"] for row in rows)
            for worker, rows in workers.items()
        }
        representative = workers[1][0]
        case_rows.append(
            {
                "case_id": case_id,
                "pattern": representative["pattern"],
                "family": representative["family"],
                "graph_strategy": strategy,
                "task_count": representative["task_count"],
                "median_collection_ns": {str(k): v for k, v in medians.items()},
                "worker_to_serial_ratio": {
                    str(worker): medians[worker] / medians[1] for worker in medians if worker != 1
                },
            }
        )

    scaling = []
    for strategy in sorted({row["graph_strategy"] for row in case_rows}):
        selected = [row for row in case_rows if row["graph_strategy"] == strategy]
        for workers in (2, 4):
            ratios = [row["worker_to_serial_ratio"][str(workers)] for row in selected]
            scaling.append(
                {
                    "graph_strategy": strategy,
                    "workers": workers,
                    "case_count": len(selected),
                    "geomean_worker_to_serial": geomean(ratios),
                    "cases_faster_than_serial": sum(ratio < 1 for ratio in ratios),
                    "median_task_count": median(row["task_count"] for row in selected),
                }
            )

    return {
        "source_artifact": "results/qml_workloads/parallel_probe.json",
        "raw_run_count": len(payload["runs"]),
        "failure_count": len(payload["metadata"]["failures"]),
        "max_abs_error": max(row["max_abs_error"] for row in payload["runs"]),
        "suite_aggregate": payload["metadata"]["suite_aggregate"],
        "scaling": scaling,
        "cases": case_rows,
    }


def render(summary):
    lines = [
        "# QML Fixed-Graph Parallel Probe",
        "",
        (
            f"Timed collections: {summary['raw_run_count']}; validation failures: "
            f"{summary['failure_count']}; maximum absolute oracle error: "
            f"{summary['max_abs_error']:.3e}."
        ),
        "",
        (
            "Ratios below 1 favor parallel collection. Per-case ratios use medians over "
            "eight repeated collections, then a geometric mean over cases. The suite "
            "lifecycle ratio adds one process-pool startup and shutdown per worker count "
            "to all probe collections; it reflects this probe's mixed workload suite, "
            "not a universal per-task speedup."
        ),
        "",
        (
            "| Fixed graph | Workers | Cases | Geomean parallel / serial | "
            "Faster cases | Median tasks |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scaling"]:
        lines.append(
            f"| {row['graph_strategy']} | {row['workers']} | {row['case_count']} | "
            f"{row['geomean_worker_to_serial']:.3f} | "
            f"{row['cases_faster_than_serial']} | {row['median_task_count']} |"
        )
    lines.extend(
        [
            "",
            (
                "| Workers | Collection sum (ns) | Pool setup (ns) | Shutdown (ns) | "
                "Setup-charged suite / serial |"
            ),
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for worker, row in sorted(summary["suite_aggregate"].items(), key=lambda item: int(item[0])):
        lines.append(
            f"| {worker} | {row['collection_sum_ns']} | {row['setup_ns']} | "
            f"{row['shutdown_ns']} | {row['suite_lifecycle_over_serial']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "This is a fixed-graph scheduling experiment. It does not include plan "
                "search in the worker comparison, and the pool is reused across the "
                "entire probe. Raw repetitions and resource observations are in "
                "`parallel_probe.json`."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/qml_workloads/parallel_probe.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qml_workloads")
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "parallel_probe_summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    (args.output_dir / "parallel_probe_SUMMARY.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
