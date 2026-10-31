#!/usr/bin/env python3
"""Summarize the local 20-qubit and IonQ noisy QML cone probes."""

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def geomean(values):
    return math.exp(statistics.mean(math.log(value) for value in values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        type=Path,
        default=ROOT / "results/qml_workloads/large_cone_probe.json",
    )
    parser.add_argument(
        "--noisy",
        type=Path,
        default=ROOT / "results/qml_workloads/ionq_noisy_qml_probe.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/qml_workloads/large_qml_probe_SUMMARY.md",
    )
    args = parser.parse_args()
    local = json.loads(args.local.read_text())
    noisy = json.loads(args.noisy.read_text())
    indexed = {}
    for row in local["summary"]:
        indexed.setdefault((row["family"], row["qubits"], row["seed_offset"]), {})[
            row["strategy"]
        ] = row

    lines = [
        "# Large-cone QML probes",
        "",
        f"Local run: {local['metadata']['case_count']} cases, "
        f"{len(local['runs'])} collections, {len(local['metadata']['failures'])} failures. "
        "Each case uses the same-state XYZ observable group, depth 4, and five "
        "interleaved repetitions for full, separate, grouped, and auto plans.",
        "",
        "| Qubits | Grouped / separate total-time GM | Grouped / full GM | "
        "Grouped wins vs full |",
        "|---:|---:|---:|---:|",
    ]
    for n in (16, 18, 20):
        cases = [v for (_, qubits, _), v in indexed.items() if qubits == n]
        grouped_separate = [
            v["grouped"]["median_total_ms"] / v["separate"]["median_total_ms"]
            for v in cases
        ]
        grouped_full = [
            v["grouped"]["median_total_ms"] / v["full"]["median_total_ms"]
            for v in cases
        ]
        wins = sum(ratio < 1 for ratio in grouped_full)
        lines.append(
            f"| {n} | {geomean(grouped_separate):.3f}× | "
            f"{geomean(grouped_full):.3f}× | {wins}/{len(grouped_full)} |"
        )

    lines.extend(
        [
            "",
            "At 20 qubits, grouped cone widths were 6 for hardware-efficient local, "
            "8 for QCNN-like local, and 20 for data re-uploading. Grouped/full "
            "geometric-mean time ratios over the two held-out seeds were 0.002, "
            "0.003, and 0.986, respectively. Data re-uploading is the saturation "
            "control: sharing avoids three same-cone evolutions versus separate, "
            "but gives no gain over the full-state execution.",
            "",
            "## IonQ Forte-1 noisy simulator",
            "",
            f"{len(noisy['runs'])} completed jobs; {noisy['metadata']['shots']} shots/job; "
            "three measurement bases (X, Y, Z) per strategy and family. The noisy "
            "simulator samples per shot and uses an approximate hardware noise model.",
            "",
            "| Family | Strategy | Task widths | Gates/job | Median simulator ms | "
            "Max |noisy − ideal| |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for case in noisy["cases"]:
        rows = [
            row
            for row in noisy["runs"]
            if row["case_id"] == case["case_id"] and row["strategy"] == case["strategy"]
        ]
        widths = ",".join(str(width) for width in case["task_widths"])
        gates = statistics.median(row["submitted_gates"] for row in rows)
        sim_ms = statistics.median(row["execution_duration_ms"] for row in rows)
        lines.append(
            f"| `{case['family']}` | `{case['strategy']}` | {widths} | "
            f"{gates:g} | {sim_ms:g} | {case['max_abs_deviation_from_ideal']:.3f} |"
        )
    lines.extend(
        [
            "",
            "For hardware-efficient local, each full-state measurement used 20 wires "
            "and 236 logical operations (237 including basis rotation); each cone "
            "measurement used 6 wires and 50 logical operations. For QCNN-like local, "
            "full used 20 wires/95 logical operations and cones used 8/40. For data "
            "re-uploading, full and cone circuits were both 20 wires/236 logical "
            "operations. All strategies submitted three jobs per family, one per "
            "noncommuting X/Y/Z measurement basis. Separate and grouped strategies "
            "have the same submitted per-basis cone circuit here, so grouping does "
            "not reduce physical measurement-circuit count in this probe.",
            "",
            "These are noisy-simulator timings, not QPU gate duration or queue delay. "
            "The noise model is approximate, and sampled deviations include both "
            "noise and finite-shot variation. Treat the results as evidence that "
            "cone reduction can lower simulated work for circuits whose output cones "
            "are narrow, and as a controlled negative case for saturated cones. Only "
            "one seeded noisy realization per output was run, so deviations are "
            "descriptive; these results do not establish quantitative hardware "
            "performance.",
            "",
            f"Local raw data: `{args.local}`. Noisy raw data: `{args.noisy}`.",
            "Runners: `benchmarks/run_qml_large_cone_probe.py` and "
            "`benchmarks/run_ionq_noisy_qml_probe.py`.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
