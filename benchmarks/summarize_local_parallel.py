#!/usr/bin/env python3
"""Aggregate fixed-DAG local runs without mixing timing probes or hiding regressions."""

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import median

from benchmark_support import geomean

ROOT = Path(__file__).resolve().parents[1]


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)


def paired_interval(ratios, seed):
    rng = random.Random(seed)
    boot = [geomean(rng.choices(ratios, k=len(ratios))) for _ in range(2000)]
    return [quantile(boot, 0.025), quantile(boot, 0.975)]


def sustained_crossing(serial, parallel):
    wins = [a >= b for a, b in zip(serial, parallel, strict=True)]
    return next((i + 1 for i in range(len(wins)) if all(wins[i:])), None)


def cumulative(setup, values):
    result = []
    for value in values:
        setup += value
        result.append(setup)
    return result


def summarize_config(case, config, calibration):
    rows = config["runs"]
    workers = config["workers"]
    stats = {}
    for mode in sorted({r["mode"] for r in rows}):
        matching = [r for r in rows if r["mode"] == mode]
        times = [r["total_ns"] for r in matching]
        stats[mode] = {
            "median_ns": median(times),
            "iqr_ns": [quantile(times, 0.25), quantile(times, 0.75)],
            "phase_median_ns": {
                field: median(r["stats"][field] for r in matching)
                for field in (
                    "lowering_ns",
                    "evolution_ns",
                    "expectation_ns",
                    "reconstruction_ns",
                    "execution_ns",
                )
            },
            "execution_modes": sorted({r["stats"]["execution_mode"] for r in matching}),
            "peak_active_state_bytes": max(r["stats"]["peak_active_state_bytes"] for r in matching),
            "peak_in_flight": max(r["stats"]["peak_in_flight"] for r in matching),
        }
    paired = {}
    costs = {}
    if workers > 1:
        base = {r["trial"]: r for r in rows if r["mode"] == "serial"}
        for mode in ("parallel", "auto"):
            ratios = [
                base[r["trial"]]["total_ns"] / r["total_ns"] for r in rows if r["mode"] == mode
            ]
            paired[mode] = {
                "geomean": geomean(ratios),
                "bootstrap_95": paired_interval(ratios, 7411),
                "ratios": ratios,
            }
        common = case["construction_ns"] + case["planning_ns"] + calibration
        serial = cumulative(
            common,
            [config["first_serial"]["total_ns"]] + [base[i]["total_ns"] for i in sorted(base)],
        )
        selected = {r["trial"]: r for r in rows if r["mode"] == "parallel"}
        parallel = cumulative(
            common + config["warmup_ns"] + config["shutdown_ns"],
            [config["first"]["total_ns"]] + [selected[i]["total_ns"] for i in sorted(selected)],
        )
        costs = {
            "collections": len(serial),
            "serial_cumulative_ns": serial,
            "parallel_cumulative_ns": parallel,
            "sustained_break_even_call": sustained_crossing(serial, parallel),
            "final_speedup": serial[-1] / parallel[-1],
            "scope": (
                "sum of measured first+warm calls plus common construction/planning/calibration; "
                "pool warmup and shutdown charged to parallel; interleaving gaps and "
                "diagnostic probes excluded"
            ),
        }
    resource = config["resources"]
    wall = resource["collection"]["total_ns"]
    return {
        "workers": workers,
        "payload_slots": config["payload_slots"],
        "total_state_bytes": config["total_state_bytes"],
        "modes": stats,
        "paired_speedups": paired,
        "lifetime": costs,
        "warmup_ns": config["warmup_ns"],
        "shutdown_ns": config["shutdown_ns"],
        "sampled_peak_sum_rss_bytes": resource["sampled_peak_sum_rss_bytes"],
        "baseline_sum_rss_bytes": resource["baseline_sum_rss_bytes"],
        "resource_probe_cpu_ns": sum(resource["cpu_ns_by_pid"].values()),
        "resource_probe_wall_ns": wall,
        "resource_probe_core_equivalents": (
            sum(resource["cpu_ns_by_pid"].values()) / wall if wall >= 100_000_000 else None
        ),
        "cpu_resolution_note": (
            "Core equivalents suppressed below 100 ms: 10 ms process ticks and "
            "sampling boundaries dominate short probes; CPU time includes profiling overhead."
        ),
        "resource_samples": resource["sample_count"],
    }


def summarize(sessions):
    if len({d["metadata"]["manifest_sha256"] for d in sessions}) != 1:
        raise ValueError("Manifest mismatch")
    for d in sessions:
        for kind in ("source_sha256", "benchmark_sha256"):
            for name, expected in d["metadata"][kind].items():
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Source mismatch: {name}")
    cases = []
    for spec in sessions[0]["manifest"]:
        records = [
            next(c for c in d["cases"] if c["case"]["name"] == spec["name"]) for d in sessions
        ]
        if len({c["dag_sha256"] for c in records}) != 1:
            raise ValueError(f"DAG mismatch: {spec['name']}")
        configs = {}
        for i, record in enumerate(records):
            for config in record["configurations"]:
                key = f"w{config['workers']}_m{config['payload_slots']}"
                configs.setdefault(key, []).append(
                    summarize_config(record, config, sessions[i]["metadata"]["calibration_wall_ns"])
                )
        cases.append(
            {
                "name": spec["name"],
                "specification": spec,
                "unique_tasks": len(records[0]["dag"]["tasks"]),
                "dag_sha256": records[0]["dag_sha256"],
                "construction_ns": [c["construction_ns"] for c in records],
                "planning_ns": [c["planning_ns"] for c in records],
                "serialized_task_bytes": [
                    sum(t["serialized_task_bytes"] for t in c["task_profiles"]) for c in records
                ],
                "serialized_result_bytes": [
                    sum(t["serialized_result_bytes"] for t in c["task_profiles"]) for c in records
                ],
                "pickle_roundtrip_ns": [
                    sum(t["pickle_roundtrip_median_ns"] for t in c["task_profiles"])
                    for c in records
                ],
                "configurations": configs,
            }
        )
    weak = []
    if any(c["name"] == "balanced_14q_2tasks" for c in cases):
        for s in range(len(sessions)):
            base = next(c for c in cases if c["name"] == "balanced_14q_2tasks")["configurations"][
                "w1_m1"
            ][s]["modes"]["serial"]["median_ns"]
            points = []
            for workers in (1, 2, 4):
                c = next(c for c in cases if c["name"] == f"balanced_14q_{2 * workers}tasks")
                config = c["configurations"][f"w{workers}_m{workers}"][s]
                elapsed = config["modes"]["serial" if workers == 1 else "parallel"]["median_ns"]
                points.append(
                    {
                        "workers": workers,
                        "tasks": 2 * workers,
                        "median_ns": elapsed,
                        "weak_efficiency": base / elapsed,
                        "throughput_tasks_s": 2 * workers * 1e9 / elapsed,
                    }
                )
            weak.append(points)
    allrows = [
        r
        for d in sessions
        for c in d["cases"]
        for config in c["configurations"]
        for r in [
            config["first"],
            *([config["first_serial"]] if config["workers"] > 1 else []),
            *config["runs"],
            config["resources"]["collection"],
        ]
    ]
    return {
        "sessions": len(sessions),
        "repeats_per_mode": [d["metadata"]["repeats"] for d in sessions],
        "manifest_sha256": sessions[0]["metadata"]["manifest_sha256"],
        "calibration_wall_ns": [d["metadata"]["calibration_wall_ns"] for d in sessions],
        "cases": cases,
        "weak_scaling": weak,
        "validated_collections": len(allrows),
        "timed_warm_collections": sum(
            len(config["runs"])
            for d in sessions
            for c in d["cases"]
            for config in c["configurations"]
        ),
        "max_abs_error": max(r["max_abs_error"] for r in allrows),
        "uncertainty": (
            "paired bootstrap within each session; ranges across interpreters on one host "
            "are not independent-machine confidence intervals"
        ),
        "resource_scope": (
            "separate diagnostic collections, parent plus active pool summed sampled RSS; "
            "shared pages counted per process; CPU uses /proc ticks, not worker phase sums"
        ),
    }


def report(data):
    lines = [
        "Local multiprocessing study",
        "",
        f"{data['sessions']} sessions, {data['timed_warm_collections']} warm timed collections. "
        f"Maximum oracle error: {data['max_abs_error']:.3g}.",
        "",
        "Ranges below span sessions on one host. Speedups use paired geometric means; "
        "time uses per-session medians.",
        "",
        "| Workload | Tasks | Serial ms | 2-worker speedup | 4-worker speedup | "
        "4-worker efficiency | Measured-lifetime speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def span(values):
        return f"{min(values):.2f}–{max(values):.2f}"

    for c in data["cases"]:
        cfg = c["configurations"]
        two = cfg["w2_m2"]
        four = cfg["w4_m4"]
        lines.append(
            f"| {c['name']} | {c['unique_tasks']} | "
            f"{span([x['modes']['serial']['median_ns'] / 1e6 for x in cfg['w1_m1']])} | "
            f"{span([x['paired_speedups']['parallel']['geomean'] for x in two])} | "
            f"{span([x['paired_speedups']['parallel']['geomean'] for x in four])} | "
            f"{span([x['paired_speedups']['parallel']['geomean'] / 4 for x in four])} | "
            f"{span([x['lifetime']['final_speedup'] for x in four])} |"
        )
    lines += [
        "",
        "Per-session setup, phases, CPU probes, RSS, memory caps, automatic dispatch, "
        "serialized bytes, and observed break-even crossings are retained in `summary.json`. "
        "Warm speedup excludes setup; lifetime accounting includes construction, cold planning, "
        "calibration, pool warmup, first numerical collection, subsequent measured calls, "
        "and shutdown. Resource probes are excluded from timing trials. "
        "No multi-host execution or energy measurement is claimed.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/local_parallel")
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("session*/evaluation.json"))
    if not paths:
        raise ValueError("No completed sessions")
    data = summarize([json.loads(p.read_text()) for p in paths])
    (args.input_dir / "summary.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.input_dir / "SUMMARY.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
