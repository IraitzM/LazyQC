"""Aggregate independent sessions; retain regressions and verify source provenance."""

import hashlib
import json
from pathlib import Path
from statistics import median

from benchmark_support import geomean

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / "results/priorities"
    sessions, calibration = [], []
    for i in range(3):
        data = json.loads((base / f"session{i}/evaluation.json").read_text())
        profile = json.loads((base / f"session{i}/profile.json").read_text())
        for name, digest in data["metadata"]["source_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source mismatch: {name}")
        if (
            data["metadata"]["runner_sha256"]
            != hashlib.sha256((ROOT / "benchmarks/run_priorities.py").read_bytes()).hexdigest()
        ):
            raise ValueError("Runner differs from recorded session")
        sessions.append(data)
        calibration.append(profile["calibration_ns"])
    if len({d["metadata"]["manifest_sha256"] for d in sessions}) != 1:
        raise ValueError("Independent sessions used different manifests")
    cases = []
    for index, initial in enumerate(sessions[0]["summary"]):
        entry = {"case": initial["case"], "policies": {}}
        for mode in ("joint", "joint_budget"):
            summaries = [d["summary"][index] for d in sessions]
            ratios = [s["comparisons"][mode]["setup_inclusive_speedup"] for s in summaries]
            with_calibration = [
                (s["cumulative_ns"]["independent"][-1] + cost)
                / (s["cumulative_ns"][mode][-1] + cost)
                for s, cost in zip(summaries, calibration, strict=True)
            ]
            entry["policies"][mode] = {
                "session_speedups": ratios,
                "geomean": geomean(ratios),
                "range": [min(ratios), max(ratios)],
                "session_speedups_including_calibration": with_calibration,
                "break_even_sustained_to_end": [
                    s["comparisons"][mode]["break_even_sustained_to_end"] for s in summaries
                ],
            }
        cases.append(entry)
    overall = {}
    for mode in ("joint", "joint_budget"):
        ratios = [r for c in cases for r in c["policies"][mode]["session_speedups"]]
        overall[mode] = {
            "wins": sum(r > 1 for r in ratios),
            "comparisons": len(ratios),
            "geomean": geomean(ratios),
            "regression_fraction": sum(r < 1 for r in ratios) / len(ratios),
        }
    workers = {}
    for name in sessions[0]["workers"]:
        values = [d["workers"][name] for d in sessions]
        workers[name] = {
            "session_median_ns": [v["median_ns"] for v in values],
            "persistent_parallel_speedups": [
                v["median_ns"]["serial_same_dag"] / v["median_ns"]["persistent_parallel"]
                for v in values
            ],
            "persistent_auto_speedups": [
                v["median_ns"]["serial_same_dag"] / v["median_ns"]["persistent_auto"]
                for v in values
            ],
            "pool_startup_ns": [v["transport_profiles"]["parallel"]["startup_ns"] for v in values],
        }
    unique = {
        mode: [
            median(r["total_ns"] for r in d["unique_bindings"] if r["mode"] == mode)
            for d in sessions
        ]
        for mode in ("independent", "joint", "joint_budget")
    }
    result = {
        "scope": (
            f"{sessions[0]['metadata']['collections']} plan/collect calls including first build; "
            "imports and request construction excluded"
        ),
        "calibration_ns": calibration,
        "manifest_sha256": sessions[0]["metadata"]["manifest_sha256"],
        "sessions": 3,
        "cases": cases,
        "overall": overall,
        "workers": workers,
        "unique_binding_median_ns": unique,
        "raw_plan_collect_runs": sum(len(d["runs"]) for d in sessions),
        "max_abs_error": max(
            r["max_abs_error"] for d in sessions for r in d["runs"] + d["unique_bindings"]
        ),
        "uncertainty": (
            "ranges across three interpreter sessions on one host, not confidence intervals"
        ),
    }
    (base / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
