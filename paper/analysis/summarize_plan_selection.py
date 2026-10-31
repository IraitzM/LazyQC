#!/usr/bin/env python3
"""Reanalyse saved plan-selection evidence without running quantum benchmarks.

Run from any directory with Python's standard library:
    python paper/analysis/summarize_plan_selection.py

The output is descriptive: workloads and reuse horizons are correlated, and
the two-policy oracle is reconstructed from timing medians, not exhaustive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "results" / "workload_manifest" / "v2"
GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def read_artifact(path: Path) -> tuple[dict, dict]:
    raw = path.read_bytes()
    data = json.loads(raw)
    provenance = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }
    metadata = data.get("metadata", {})
    for key in ("source_sha256", "timestamp_utc", "command", "cpu", "packages"):
        if key in metadata:
            provenance[key] = metadata[key]
    return data, provenance


def timing_medians(rows: list[dict], repeats: list[int]) -> dict:
    return {
        policy: [
            median(
                r["planning_ns"]
                for r in rows
                if r["mode"] == cold
                and r["repetition"] in repeats
                and r["status"] == "ok"
            ),
            median(
                r["total_ns"]
                for r in rows
                if r["mode"] == warm
                and r["repetition"] in repeats
                and r["status"] == "ok"
            ),
        ]
        for policy, cold, warm in (
            ("B3", "B3_independent", "B3_cached"),
            ("B5", "B5_joint", "B5_cached"),
        )
    }


def summarise_policy(policy_name: str, timing_name: str) -> dict:
    policy, policy_provenance = read_artifact(EVIDENCE / policy_name)
    timing, timing_provenance = read_artifact(EVIDENCE / timing_name)
    if set(policy["train"]) & set(policy["test"]):
        raise ValueError("Timing train and test repetitions overlap")
    if any(i < 0 for i in policy["train"] + policy["test"]):
        raise ValueError("Warmup repetition included in policy split")

    aggregate = {name: {"errors": [], "regrets_ns": [], "ratios": []}
                 for name in policy["summary"]}
    cases = []
    for case in policy["cases"]:
        name = case["case"]
        rows = [r for r in timing["runs"] if r["case"] == name]
        train = timing_medians(rows, policy["train"])
        test = timing_medians(rows, policy["test"])
        if train != case["H_ns"] or test != case["H_test_ns"]:
            raise ValueError(f"Saved timing medians do not match: {name}")
        if [r["R"] for r in case["R"]] != GRID:
            raise ValueError(f"Unexpected reuse grid: {name}")

        joint = timing["plans"][f"{name}/B5_joint"]
        standalone = timing["plans"][f"{name}/B3_independent"]
        incumbent = min(
            alternative["score"]
            for alternative in joint["alternatives"]
            if alternative["candidate"].split("/")[0] != "joint"
        )
        if incumbent != case["S_inc_pred_ns"]:
            raise ValueError(f"Saved incumbent score does not match: {name}")
        if joint["estimated_cost"] != case["S_joint_pred_ns"]:
            raise ValueError(f"Saved joint score does not match: {name}")

        for entry in case["R"]:
            reuse = entry["R"]
            lifetimes = {mode: h + reuse * t for mode, (h, t) in test.items()}
            oracle = "B5" if lifetimes["B5"] < lifetimes["B3"] else "B3"
            oracle_ns = lifetimes[oracle]
            if entry["oracle"] != oracle:
                raise ValueError(f"Saved oracle differs: {name}, R={reuse}")
            for rule, values in aggregate.items():
                pick = entry[rule]["pick"]
                actual_ns = lifetimes[pick]
                if rule == "both_search":
                    actual_ns = sum(h for h, _ in test.values()) + reuse * test[pick][1]
                regret = actual_ns - oracle_ns
                error = pick != oracle
                if error != entry[rule]["error"] or not math.isclose(
                    regret, entry[rule]["regret_ns"], rel_tol=1e-12, abs_tol=1e-6
                ):
                    raise ValueError(f"Saved policy outcome differs: {name}, {rule}, R={reuse}")
                values["errors"].append(error)
                values["regrets_ns"].append(regret)
                values["ratios"].append(actual_ns / oracle_ns)

        cases.append({
            "case": name,
            "train_H_T_ms": {k: [v / 1e6 for v in pair] for k, pair in train.items()},
            "test_H_T_ms": {k: [v / 1e6 for v in pair] for k, pair in test.items()},
            "threshold_R_train": case["R_star_train"],
            "oracle_selects_joint_at_R": [e["R"] for e in case["R"] if e["oracle"] == "B5"],
            "predicted_B3_ms": incumbent / 1e6,
            "predicted_B5_ms": joint["estimated_cost"] / 1e6,
            "nested_incumbent_equals_standalone_B3_score": incumbent == standalone["estimated_cost"],
        })

    metrics = {}
    for rule, values in aggregate.items():
        count = len(values["errors"])
        metrics[rule] = {
            "selection_errors": sum(values["errors"]),
            "decisions": count,
            "selection_error_rate": mean(values["errors"]),
            "mean_regret_ms": mean(values["regrets_ns"]) / 1e6,
            "max_regret_ms": max(values["regrets_ns"]) / 1e6,
            "mean_normalized_regret": mean(r - 1 for r in values["ratios"]),
            "geomean_lifetime_ratio_to_two_policy_oracle": math.exp(mean(map(math.log, values["ratios"]))),
        }
        for key in ("selection_error_rate", "mean_regret_ms", "max_regret_ms"):
            if not math.isclose(metrics[rule][key], policy["summary"][rule][key], abs_tol=1e-10):
                raise ValueError(f"Saved aggregate differs: {rule}, {key}")

    return {
        "policy_artifact": policy_provenance,
        "timing_artifact": timing_provenance,
        "train_repetitions": policy["train"],
        "test_repetitions": policy["test"],
        "warmup_repetition_excluded": -1,
        "workloads": len(cases),
        "reuse_grid": GRID,
        "recorded_calibration_ms": timing["metadata"]["calibration_training"]["calibration_ns"] / 1e6,
        "metrics": metrics,
        "cases": cases,
    }


def summarise_trajectory(filename: str) -> dict:
    data, provenance = read_artifact(EVIDENCE / filename)
    return {
        "artifact": provenance,
        "failures": data["metadata"]["failures"],
        "measurement": "Planning wall time only; no execution or lifetime speedup measured.",
        "statistic": "Runner selects sorted[n // 2]: fifth order statistic for eight bindings, not conventional even-sample median.",
        "scope": [
            "The seed binding is included in both timing summaries and hit/fidelity counts.",
            "Full planning runs before replay for every binding; order is not randomized.",
            "Replay uses a template extracted from full planning of binding zero; replay timing excludes template acquisition.",
            "Independent NumPy oracle checks replay outputs; decision fidelity checks task count and score equality only.",
            "Artifact retains summaries only; raw binding times, full plan identities, and seed are not stored.",
            "Current runner defaults to seed 11 and samples angles uniformly in [-1, 1]; this is runner information, not a persisted artifact seed.",
            "No zero-angle topology-mismatch result is persisted in these trajectory artifacts.",
        ],
        "cases": [dict(row, reported_planning_time_ratio=row["median_full_ms"] / row["median_hint_ms"])
                  for row in data["summary"]],
    }


def summarise_forced_strategies() -> dict:
    data, provenance = read_artifact(ROOT / "results/workload_manifest/v1/evaluation.json")
    cases = []
    for name in sorted({r["case"] for r in data["runs"]}):
        rows = [r for r in data["runs"] if r["case"] == name]
        summaries = {r["strategy"]: r for r in data["summary"] if r["case"] == name}
        medians = {}
        for strategy in ("full", "separate", "grouped", "auto"):
            observations = [r for r in rows if r["strategy"] == strategy]
            medians[strategy] = median(r["total_ns"] for r in observations) / 1e6
            if not math.isclose(medians[strategy], summaries[strategy]["median_ms"], abs_tol=1e-12):
                raise ValueError(f"Saved v1 timing median differs: {name}, {strategy}")
            if len(observations) != summaries[strategy]["n"]:
                raise ValueError(f"Saved v1 observation count differs: {name}, {strategy}")
        selected = {r["selected"] for r in rows if r["strategy"] == "auto"}
        if len(selected) != 1:
            raise ValueError(f"Auto choices vary across repeats: {name}")
        pick = next(iter(selected))
        best = min(("full", "separate", "grouped"), key=medians.get)
        optimum = medians[best]
        cases.append({
            "case": name,
            "repetitions_per_strategy": summaries["auto"]["n"],
            "median_ms": medians,
            "minimum_median_forced_strategy": best,
            "auto_selected_strategy": pick,
            "selection_matches_minimum_median": pick == best,
            "selected_forced_to_best_forced_ratio": medians[pick] / optimum,
            "selected_forced_regret_ms": medians[pick] - optimum,
            "auto_total_to_best_forced_ratio": medians["auto"] / optimum,
            "auto_total_regret_ms": medians["auto"] - optimum,
            "max_abs_error": max(r["max_abs_error"] for r in rows),
        })
    return {
        "artifact": provenance,
        "workloads": len(cases),
        "timed_observations": len(data["runs"]),
        "selection_matches_minimum_median_count": sum(c["selection_matches_minimum_median"] for c in cases),
        "auto_total_slower_than_best_forced_count": sum(c["auto_total_to_best_forced_ratio"] > 1 for c in cases),
        "geomean_selected_forced_to_best_forced_ratio": math.exp(mean(math.log(c["selected_forced_to_best_forced_ratio"]) for c in cases)),
        "geomean_auto_total_to_best_forced_ratio": math.exp(mean(math.log(c["auto_total_to_best_forced_ratio"]) for c in cases)),
        "max_abs_error": max(c["max_abs_error"] for c in cases),
        "scope": [
            "Five independent-pair sizes have 20 repetitions per strategy; ten connected batches have 10 repetitions per strategy.",
            "Randomized interleaved strategy order; timing includes plan and collect after request and backend construction.",
            "Best forced strategy is minimum of measured per-case full/separate/grouped medians, using the same sample; no held-out optimality claim.",
            "Auto selected strategy is constant across all recorded repetitions of each case.",
            "The selected-forced ratio uses separately timed execution of the same strategy label as a proxy for choice quality; it does not isolate auto execution phases.",
            "The auto-total ratio includes automatic selection overhead, but differences from forced execution also include ordinary timing variation.",
            "Minimum-median matches do not establish statistical significance: connected_8q_overlap_1 differs by only 0.000553 ms between full and grouped.",
            "Connected disjoint batches have disjoint observable supports; their execution cones can overlap.",
            "Pairs use analytic references and connected cases use independent NumPy references; source version differs from the v2 policy artifacts.",
        ],
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("plan_selection_summary.json"))
    args = parser.parse_args()
    report = {
        "analysis_version": 1,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evidence_scope": {
            "lifetime_model": "L_p(R) = median(cold planning p) + R * median(warm cached construction + planning + execution p).",
            "oracle": "Minimum of the two forced policy lifetime reconstructions on test timing repetitions; no global-optimality certificate.",
            "measured_versus_reconstructed": "H and T are measured; R-horizon lifetimes and regrets are reconstructed, not measured R-call runs.",
            "both_search": "Charges H_B3 + H_B5 + R*T_selected. Selection error tracks chosen label only; regret also charges both searches.",
            "costs_excluded": [
                "Calibration, imports, independent oracle checks, and untimed warmup.",
                "Acquisition of training timing data and both candidate scores.",
                "Cold cache insertion and cold first execution are approximated by H plus warm T, not measured as a complete deployment trajectory.",
                "Runtime overhead of invoking the offline selection rules themselves.",
            ],
            "score_rule": "Offline rule using measured training H and candidate scores produced after search; not evidence of predicting savings before search.",
            "validation_split": "Timing repetitions split within each workload; thresholds fit per workload, including the additional heldout-file workloads.",
            "dependence": "Ten R decisions per workload share the same H/T medians; decisions are correlated and are not independent experimental replicates.",
            "uncertainty": "One host, one session per artifact, very small timing splits; descriptive aggregates without statistical generalization claims.",
            "speedup_summary_warning": "Raw v2 paired speedup summaries include repetition -1; this analysis uses explicit nonnegative policy splits and does not use those summaries.",
        },
        "policy_sets": {
            "qaoa": summarise_policy("policy_sched.json", "evaluation_r7_sched.json"),
            "additional_workloads": summarise_policy("policy_heldout.json", "evaluation_heldout.json"),
        },
        "forced_strategy_selection": summarise_forced_strategies(),
        "structural_replay": {
            "none": summarise_trajectory("trajectory.json"),
            "zx": summarise_trajectory("trajectory_zx.json"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
