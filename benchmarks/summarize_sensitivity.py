"""Validate frozen provenance and aggregate repeated sensitivity measurements."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/sensitivity"


def interval(values):
    return [min(values), max(values)]


def summarize(runs, source_root):
    if not runs:
        raise ValueError("No sensitivity runs")
    fingerprints = runs[0]["metadata"]["source_sha256"]
    for path, digest in fingerprints.items():
        if hashlib.sha256((source_root / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source mismatch: {path}")
    expected = [(c["name"], c["workers"]) for c in runs[0]["cases"]]
    for run in runs:
        if run["metadata"]["source_sha256"] != fingerprints:
            raise ValueError("Measured sources differ across sessions")
        if [(c["name"], c["workers"]) for c in run["cases"]] != expected:
            raise ValueError("Case coverage differs across sessions")
    grouped = defaultdict(list)
    for run in runs:
        for row in run["cases"]:
            grouped[row["name"].rsplit("_s", 1)[0], row["workers"]].append(row)
            if any(
                a > b + 1e-13
                for a, b in zip(row["actual_errors"], row["error_bounds"], strict=True)
            ):
                raise ValueError("Observed approximation exceeds recorded bound")
            if row["max_numerical_error"] > run["metadata"]["evaluation_error_assumption"]:
                raise ValueError("Native/reference disagreement exceeds assumed allowance")
    families = []
    for (name, workers), rows in grouped.items():
        families.append(
            dict(
                name=name,
                workers=workers,
                measured_configurations=len(rows),
                independent_seeds=len({r["seed"] for r in rows}),
                exact_width=interval([r["exact_width"] for r in rows]),
                bounded_width=interval([r["bounded_width"] for r in rows]),
                payload_reduction=interval(
                    [r["exact_payload"] / r["bounded_payload"] for r in rows]
                ),
                warm_speedup=interval([r["warm_speedup"] for r in rows]),
                lifetime_speedup=interval([r["lifetime_speedup"] for r in rows]),
                lifetime_median=median(r["lifetime_speedup"] for r in rows),
                exact_warm_ms=interval([median(r["warm_ns"]["exact"]) / 1e6 for r in rows]),
                bounded_warm_ms=interval([median(r["warm_ns"]["bounded"]) / 1e6 for r in rows]),
                profile_ms=interval([r["profile_ns"] / 1e6 for r in rows]),
                search_ms=interval([r["approximation_planning_ns"] / 1e6 for r in rows]),
                startup_ms=interval([r["startup_ns"] / 1e6 for r in rows]),
                break_even_calls=[r["observed_break_even"] for r in rows],
                max_observable_error=max(max(r["actual_errors"]) for r in rows),
                max_bound=max(max(r["error_bounds"]) for r in rows),
                exact_rss_mib=interval(
                    [r["resources"]["exact"]["sampled_peak_sum_rss_bytes"] / 2**20 for r in rows]
                ),
                bounded_rss_mib=interval(
                    [r["resources"]["bounded"]["sampled_peak_sum_rss_bytes"] / 2**20 for r in rows]
                ),
            )
        )
    accuracy = runs[0]["accuracy"]
    random = accuracy["random"]
    budgets = defaultdict(
        lambda: dict(
            cases=0,
            uniform_violations=0,
            gradient_violations=0,
            bounded_violations=0,
            bounded_changes=0,
        )
    )
    for row in runs[0]["cases"]:
        if row["workers"] != 1:
            continue
        for sweep in row["sweeps"]:
            item = budgets[str(sweep["budget"])]
            item["cases"] += 1
            item["uniform_violations"] += sweep["uniform_violates"]
            item["gradient_violations"] += sweep["gradient_would_violate"]
            item["bounded_changes"] += bool(sweep["bounded_zeroed"])
            item["bounded_violations"] += any(
                e > sweep["budget"] + 1e-13 for e in sweep["bounded_errors"]
            )
    return dict(
        sessions=len(runs),
        families=families,
        policy_sweeps=dict(budgets),
        unique_random_circuits=len({r["seed"] for r in random}),
        unique_perturbation_vectors=len(random),
        unique_observable_comparisons=sum(len(r["actual"]) for r in random),
        random_bound_violations=sum(
            a > b + 1e-13 for r in random for a, b in zip(r["actual"], r["bound"], strict=True)
        ),
        linear_underestimates=sum(
            a > b + 1e-12 for r in random for a, b in zip(r["actual"], r["linear"], strict=True)
        ),
        max_random_reference_error=max(r["max_reference_error"] for r in random),
        max_random_fd_error=max(r["max_fd_error"] for r in random),
        max_timing_reference_error=max(c["max_numerical_error"] for r in runs for c in r["cases"]),
        timed_warm_collections=sum(
            len(c["warm_ns"]["exact"]) + len(c["warm_ns"]["bounded"])
            for r in runs
            for c in r["cases"]
        ),
        timed_first_collections=2 * sum(len(r["cases"]) for r in runs),
        resource_probe_collections=2 * sum(len(r["cases"]) for r in runs),
        stationary=accuracy["stationary"],
        shared_parameter=accuracy["shared_parameter"],
        finite_difference=accuracy["finite_difference"],
        parameter_shift_error=accuracy["parameter_shift_error"],
    )


def span(values, digits=2):
    return f"{values[0]:.{digits}f}–{values[1]:.{digits}f}"


def main():
    paths = sorted(BASE.glob("session*/evaluation.json"))
    result = summarize([json.loads(p.read_text()) for p in paths], BASE / "frozen_source")
    (BASE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Sensitivity validation results",
        "",
        f"{result['sessions']} sequential interpreter sessions; 9 timing fixtures per session; "
        f"{result['timed_warm_collections']} warm, {result['timed_first_collections']} first, "
        f"and {result['resource_probe_collections']} separate resource collections.",
        "",
        f"Independent accuracy study: {result['unique_random_circuits']} circuit seeds, "
        f"{result['unique_perturbation_vectors']} simultaneous perturbations, "
        f"{result['unique_observable_comparisons']} observable comparisons. "
        f"Bound violations: {result['random_bound_violations']}; "
        f"linear underestimates: {result['linear_underestimates']}.",
        "",
        f"Maximum native/reference disagreement: {result['max_timing_reference_error']:.3g} "
        f"(timing fixtures), {result['max_random_reference_error']:.3g} (random circuits). "
        "Maximum Jacobian/independent finite-difference discrepancy: "
        f"{result['max_random_fd_error']:.3g}.",
        "",
        "Ranges below span session/seed configurations, not confidence intervals. "
        "Warm speedup compares median exact and approximate collections at the same worker count. "
        "Lifetime charges profiling, search, startup and shutdown over eight collections. "
        "All timing comparisons use an absolute error budget of 0.01 per observable.",
        "",
        "| Fixture | Workers | Width | Payload factor | Warm speedup | "
        "Lifetime speedup | Profile ms |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in result["families"]:
        lines.append(
            f"| {row['name']} | {row['workers']} | {row['exact_width'][0]} → "
            f"{row['bounded_width'][0]} | {row['payload_reduction'][0]:.0f} | "
            f"{span(row['warm_speedup'])} | {span(row['lifetime_speedup'])} | "
            f"{span(row['profile_ms'])} |"
        )
    lines += [
        "",
        "## Accuracy-policy ablations",
        "",
        "Counts use each of the nine timing fixtures once (worker=1, session=0). "
        "Uniform rounding and gradient-only tests target the same eligible bridge gates.",
        "",
        "| Budget | Uniform violations | Gradient-only violations | "
        "Bounded violations | Bounded changes |",
        "|---:|---:|---:|---:|---:|",
    ]
    for budget, row in result["policy_sweeps"].items():
        lines.append(
            f"| {budget} | {row['uniform_violations']}/{row['cases']} | "
            f"{row['gradient_violations']}/{row['cases']} | "
            f"{row['bounded_violations']}/{row['cases']} | "
            f"{row['bounded_changes']}/{row['cases']} |"
        )
    lines += [
        "",
        "The separate stationary-point counterexample has zero local gradient and "
        "an actual zeroing error of 2; bounded selection rejects it at budget 0.01. "
        "Zero violations on the random fixtures do not prove the floating-point allowance.",
        "",
        "Largest-state payload reductions are not RSS reductions. The suite uses constructed "
        "weak bridges, an unpinned heterogeneous CPU, forced worker counts, and repeated fixed "
        "bindings. No general VQE trajectory or multi-host claim follows. See docs/SENSITIVITY.md "
        "for the derivation and timing scope. Frozen sources accompany the raw artifacts.",
    ]
    (BASE / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
