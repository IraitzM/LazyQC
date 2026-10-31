#!/usr/bin/env python3
"""Frozen exact-cut/batch validation with B0-B5, ablations and paired raw timings."""

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import (
    bootstrap_interval,
    dense_expectations,
    pin_single_thread,
    positive_int,
)

ROOT = Path(__file__).resolve().parents[1]
MODES = (
    "B0_full",
    "B1_cones",
    "B2_fixed_zx_cut",
    "B3_independent",
    "B4_tensor",
    "B5_joint",
    "B3_cached",
    "B5_cached",
    "no_sharing",
    "no_zx",
    "full_union",
    "heuristic",
    "workers2_cached",
)


def build(case, repetition):
    from lazyqc.fixtures import build_requests
    from lazyqc.ir import Program

    requests = build_requests(case)
    if "binding_deltas" not in case:
        return requests
    delta = case["binding_deltas"][max(0, repetition) % len(case["binding_deltas"])]
    p = requests[0].program
    p = Program(
        p.qubits,
        tuple(
            replace(g, angle=g.angle + delta) if g.angle is not None else g for g in p.operations
        ),
    )
    return tuple(p.expectation(r.observable) for r in requests)


def summarize(rows):
    result = []
    for name in dict.fromkeys(r["case"] for r in rows):
        group = [r for r in rows if r["case"] == name]
        metrics = {}
        for mode in dict.fromkeys(r["mode"] for r in group):
            success = [r for r in group if r["mode"] == mode and r["status"] == "ok"]
            failures = [r for r in group if r["mode"] == mode and r["status"] != "ok"]
            metrics[mode] = {"successful_runs": len(success), "infeasible_runs": len(failures)}
            if success:
                metrics[mode].update(
                    {
                        "median_total_ns": median(r["total_ns"] for r in success),
                        "median_planning_ns": median(r["planning_ns"] for r in success),
                        "median_execution_ns": median(r["execution_ns"] for r in success),
                        "max_abs_error": max(r["max_abs_error"] for r in success),
                        "unique_experiments": sorted(
                            {r["unique_experiments"] for r in success if "unique_experiments" in r}
                        ),
                    }
                )
        comparisons = {}
        for base, proposed in (
            ("B3_independent", "B5_joint"),
            ("B3_cached", "B5_cached"),
            ("B0_full", "B5_cached"),
            ("B4_tensor", "B5_joint"),
            ("B5_cached", "workers2_cached"),
        ):
            a = {r["repetition"]: r for r in group if r["mode"] == base and r["status"] == "ok"}
            b = {r["repetition"]: r for r in group if r["mode"] == proposed and r["status"] == "ok"}
            paired = sorted(a.keys() & b.keys())
            if paired:
                ratios = [a[i]["total_ns"] / b[i]["total_ns"] for i in paired]
                comparisons[f"{base}/{proposed}"] = {
                    "paired_speedups": ratios,
                    "geomean": math.exp(sum(map(math.log, ratios)) / len(ratios)),
                    "bootstrap_95_percent_interval": bootstrap_interval(ratios),
                    "regression_fraction": sum(r < 1 for r in ratios) / len(ratios),
                }
        result.append({"case": name, "metrics": metrics, "comparisons": comparisons})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/p4_p5.json")
    parser.add_argument("--repeats", type=positive_int, default=7)
    parser.add_argument("--calibration-repeats", type=positive_int, default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/p4_p5")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    args = parser.parse_args()
    pin_single_thread()
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/lazyqc-numba")
    from lazyqc import CutOptions, NoFeasiblePlan, PlanningCache, calibrate_batch, plan_all
    from lazyqc.cost import environment
    from lazyqc.fixtures import load_fixture
    from lazyqc.frontends import from_rhyme_ir

    fixture, cases = load_fixture(args.fixture)
    if args.cases:
        if set(args.cases) - {c["name"] for c in cases}:
            raise ValueError("Unknown case")
        cases = [c for c in cases if c["name"] in args.cases]
    tolerance = fixture["absolute_tolerance"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Tolerance must be finite and positive")
    print("Frozen manifest loaded; calibrating instrument execution separately...", flush=True)
    model, training = calibrate_batch(widths=(1, 2, 3, 4, 6, 8), repeats=args.calibration_repeats)
    import_times = {}
    for module in ("pyzx", "qiskit_addon_cutting", "quimb.tensor"):
        start = perf_counter_ns()
        __import__(module)
        import_times[module] = perf_counter_ns() - start
    # Prime the decomposition import, without evolving a state or warming case plans.
    from lazyqc.cutting import cx_branches

    cx_branches()
    first, rows, explanations, oracles = [], [], {}, {}
    cache_info = {}
    rng = random.Random(9201)
    for case in cases:
        caches = {mode: PlanningCache(8) for mode in args.modes if mode.endswith("cached")}
        opts = CutOptions(
            max_fragment_qubits=case["fragment_width"],
            max_cuts=case["max_cuts"],
            **fixture["cut_limits"],
        )
        expected = {}
        for repetition in range(-1, args.repeats):
            key = max(0, repetition) % len(case.get("binding_deltas", [0]))
            if key not in expected:
                expected[key] = dense_expectations(build(case, repetition))
            modes = list(args.modes)
            rng.shuffle(modes)
            for mode in modes:
                # Fresh process startup is deliberately tested on a small and a larger batch.
                if mode == "workers2_cached" and case["name"] not in (
                    "bell_1q_fragments",
                    "overlapping_uncut_batch16",
                ):
                    continue
                start = perf_counter_ns()
                requests = build(case, repetition)
                built = perf_counter_ns()
                row = {
                    "case": case["name"],
                    "mode": mode,
                    "repetition": repetition,
                    "binding_index": key,
                    "construction_ns": built - start,
                }
                try:
                    if mode == "B4_tensor":
                        from tensor_baseline import prepare

                        plan = prepare(requests)
                        planned = perf_counter_ns()
                        values = plan.collect()
                        finished = perf_counter_ns()
                        explanation = {
                            "representation": "quimb local expectation TN",
                            "contraction_width": plan.contraction_width,
                            "shared_intermediates": True,
                            "constraint_scope": "unconstrained tensor comparator",
                        }
                    else:
                        kwargs = {}
                        if mode in ("B0_full", "B1_cones"):
                            kwargs = {
                                "strategy": "full" if mode == "B0_full" else "separate",
                                "optimization": "local",
                            }
                            if mode == "B1_cones":
                                kwargs["max_state_qubits"] = case["fragment_width"]
                        else:
                            selected_opts = replace(
                                opts,
                                search="fixed"
                                if mode == "B2_fixed_zx_cut"
                                else "independent"
                                if mode in ("B3_independent", "B3_cached", "no_sharing")
                                else "joint",
                                reuse=mode != "no_sharing",
                            )
                            kwargs = {
                                "cutting": selected_opts,
                                "optimization": "none" if mode == "no_zx" else "zx",
                                "strategy": "full" if mode == "full_union" else "auto",
                                "cost_model": None if mode == "heuristic" else model,
                                "workers": 2 if mode == "workers2_cached" else 1,
                                "max_state_qubits": case["fragment_width"],
                                "max_state_bytes": 16 * 2 ** case["fragment_width"],
                                "total_state_bytes": 32 * 2 ** case["fragment_width"],
                                "cache": caches.get(mode),
                            }
                        plan = plan_all(requests, **kwargs)
                        planned = perf_counter_ns()
                        result = plan.collect()
                        finished = perf_counter_ns()
                        values = result.values
                        row.update(asdict(result.stats))
                        row.update(
                            {
                                "unique_experiments": len(plan.selected.tasks),
                                "selected_candidate": plan.selected.label,
                                "cache_status": plan.cache_status,
                                "optimization_ns": plan.optimization_ns,
                            }
                        )
                        if hasattr(plan, "cutting"):
                            row.update(
                                {
                                    "cut_count_across_groups": plan.selected.cut_count,
                                    "reconstruction_terms": plan.selected.term_count,
                                    "predicted_execution_ns": plan.selected.score
                                    if mode != "heuristic"
                                    else None,
                                    "search_states": plan.search_states,
                                    "search_exhausted": plan.search_exhausted,
                                }
                            )
                        explanation = plan.explain()
                    error = max(abs(a - b) for a, b in zip(values, expected[key], strict=True))
                    if not all(math.isfinite(v) for v in values) or error > tolerance:
                        raise AssertionError(f"Incorrect {case['name']}/{mode}: {error}")
                    row.update(
                        {
                            "status": "ok",
                            "max_abs_error": error,
                            "values": values,
                            "planning_ns": planned - built,
                            "execution_ns": finished - planned,
                            "total_ns": finished - start,
                        }
                    )
                    if repetition == -1:
                        explanations[f"{case['name']}/{mode}"] = explanation
                except NoFeasiblePlan as exc:
                    row.update(
                        {
                            "status": "infeasible_within_limits",
                            "reason": str(exc),
                            "total_ns": perf_counter_ns() - start,
                        }
                    )
                (first if repetition == -1 else rows).append(row)
        oracles[case["name"]] = expected
        cache_info[case["name"]] = {m: asdict(c.info()) for m, c in caches.items()}
        print(f"Validated {case['name']}; infeasible modes retained as failures", flush=True)
    mark = perf_counter_ns()
    imported = from_rhyme_ir(json.loads((ROOT / "benchmarks/rhyme_qutrit.json").read_text()))
    frontend_ns = perf_counter_ns() - mark
    queries = [imported.probability("level", v) for v in (10, 20, 30)]
    queries.extend(imported.invalid_probabilities("level"))
    decoded = [
        q.evaluate(plan_all(q.requests, cutting=CutOptions(max_cuts=0)).collect().values)
        for q in queries
    ]
    if max(abs(a - b) for a, b in zip(decoded, (0.5, 0.25, 0.25, 0), strict=True)) > tolerance:
        raise AssertionError("Logical probability/leakage validation failed")
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "versions": {
            p: version(p)
            for p in ("qulacs", "pyzx", "qiskit-addon-cutting", "quimb", "cotengra", "autoray")
        },
        "manifest_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/lazyqc").rglob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_id": model.profile_id,
        "import_ns": import_times,
        "repeats": args.repeats,
        "order_seed": 9201,
        "bootstrap_seed": 9701,
        "timing_scope": "request construction + planning + execution; imports/calibration separate",
        "interval_scope": "paired repetitions in one session; not between-machine uncertainty",
        "constraints": "B0 unconstrained dense; B4 unconstrained TN; B1-B3/B5 bounded fragments",
        "memory_scope": "raw live state payload; worker RSS is a process lifetime high-water mark",
        "numerical_cache": False,
    }
    output = {
        "metadata": metadata,
        "cases": cases,
        "first_calls": first,
        "runs": rows,
        "plans": explanations,
        "oracles": oracles,
        "cache_info": cache_info,
        "summary": summarize(rows),
        "frontend": {
            "compile_ns": frontend_ns,
            "source_sha256": imported.source_sha256,
            "decoded_probabilities_and_leakage": decoded,
            "encoded_qubits": imported.program.n_qubits,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "profile.json")
    (args.output_dir / "calibration.json").write_text(json.dumps(training, indent=2) + "\n")
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(f"Results: {args.output_dir / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
