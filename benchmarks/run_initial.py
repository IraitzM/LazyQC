#!/usr/bin/env python3
"""Run exact expectations with fresh plans/states; save all timing repetitions."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from benchmark_support import cpu_name, pair_case, percentile, positive_int, source_hash

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixture", type=Path, default=ROOT / "tests/fixtures/initial.json")
    p.add_argument("--pairs", help="Generate pair cases instead, e.g. 1,2,4,6,8")
    p.add_argument("--strategies", default="full,separate,grouped,auto")
    p.add_argument("--repeats", type=positive_int, default=20)
    p.add_argument("--threads", type=positive_int, default=1)
    p.add_argument("--output", type=Path, default=ROOT / "results/initial.json")
    return p


def main():
    args = parser().parse_args()
    thread_keys = (
        "QULACS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    for key in thread_keys:
        os.environ[key] = str(args.threads)

    import numpy as np
    import qulacs  # noqa: F401 -- import time is excluded from measurements

    from lazyqc import Program, plan_all
    from lazyqc.backends.qulacs import QulacsBackend
    from lazyqc.fixtures import build_requests, load_fixture
    from lazyqc.planner import STRATEGIES

    strategies = args.strategies.split(",")
    if (
        not strategies
        or len(set(strategies)) != len(strategies)
        or any(s not in STRATEGIES for s in strategies)
    ):
        raise ValueError(f"Choose distinct strategies from {STRATEGIES}")
    fixture, cases = load_fixture(args.fixture)
    if args.pairs:
        sizes = [int(n) for n in args.pairs.split(",")]
        if len(set(sizes)) != len(sizes):
            raise ValueError("Pair sizes must be distinct")
        cases = [pair_case(n) for n in sizes]
    tolerance = fixture["absolute_tolerance"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Fixture tolerance must be finite and positive")
    rows, first_calls, plans = [], [], {}
    timing_fields = [
        "construction_ns",
        "planning_ns",
        "lowering_ns",
        "evolution_ns",
        "expectation_ns",
        "assembly_ns",
        "execution_ns",
        "total_ns",
    ]

    def run(case, strategy, repetition):
        expected = np.array([r["expected"] for r in case["requests"]])
        backend = QulacsBackend()
        start = perf_counter_ns()
        requests = build_requests(case)
        built = perf_counter_ns()
        plan = plan_all(requests, strategy=strategy)
        planned = perf_counter_ns()
        if backend.counters.allocations or backend.counters.evolutions:
            raise AssertionError("Execution occurred before collect")
        result = plan.collect(backend=backend)
        finished = perf_counter_ns()
        error = float(np.max(np.abs(np.array(result.values) - expected)))
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError(f"{case['name']}/{strategy}: error {error}")
        if case["name"].startswith("independent_pairs_"):
            counts = {"full": 1, "separate": len(requests), "grouped": case["qubits"] // 2}
            chosen = result.selected_strategy
            width = case["qubits"] if chosen == "full" else 2
            if result.stats.evolutions != counts[chosen] or result.stats.max_state_qubits != width:
                raise AssertionError("Unexpected pair execution structure")
        row = {
            "case": case["name"],
            "strategy": strategy,
            "selected_strategy": result.selected_strategy,
            "repetition": repetition,
            "request_count": len(requests),
            "max_abs_error": error,
            "construction_ns": built - start,
            "planning_ns": planned - built,
            **asdict(result.stats),
            "total_ns": finished - start,
        }
        return row, plan.explain()

    warm = Program(1).h(0)
    plan_all([warm.expectation({0: "Z"})], strategy="full").collect()
    rng = random.Random(0)
    for case in cases:
        first_order = list(strategies)
        rng.shuffle(first_order)
        for strategy in first_order:
            row, explanation = run(case, strategy, -1)
            first_calls.append(row)
            plans[f"{case['name']}/{strategy}"] = explanation
        for repetition in range(args.repeats):
            order = list(strategies)
            rng.shuffle(order)
            for strategy in order:
                row, _ = run(case, strategy, repetition)
                rows.append(row)

    summaries = []
    for case in cases:
        for strategy in strategies:
            subset = [r for r in rows if r["case"] == case["name"] and r["strategy"] == strategy]
            summary = {
                "case": case["name"],
                "strategy": strategy,
                "selected_strategy": subset[0]["selected_strategy"],
                "evolutions": subset[0]["evolutions"],
                "max_state_qubits": subset[0]["max_state_qubits"],
                "estimated_state_bytes": subset[0]["estimated_state_bytes"],
                "max_abs_error": max(r["max_abs_error"] for r in subset),
                "median_ns": {key: median(r[key] for r in subset) for key in timing_fields},
                "total_q25_ns": percentile([r["total_ns"] for r in subset], 0.25),
                "total_q75_ns": percentile([r["total_ns"] for r in subset], 0.75),
            }
            summaries.append(summary)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "packages": {
            p: importlib.metadata.version(p) for p in ("lazyqc", "qulacs", "numpy")
        },
        "threads": {k: os.environ[k] for k in thread_keys},
        "source_sha256": source_hash(__file__),
        "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "cases_sha256": hashlib.sha256(json.dumps(cases, sort_keys=True).encode()).hexdigest(),
        "command": sys.argv,
        "repeats": args.repeats,
        "order_seed": 0,
        "excluded_from_total": ["imports", "JSON parsing", "oracle checks", "report generation"],
        "included_in_total": [
            "construction",
            "planning",
            "lowering",
            "fresh evolution",
            "normalization check",
            "expectations",
            "assembly",
        ],
        "memory_scope": "estimated raw state payload, not process RSS",
        "first_calls": "first case/strategy executions after one untimed one-qubit warmup",
    }
    output = {
        "metadata": metadata,
        "cases": cases,
        "summary": summaries,
        "first_calls": first_calls,
        "runs": rows,
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("case  strategy(selected)  total median [IQR] ms  evolutions  max qubits  error")
    for s in summaries:
        print(
            f"{s['case']}  {s['strategy']}({s['selected_strategy']})  "
            f"{s['median_ns']['total_ns'] / 1e6:.3f} "
            f"[{s['total_q25_ns'] / 1e6:.3f}, {s['total_q75_ns'] / 1e6:.3f}]  "
            f"{s['evolutions']}  {s['max_state_qubits']}  {s['max_abs_error']:.2e}"
        )
    print(f"Saved {len(rows)} repetitions and {len(first_calls)} first-case runs to {args.output}")


if __name__ == "__main__":
    main()
