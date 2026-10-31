#!/usr/bin/env python3
"""Repeat a matched two-output cone-sharing comparison on IQM Garnet."""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from qml_workload_support import make_batch, numpy_oracle_expectations  # noqa: E402
from run_iqm_qml_pilot import expectation_from_counts, json_safe, lower_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output-qubits", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--seed-offset", type=int, default=202)
    parser.add_argument("--order-seed", type=int, default=1907)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--fixed-layout",
        action="store_true",
        help="Anchor logical wires to the same-index physical Garnet qubits.",
    )
    args = parser.parse_args()
    if args.shots < 1 or args.repetitions < 1:
        parser.error("--shots and --repetitions must be positive")

    from iqm.qiskit_iqm import IQMProvider
    from lazyqc import plan_all
    from qiskit import transpile

    manifest = json.loads((ROOT / "benchmarks/qml_workload_manifest.json").read_text())
    cfg = manifest["large_cone_probe"]
    family, n, depth = "hardware_efficient_local", 20, cfg["depth"]
    seed = manifest["seed"] + args.seed_offset
    all_requests = make_batch(family, n, depth, "same_state_local_outputs", seed)
    if len(set(args.output_qubits)) != 2 or any(q < 0 or q >= n for q in args.output_qubits):
        parser.error("--output-qubits must be two distinct indices in [0, 19]")
    requests = tuple(all_requests[q] for q in args.output_qubits)
    ideal = numpy_oracle_expectations(requests)

    provider = IQMProvider(url="https://resonance.iqm.tech", quantum_computer="garnet")
    backend = provider.get_backend()
    strategies = ("full", "separate", "grouped")
    prepared = {}
    for strategy in strategies:
        plan = plan_all(
            requests,
            strategy=strategy,
            optimization="none",
            max_state_qubits=cfg["max_state_qubits"],
            max_state_bytes=cfg["max_state_bytes"],
        ).selected
        circuits, circuit_metadata = [], []
        for index, task in enumerate(plan.tasks):
            circuit, _ = lower_task(task)
            transpile_options = {
                "backend": backend,
                "optimization_level": 1,
                "seed_transpiler": 508,
            }
            if args.fixed_layout:
                transpile_options["initial_layout"] = list(range(circuit.num_qubits))
            compiled = transpile(circuit, **transpile_options)
            ops = compiled.count_ops()
            measurement_layout = {
                compiled.find_bit(instruction.clbits[0]).index: compiled.find_bit(
                    instruction.qubits[0]
                ).index
                for instruction in compiled.data
                if instruction.operation.name == "measure"
            }
            circuits.append(compiled)
            circuit_metadata.append(
                {
                    "task_index": index,
                    "logical_width": task.width,
                    "logical_wires": list(task.wires),
                    "output_qubits": [obs.factors[0][0] for _, obs in task.outputs],
                    "measurement_physical_qubits_by_cbit": {
                        str(key): value for key, value in measurement_layout.items()
                    },
                    "compiled_width": compiled.num_qubits,
                    "compiled_depth": compiled.depth(),
                    "compiled_operations": {str(k): int(v) for k, v in ops.items()},
                    "compiled_two_qubit_gates": sum(
                        int(v)
                        for k, v in ops.items()
                        if str(k).lower() in {"cz", "cx", "ecr", "iswap", "rxx", "rzz", "swap"}
                    ),
                }
            )
        prepared[strategy] = {
            "circuits": circuits,
            "tasks": plan.tasks,
            "circuit_metadata": circuit_metadata,
        }

    if args.preflight:
        for strategy in strategies:
            item = prepared[strategy]
            print(
                strategy,
                "circuits=", len(item["circuits"]),
                "logical_widths=", sorted({m["logical_width"] for m in item["circuit_metadata"]}),
                "compiled_depths=", [m["compiled_depth"] for m in item["circuit_metadata"]],
                "compiled_2q_gates=", sum(
                    m["compiled_two_qubit_gates"] for m in item["circuit_metadata"]
                ),
            )
        print("preflight only; no jobs submitted")
        return
    if not args.submit:
        parser.error("pass --submit to run QPU jobs, or --preflight to stop before submission")

    output = args.output or ROOT / "results/qml_workloads/iqm_garnet_pair_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not output.exists():
            parser.error("--resume requires an existing checkpoint file")
        previous = json.loads(output.read_text())
        previous_meta = previous.get("metadata", {})
        expected = (
            args.shots,
            args.repetitions,
            list(args.output_qubits),
            seed,
            args.fixed_layout,
        )
        found = (
            previous_meta.get("shots_per_circuit"),
            previous_meta.get("repetitions"),
            previous_meta.get("output_qubits"),
            previous_meta.get("seed"),
            previous_meta.get("fixed_layout", False),
        )
        if found != expected:
            parser.error(f"checkpoint parameters {found} do not match requested {expected}")
        runs = previous.get("runs", [])
    else:
        runs = []

    rng = random.Random(args.order_seed)
    repetition_orders = []
    for _ in range(args.repetitions):
        order = list(strategies)
        rng.shuffle(order)
        repetition_orders.append(order)
    completed = {(row["repetition"], row["strategy"]) for row in runs}
    for repetition in range(args.repetitions):
        order = repetition_orders[repetition]
        for strategy in order:
            if (repetition, strategy) in completed:
                continue
            item = prepared[strategy]
            submitted_at = datetime.now(timezone.utc).isoformat()
            start = perf_counter()
            job = backend.run(item["circuits"], shots=args.shots)
            job_id = str(job.job_id())
            result = job.result()
            wall_seconds = perf_counter() - start
            counts = result.get_counts()
            if isinstance(counts, dict):
                counts = [counts]
            outputs = []
            for task, task_counts in zip(item["tasks"], counts, strict=True):
                for classical_bit, (_, observable) in enumerate(task.outputs):
                    q, pauli = observable.factors[0]
                    value = expectation_from_counts(
                        task_counts, classical_bit, len(task.outputs)
                    )
                    outputs.append(
                        {
                            "qubit": q,
                            "pauli": pauli,
                            "expectation": value,
                            "ideal_expectation": ideal[args.output_qubits.index(q)],
                            "binomial_standard_error": (1.0 - value * value) ** 0.5
                            / (args.shots**0.5),
                        }
                    )
            runs.append(
                {
                    "repetition": repetition,
                    "strategy": strategy,
                    "job_id": job_id,
                    "submitted_at_utc": submitted_at,
                    "shots_per_circuit": args.shots,
                    "circuit_count": len(item["circuits"]),
                    "total_shots": len(item["circuits"]) * args.shots,
                    "submit_to_result_seconds": wall_seconds,
                    "timeline": json_safe(getattr(result, "timeline", None)),
                    "circuits": item["circuit_metadata"],
                    "outputs": outputs,
                    "counts": json_safe(counts),
                }
            )
            completed.add((repetition, strategy))
            checkpoint = {
                "metadata": {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "backend": str(backend.name),
                    "family": family,
                    "qubits": n,
                    "depth": depth,
                    "seed": seed,
                    "output_qubits": list(args.output_qubits),
                    "fixed_layout": args.fixed_layout,
                    "shots_per_circuit": args.shots,
                    "repetitions": args.repetitions,
                    "repetition_orders": repetition_orders,
                    "submitted": True,
                    "total_jobs": len(runs),
                    "total_circuit_shots": sum(row["total_shots"] for row in runs),
                    "account_cost_quote_available": False,
                    "status": "in_progress",
                },
                "runs": runs,
            }
            output.write_text(json.dumps(checkpoint, indent=2) + "\n")

    summaries = {}
    for strategy in strategies:
        strategy_runs = [row for row in runs if row["strategy"] == strategy]
        execution_windows = []
        for row in strategy_runs:
            def timestamp(status):
                event = next(
                    event
                    for event in row["timeline"]
                    if event["source"] == "iqm-station-control"
                    and event["status"] == status
                )
                return event["timestamp"]

            from datetime import datetime as dt
            import re

            def parse(value):
                parts = re.search(
                    r"\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)",
                    value,
                )
                return dt(*map(int, parts.groups()), tzinfo=timezone.utc)

            execution_windows.append(
                (
                    parse(timestamp("execution_ended"))
                    - parse(timestamp("execution_started"))
                ).total_seconds()
            )
        output_values = {
            q: [
                output["expectation"]
                for row in strategy_runs
                for output in row["outputs"]
                if output["qubit"] == q
            ]
            for q in args.output_qubits
        }
        summaries[strategy] = {
            "jobs": len(strategy_runs),
            "circuits_per_job": strategy_runs[0]["circuit_count"],
            "total_shots": sum(row["total_shots"] for row in strategy_runs),
            "execution_window_seconds": execution_windows,
            "mean_execution_window_seconds": mean(execution_windows),
            "sample_sd_execution_window_seconds": (
                stdev(execution_windows) if len(execution_windows) > 1 else None
            ),
            "per_output_expectation_means": {
                str(q): mean(values) for q, values in output_values.items()
            },
            "per_output_expectation_sd": {
                str(q): stdev(values) if len(values) > 1 else None
                for q, values in output_values.items()
            },
            "compiled_circuits": strategy_runs[0]["circuits"],
        }

    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": str(backend.name),
            "family": family,
            "qubits": n,
            "depth": depth,
            "seed": seed,
            "output_qubits": list(args.output_qubits),
            "fixed_layout": args.fixed_layout,
            "shots_per_circuit": args.shots,
            "repetitions": args.repetitions,
            "repetition_orders": repetition_orders,
            "submitted": True,
            "total_jobs": len(runs),
            "total_circuit_shots": sum(row["total_shots"] for row in runs),
            "account_cost_quote_available": False,
        },
        "summary": summaries,
        "runs": runs,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for strategy, summary in summaries.items():
        print(
            strategy,
            "jobs=", summary["jobs"],
            "shots=", summary["total_shots"],
            "execution_mean_s=", round(summary["mean_execution_window_seconds"], 4),
            "execution_sd_s=", round(summary["sample_sd_execution_window_seconds"], 4),
        )
    print("output=", output)


if __name__ == "__main__":
    main()
