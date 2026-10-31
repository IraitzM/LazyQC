#!/usr/bin/env python3
"""Screen independent-occurrence and layer-tied parameter-shift Jacobians."""

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from benchmark_support import cpu_name, pin_single_thread, source_hash
from qml_workload_support import (
    layer_ry_occurrence_groups,
    make_batch,
    numpy_oracle_expectations,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "tied_qml_gradient_manifest.json"


def with_group_shift(program, indices, shift):
    from lazyqc import Gate, Program

    operations = list(program.operations)
    for index in indices:
        gate = operations[index]
        operations[index] = Gate(gate.name, gate.qubits, gate.angle + shift)
    return Program(program.n_qubits, operations)


def structural_zero_counts(program, observables, indices, groups):
    from lazyqc.passes import cone_indices

    qubits = tuple(g.qubits for g in program.operations)
    independent_zeros = tied_zeros = 0
    for observable in observables:
        _, retained = cone_indices(qubits, (q for q, _ in observable.factors))
        retained = set(retained)
        independent_zeros += len(set(indices) - retained)
        tied_zeros += sum(not (set(group) & retained) for group in groups)
    return independent_zeros, tied_zeros


def write(path, metadata, summaries, runs, checks, failures):
    payload = {
        "metadata": metadata,
        "summary": summaries,
        "runs": runs,
        "gradient_checks": checks,
        "failures": failures,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run one 8-qubit case")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    config = json.loads(manifest_bytes)
    pin_single_thread()

    from lazyqc import Pauli, plan_all
    from lazyqc.backends.qulacs import QulacsBackend

    from direct_support import assemble_jacobian, jacobian_requests, shifted_programs

    # Initialize Qulacs before collecting timings.
    warm = make_batch("hardware_efficient_local", 1, 1, "same_state_local_outputs", 7)
    plan_all(warm, strategy="full").collect(backend=QulacsBackend())

    families = config["families"][:1] if args.smoke else config["families"]
    qubits = config["qubits"][:1] if args.smoke else config["qubits"]
    depths = config["depths"][:1] if args.smoke else config["depths"]
    offsets = config["seed_offsets"][:1] if args.smoke else config["seed_offsets"]
    strategies = config["strategies"]
    order_rng = random.Random(config["randomized_order_seed"])
    output = args.output or ROOT / "results/qml_workloads/tied_gradients_v1.json"
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "exploratory local gradient screen; not confirmatory",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_sha256": source_hash(__file__),
        "workload_support_sha256": hashlib.sha256(
            (ROOT / "benchmarks/qml_workload_support.py").read_bytes()
        ).hexdigest(),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "python": sys.version,
        "backend": "QulacsBackend, single native thread",
        "command": sys.argv,
        "smoke": args.smoke,
    }
    summaries, runs, checks, failures = [], [], [], []
    repeats = config["collections_per_plan"]
    epsilon = config["finite_difference_epsilon"]

    for family in families:
        for n in qubits:
            for depth in depths:
                for offset in offsets:
                    seed = config["base_seed"] + offset
                    program = make_batch(
                        family, n, depth, "same_state_local_outputs", seed
                    )[0].program
                    observables = [Pauli(((q, "Z"),)) for q in range(n)]
                    groups = layer_ry_occurrence_groups(family, n, depth)
                    indices = tuple(index for group in groups for index in group)
                    parameter_column = {index: column for column, index in enumerate(indices)}
                    if not indices or sorted(indices) != list(indices):
                        failures.append(
                            {"family": family, "qubits": n, "depth": depth,
                             "seed_offset": offset, "stage": "parameter-map",
                             "error": "invalid or empty RY occurrence map"}
                        )
                        continue
                    programs = shifted_programs(program, indices)
                    requests = jacobian_requests(programs, observables)
                    expected = np.asarray(numpy_oracle_expectations(requests))
                    independent_zeros, tied_zeros = structural_zero_counts(
                        program, observables, indices, groups
                    )
                    case = {
                        "case_id": f"{family}_{n}q_d{depth}_tied_gradient_s{offset}",
                        "family": family,
                        "qubits": n,
                        "depth": depth,
                        "seed": seed,
                        "seed_offset": offset,
                        "outputs": len(observables),
                        "independent_parameters": len(indices),
                        "tied_layer_parameters": len(groups),
                        "shifted_programs": len(programs),
                        "requests": len(requests),
                        "independent_structural_zero_fraction": independent_zeros
                        / (n * len(indices)),
                        "tied_structural_zero_fraction": tied_zeros / (n * len(groups)),
                    }
                    plans, planning_ns = {}, {}
                    try:
                        for strategy in strategies:
                            start = perf_counter_ns()
                            plans[strategy] = plan_all(
                                requests,
                                strategy=strategy,
                                optimization="none",
                                max_state_qubits=config["max_state_qubits"],
                                max_state_bytes=config["max_state_bytes"],
                            )
                            planning_ns[strategy] = perf_counter_ns() - start
                    except Exception as exc:
                        failures.append({**case, "stage": "planning", "error": repr(exc)})
                        write(output, metadata, summaries, runs, checks, failures)
                        continue

                    samples = {strategy: [] for strategy in strategies}
                    for repetition in range(repeats):
                        order = list(strategies)
                        order_rng.shuffle(order)
                        for strategy in order:
                            plan = plans[strategy]
                            try:
                                start = perf_counter_ns()
                                result = plan.collect(backend=QulacsBackend())
                                collection_ns = perf_counter_ns() - start
                                values = np.asarray(result.values)
                                error = float(np.max(np.abs(values - expected)))
                                if not np.isfinite(error) or error > config["oracle_tolerance"]:
                                    failures.append(
                                        {**case, "strategy": strategy,
                                         "repetition": repetition, "stage": "oracle",
                                         "max_abs_error": error}
                                    )
                                    continue
                                forward, jacobian = assemble_jacobian(
                                    values, len(indices), n
                                )
                                tied_jacobian = np.column_stack(
                                    [
                                        jacobian[
                                            :,
                                            [parameter_column[index] for index in group],
                                        ].sum(axis=1)
                                        for group in groups
                                    ]
                                )
                                record = {
                                    **case,
                                    "strategy": strategy,
                                    "selected_strategy": result.selected_strategy,
                                    "repetition": repetition,
                                    "planning_ns": planning_ns[strategy],
                                    "collection_ns": collection_ns,
                                    "evolutions": result.stats.evolutions,
                                    "task_count": len(plan.selected.tasks),
                                    "task_widths": [t.width for t in plan.selected.tasks],
                                    "max_abs_error": error,
                                    "forward_values": np.asarray(forward).tolist(),
                                    "independent_jacobian": jacobian.tolist(),
                                    "layer_tied_jacobian": tied_jacobian.tolist(),
                                }
                                runs.append(record)
                                samples[strategy].append(record)
                            except Exception as exc:
                                failures.append(
                                    {**case, "strategy": strategy,
                                     "repetition": repetition, "stage": "collection",
                                     "error": repr(exc)}
                                )

                    if samples["auto"]:
                        reference = np.asarray(samples["auto"][0]["layer_tied_jacobian"])
                        for group_index, group in enumerate(groups):
                            plus = with_group_shift(program, group, epsilon)
                            minus = with_group_shift(program, group, -epsilon)
                            fd_requests = jacobian_requests([plus, minus], observables)
                            fd = np.asarray(numpy_oracle_expectations(fd_requests)).reshape(2, n)
                            finite_difference = (fd[0] - fd[1]) / (2 * epsilon)
                            check_error = float(
                                np.max(np.abs(finite_difference - reference[:, group_index]))
                            )
                            checks.append(
                                {
                                    **case,
                                    "group_index": group_index,
                                    "occurrences": list(group),
                                    "epsilon": epsilon,
                                    "max_abs_finite_difference_error": check_error,
                                }
                            )
                            if check_error > config["finite_difference_tolerance"]:
                                failures.append(
                                    {**case, "stage": "finite-difference",
                                     "group_index": group_index,
                                     "max_abs_error": check_error}
                                )

                    for strategy, records in samples.items():
                        if not records:
                            continue
                        collections = [r["collection_ns"] for r in records]
                        summaries.append(
                            {
                                **case,
                                "strategy": strategy,
                                "median_planning_ms": median(
                                    r["planning_ns"] for r in records
                                ) / 1e6,
                                "median_collection_ms": median(collections) / 1e6,
                                "two_collection_lifetime_ms": (
                                    planning_ns[strategy] + sum(collections)
                                ) / 1e6,
                                "evolutions": records[0]["evolutions"],
                                "task_count": records[0]["task_count"],
                                "task_widths": records[0]["task_widths"],
                                "successful_collections": len(records),
                            }
                        )
                    write(output, metadata, summaries, runs, checks, failures)
                    print(
                        f"finished {case['case_id']} ({len(failures)} failures)",
                        flush=True,
                    )

    metadata["case_count"] = len({row["case_id"] for row in summaries})
    metadata["repeats"] = repeats
    write(output, metadata, summaries, runs, checks, failures)
    print(
        f"cases={metadata['case_count']} rows={len(runs)} "
        f"gradient_checks={len(checks)} failures={len(failures)} output={output}"
    )
    if failures:
        raise SystemExit(f"{len(failures)} failures; inspect the recorded artifact")


if __name__ == "__main__":
    main()
