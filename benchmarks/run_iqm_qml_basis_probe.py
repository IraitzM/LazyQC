#!/usr/bin/env python3
"""Compare matched-basis IQM full, separate and grouped QML plans."""

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from qml_workload_support import make_batch, numpy_oracle_expectations  # noqa: E402
from run_iqm_qml_pilot import expectation_from_counts, json_safe  # noqa: E402


def make_requests(family, n, depth, seed, output_qubits):
    from lazyqc import Pauli

    program = make_batch(
        family, n, depth, "same_state_local_outputs", seed
    )[0].program
    requests = tuple(
        program.expectation(Pauli(((q, axis),)))
        for axis in ("X", "Z")
        for q in output_qubits
    )
    return requests


def lower_basis_task(task, basis):
    """Lower compatible single-qubit Paulis to a common measurement basis."""
    from qiskit import QuantumCircuit

    outputs = [
        (program_id, observable)
        for program_id, observable in task.outputs
        if len(observable.factors) == 1 and observable.factors[0][1] == basis
    ]
    if not outputs:
        return None
    circuit = QuantumCircuit(task.width, len(outputs))
    mapping = {wire: index for index, wire in enumerate(task.wires)}
    for gate in task.operations:
        qubits = [mapping[wire] for wire in gate.qubits]
        if gate.name == "CX":
            circuit.cx(*qubits)
        elif gate.name in ("RX", "RY", "RZ"):
            getattr(circuit, gate.name.lower())(gate.angle, *qubits)
        elif gate.name in ("H", "X", "Y", "Z"):
            getattr(circuit, gate.name.lower())(*qubits)
        else:
            raise ValueError(f"Unsupported gate in IQM basis probe: {gate.name}")

    measurement_map = []
    for cbit, (program_id, observable) in enumerate(outputs):
        wire, pauli = observable.factors[0]
        local = mapping[wire]
        if pauli == "X":
            circuit.h(local)
        elif pauli != "Z":
            raise ValueError(f"Unsupported single-qubit basis {pauli}")
        circuit.measure(local, cbit)
        measurement_map.append(
            {"program_id": program_id, "qubit": wire, "pauli": pauli, "classical_bit": cbit}
        )
    return circuit, measurement_map


def route_output_tokens(starts, targets, adjacency):
    """Find a terminal adjacent-swap routing that places measured states at targets."""
    from collections import deque

    def path(start, goal, blocked):
        queue = deque([start])
        previous = {start: None}
        while queue:
            here = queue.popleft()
            if here == goal:
                result = []
                while here is not None:
                    result.append(here)
                    here = previous[here]
                return list(reversed(result))
            for neighbor in sorted(adjacency[here]):
                if neighbor in blocked or neighbor in previous:
                    continue
                previous[neighbor] = here
                queue.append(neighbor)
        return None

    for order in (tuple(range(len(starts))), tuple(reversed(range(len(starts))))):
        positions = list(starts)
        swaps, fixed = [], set()
        valid = True
        for token in order:
            blockers = {
                positions[other]
                for other in range(len(starts))
                if other != token and other not in fixed
            } | fixed
            route = path(positions[token], targets[token], blockers)
            if route is None:
                valid = False
                break
            for left, right in zip(route, route[1:]):
                swaps.append((left, right))
            positions[token] = targets[token]
            fixed.add(targets[token])
        if valid:
            return swaps
    raise ValueError(f"Cannot route measured output states from {starts} to {targets}")


def line_layout(adjacency, n, output_qubits):
    """Map logical wire w to a physical path that starts at physical 0, then 1.

    Nearest-neighbour logical CX gates on wires inside the path need no routing,
    so cone circuits embed without SWAPs and every plan reads its outputs on the
    same physical pair. Logical wires beyond the longest such path take the
    remaining physical qubits in index order and are routed by the compiler.
    """
    if tuple(output_qubits) != (0, 1):
        raise ValueError("line placement anchors logical outputs (0, 1) on physical (0, 1)")
    best = []

    def extend(path):
        nonlocal best
        if len(path) > len(best):
            best = list(path)
        if len(best) == n:
            return
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor not in path:
                extend(path + [neighbor])

    extend([0, 1])
    rest = [q for q in range(len(adjacency)) if q not in best]
    return (best + rest)[:n], len(best)


def compile_strategy(
    strategy, requests, backend, shots, repetitions, seed, output_qubits, placement="swap"
):
    from qiskit import transpile
    from lazyqc import plan_all

    plan = plan_all(
        requests,
        strategy=strategy,
        optimization="none",
        max_state_qubits=24,
        max_state_bytes=256 * 2**20,
    ).selected
    circuits, metadata = [], []
    coupling = backend.target.build_coupling_map().get_edges()
    adjacency = [set() for _ in range(backend.num_qubits)]
    for left, right in coupling:
        adjacency[left].add(right)
        adjacency[right].add(left)
    if placement == "line":
        layout, _ = line_layout(adjacency, requests[0].program.n_qubits, output_qubits)
    for task_index, task in enumerate(plan.tasks):
        for basis in ("X", "Z"):
            lowered = lower_basis_task(task, basis)
            if lowered is None:
                continue
            circuit, measurement_map = lowered
            auto_compiled = transpile(
                circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=508,
                initial_layout=(
                    [layout[wire] for wire in task.wires] if placement == "line" else None
                ),
            )
            auto_measurements = {
                auto_compiled.find_bit(instruction.clbits[0]).index: auto_compiled.find_bit(
                    instruction.qubits[0]
                ).index
                for instruction in auto_compiled.data
                if instruction.operation.name == "measure"
            }
            start_physical = [
                auto_measurements[record["classical_bit"]] for record in measurement_map
            ]
            target_physical = [output_qubits.index(record["qubit"]) for record in measurement_map]
            correction_swaps = route_output_tokens(
                start_physical, target_physical, adjacency
            )
            from qiskit import QuantumCircuit

            unitary = auto_compiled.remove_final_measurements(inplace=False)
            corrected = QuantumCircuit(backend.num_qubits, len(measurement_map))
            corrected.compose(unitary, qubits=corrected.qubits, inplace=True)
            for left, right in correction_swaps:
                corrected.swap(left, right)
            for record, target in zip(measurement_map, target_physical, strict=True):
                corrected.measure(target, record["classical_bit"])
            compiled = transpile(
                corrected,
                backend=backend,
                optimization_level=1,
                seed_transpiler=508,
                initial_layout=list(range(backend.num_qubits)),
            )
            ops = compiled.count_ops()
            final_measurements = {
                compiled.find_bit(instruction.clbits[0]).index: compiled.find_bit(
                    instruction.qubits[0]
                ).index
                for instruction in compiled.data
                if instruction.operation.name == "measure"
            }
            measured_physical = [
                final_measurements[record["classical_bit"]] for record in measurement_map
            ]
            expected_physical = target_physical
            for repetition_index in range(repetitions):
                circuits.append(compiled)
                metadata.append(
                    {
                        "repetition_index": repetition_index,
                        "task_index": task_index,
                        "basis": basis,
                        "logical_width": task.width,
                        "logical_wires": list(task.wires),
                        "automatic_measured_physical_qubits": start_physical,
                        "output_correction_swaps": correction_swaps,
                        "outputs": measurement_map,
                        "compiled_width": compiled.num_qubits,
                        "compiled_depth": compiled.depth(),
                        "compiled_operations": {str(k): int(v) for k, v in ops.items()},
                        "automatic_layout_depth": auto_compiled.depth(),
                        "automatic_layout_two_qubit_gates": sum(
                            int(v)
                            for k, v in auto_compiled.count_ops().items()
                            if str(k).lower()
                            in {"cz", "cx", "ecr", "iswap", "rxx", "rzz", "swap"}
                        ),
                        "compiled_two_qubit_gates": sum(
                            int(v)
                            for k, v in ops.items()
                            if str(k).lower()
                            in {"cz", "cx", "ecr", "iswap", "rxx", "rzz", "swap"}
                        ),
                        "measured_physical_qubits": measured_physical,
                        "expected_physical_qubits": expected_physical,
                    }
                )
    randomized = list(zip(circuits, metadata, strict=True))
    random.Random(seed + sum(map(ord, strategy))).shuffle(randomized)
    circuits, metadata = map(list, zip(*randomized, strict=True))
    return plan, circuits, metadata


def historical_seconds_per_circuit_shot():
    reference_files = (
        ROOT / "results/qml_workloads/iqm_garnet_pair_probe.json",
        ROOT / "results/qml_workloads/iqm_garnet_pair_fixed_layout.json",
    )
    ratios = []
    for path in reference_files:
        payload = json.loads(path.read_text())
        seconds = 0.0
        for row in payload["runs"]:
            events = {
                event.get("status"): event.get("timestamp")
                for event in row.get("timeline", [])
                if event.get("source") == "iqm-station-control"
            }
            start, end = events.get("execution_started"), events.get("execution_ended")
            if not start or not end:
                continue
            def parse(value):
                match = re.search(
                    r"\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)",
                    value,
                )
                return datetime(*map(int, match.groups()), tzinfo=timezone.utc)
            seconds += (parse(end) - parse(start)).total_seconds()
        shots = payload["metadata"]["total_circuit_shots"]
        ratios.append(seconds / shots)
    return max(ratios)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--shot-ceiling", type=int, default=6500)
    parser.add_argument("--estimated-qpu-seconds-ceiling", type=float, default=60.0)
    parser.add_argument("--previously-spent-shots", type=int, default=0)
    parser.add_argument("--previously-submitted-jobs", type=int, default=0)
    parser.add_argument("--seed-offsets", nargs="+", type=int, default=(202, 303))
    parser.add_argument("--output-qubits", nargs=2, type=int, default=(0, 1))
    parser.add_argument("--order-seed", type=int, default=241028)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--placement",
        choices=("swap", "line"),
        default="swap",
        help="swap: automatic layout plus terminal output SWAPs; line: shared line embedding",
    )
    args = parser.parse_args()
    if args.shots < 1 or args.repetitions < 1:
        parser.error("shots and repetitions must be positive")
    if args.preflight == args.submit:
        parser.error("choose exactly one of --preflight or --submit")

    from iqm.qiskit_iqm import IQMProvider
    from qiskit import transpile

    import qiskit  # noqa: F401 - record the local compile stack

    manifest = json.loads((ROOT / "benchmarks/qml_workload_manifest.json").read_text())
    cfg = manifest["large_cone_probe"]
    family, n, depth = "hardware_efficient_local", 20, cfg["depth"]
    output_qubits = tuple(args.output_qubits)
    if len(set(output_qubits)) != 2 or any(q < 0 or q >= n for q in output_qubits):
        parser.error("--output-qubits must be two distinct indices in [0, 19]")

    provider = IQMProvider(url="https://resonance.iqm.tech", quantum_computer="garnet")
    backend = provider.get_backend()
    strategies = ("full", "separate", "grouped")
    prepared_by_seed = {}
    ideals_by_seed = {}
    ideal_indices_by_seed = {}
    for offset in args.seed_offsets:
        seed = manifest["seed"] + offset
        requests = make_requests(family, n, depth, seed, output_qubits)
        ideals_by_seed[offset] = numpy_oracle_expectations(requests)
        ideal_indices_by_seed[offset] = {
            tuple(request.observable.factors): index for index, request in enumerate(requests)
        }
        prepared = {}
        for strategy in strategies:
            plan, circuits, circuit_metadata = compile_strategy(
                strategy, requests, backend, args.shots, args.repetitions, seed,
                output_qubits, args.placement,
            )
            prepared[strategy] = {
                "plan": plan,
                "circuits": circuits,
                "metadata": circuit_metadata,
            }
        prepared_by_seed[offset] = prepared

    new_shots_total = sum(
        len(prepared[strategy]["circuits"])
        for prepared in prepared_by_seed.values()
        for strategy in strategies
    ) * args.shots
    shots_total = new_shots_total + args.previously_spent_shots
    historical_ratio = historical_seconds_per_circuit_shot()
    estimated_qpu_seconds = shots_total * historical_ratio * 2.0
    if shots_total > args.shot_ceiling:
        parser.error(
            f"planned {shots_total} circuit-shots exceed --shot-ceiling={args.shot_ceiling}"
        )
    if estimated_qpu_seconds > args.estimated_qpu_seconds_ceiling:
        parser.error(
            f"conservative estimate {estimated_qpu_seconds:.1f} QPU seconds exceeds "
            f"--estimated-qpu-seconds-ceiling={args.estimated_qpu_seconds_ceiling:.1f}"
        )
    rows = []
    rng = random.Random(args.order_seed)
    default_name = "iqm_garnet_basis_probe" + ("_line" if args.placement == "line" else "")
    output_path = args.output or ROOT / "results/qml_workloads" / f"{default_name}.json"

    if args.preflight:
        for offset, prepared in prepared_by_seed.items():
            print(f"seed_offset={offset} ideal={ideals_by_seed[offset]}")
            for strategy in strategies:
                item = prepared[strategy]
                print(
                    strategy,
                    f"circuits={len(item['circuits'])}",
                    f"repetitions={args.repetitions}",
                    f"total_shots={len(item['circuits']) * args.shots}",
                    f"compiled_depths={[m['compiled_depth'] for m in item['metadata']]}",
                    f"compiled_2q_gates={sum(m['compiled_two_qubit_gates'] for m in item['metadata'])}",
                    f"correction_swaps={[len(m['output_correction_swaps']) for m in item['metadata'] if m['repetition_index'] == 0]}",
                    f"outputs_match_layout={all(m['measured_physical_qubits'] == m['expected_physical_qubits'] for m in item['metadata'])}",
                )
        print(
            f"preflight only; no jobs submitted; aggregate_circuit_shots={shots_total}; "
            f"new_jobs={len(prepared_by_seed) * len(strategies)}; "
            f"previously_submitted_jobs={args.previously_submitted_jobs}; "
            f"conservative_qpu_seconds_estimate={estimated_qpu_seconds:.1f}; "
            f"at_IQM_public_starting_rate_usd={estimated_qpu_seconds * 0.30:.2f}"
        )
        return

    def checkpoint(status, pending=None):
        payload = {
            "metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "backend": str(backend.name),
                "family": family,
                "qubits": n,
                "depth": depth,
                "output_qubits": list(output_qubits),
                "placement": args.placement,
                "bases": ["X", "Z"],
                "shots_per_circuit": args.shots,
                "repetitions_per_seed": args.repetitions,
                "seed_offsets": list(args.seed_offsets),
                "aggregate_circuit_shots": shots_total,
                "previously_spent_circuit_shots": args.previously_spent_shots,
                "previously_submitted_jobs": args.previously_submitted_jobs,
                "historical_seconds_per_shot_reference": historical_ratio,
                "estimated_qpu_seconds_with_2x_buffer": estimated_qpu_seconds,
                "public_starting_rate_usd_per_second": 0.30,
                "estimated_cost_at_public_starting_rate_usd": estimated_qpu_seconds * 0.30,
                "submitted": True,
                "account_cost_quote_available": False,
                "status": status,
            },
            "runs": rows,
            "pending_jobs": pending or [],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        return payload

    completed, pending_ids = set(), {}
    if args.resume and output_path.exists():
        previous = json.loads(output_path.read_text())
        rows.extend(previous["runs"])
        completed = {(r["seed_offset"], r["strategy"]) for r in rows}
        pending_ids = {
            (p["seed_offset"], p["strategy"]): p["job_id"]
            for p in previous.get("pending_jobs", [])
        }
        print(f"resuming: {len(completed)} completed, {len(pending_ids)} pending", flush=True)
    elif output_path.exists():
        parser.error(f"{output_path} exists; pass --resume or a different --output")

    for offset, prepared in prepared_by_seed.items():
        order = list(strategies)
        rng.shuffle(order)
        for strategy in order:
            if (offset, strategy) in completed:
                continue
            item = prepared[strategy]
            submitted_at = datetime.now(timezone.utc).isoformat()
            started = perf_counter()
            if (offset, strategy) in pending_ids:
                job_id = pending_ids[(offset, strategy)]
                job = provider.retrieve_job(job_id)
                print(f"reattached seed={offset} strategy={strategy} job={job_id}", flush=True)
            else:
                job = backend.run(item["circuits"], shots=args.shots)
                job_id = str(job.job_id())
                pending_ids[(offset, strategy)] = job_id
            # Record the job ID before waiting so a crash cannot lose a paid result.
            checkpoint(
                "in_progress",
                [
                    {"seed_offset": o, "strategy": s, "job_id": j}
                    for (o, s), j in pending_ids.items()
                ],
            )
            print(f"submitted seed={offset} strategy={strategy} job={job_id}", flush=True)
            result = job.result()
            elapsed = perf_counter() - started
            counts = result.get_counts()
            if isinstance(counts, dict):
                counts = [counts]
            if len(counts) != len(item["metadata"]):
                raise RuntimeError("Returned count-circuit count does not match submission")
            run_outputs = []
            for circuit_index, (task_counts, circuit_meta) in enumerate(
                zip(counts, item["metadata"], strict=True)
            ):
                for output in circuit_meta["outputs"]:
                    value = expectation_from_counts(
                        task_counts,
                        output["classical_bit"],
                        len(circuit_meta["outputs"]),
                    )
                    ideal_index = ideal_indices_by_seed[offset][
                        ((output["qubit"], output["pauli"]),)
                    ]
                    run_outputs.append(
                        {
                            "circuit_index": circuit_index,
                            "repetition_index": circuit_meta["repetition_index"],
                            "basis": circuit_meta["basis"],
                            "qubit": output["qubit"],
                            "pauli": output["pauli"],
                            "expectation": value,
                            "ideal_expectation": ideals_by_seed[offset][ideal_index],
                            "binomial_standard_error": (1 - value * value) ** 0.5
                            / (args.shots**0.5),
                        }
                    )
            rows.append(
                {
                    "seed_offset": offset,
                    "repetition": 0,
                    "strategy": strategy,
                    "job_id": job_id,
                    "submitted_at_utc": submitted_at,
                    "shots_per_circuit": args.shots,
                    "circuit_count": len(item["circuits"]),
                    "total_shots": len(item["circuits"]) * args.shots,
                    "submit_to_result_seconds": elapsed,
                    "timeline": json_safe(getattr(result, "timeline", None)),
                    "circuits": item["metadata"],
                    "outputs": run_outputs,
                    "counts": json_safe(counts),
                }
            )
            del pending_ids[(offset, strategy)]
            payload = checkpoint(
                "in_progress",
                [
                    {"seed_offset": o, "strategy": s, "job_id": j}
                    for (o, s), j in pending_ids.items()
                ],
            )
            print(
                f"finished seed={offset} strategy={strategy} job={job_id} "
                f"circuits={len(item['circuits'])} shots={rows[-1]['total_shots']}",
                flush=True,
            )

    summaries = {}
    for strategy in strategies:
        strategy_rows = [r for r in rows if r["strategy"] == strategy]
        by_output = {}
        for row in strategy_rows:
            for output_row in row["outputs"]:
                key = f"{output_row['qubit']}{output_row['pauli']}"
                by_output.setdefault(key, []).append(output_row["expectation"])
        execution_windows = []
        for row in strategy_rows:
            timeline = row["timeline"] or []
            events = {
                event.get("status"): event.get("timestamp")
                for event in timeline
                if event.get("source") == "iqm-station-control"
            }
            if "execution_started" in events and "execution_ended" in events:
                from datetime import datetime as dt
                import re

                def parse(value):
                    match = re.search(
                        r"\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)",
                        value,
                    )
                    return dt(*map(int, match.groups()), tzinfo=timezone.utc)

                execution_windows.append(
                    (parse(events["execution_ended"]) - parse(events["execution_started"]))
                    .total_seconds()
                )
        summaries[strategy] = {
            "jobs": len(strategy_rows),
            "circuits_per_job": [r["circuit_count"] for r in strategy_rows],
            "total_shots": sum(r["total_shots"] for r in strategy_rows),
            "execution_windows_seconds": execution_windows,
            "mean_execution_window_seconds": (
                mean(execution_windows) if execution_windows else None
            ),
            "sample_sd_execution_window_seconds": (
                stdev(execution_windows) if len(execution_windows) > 1 else None
            ),
            "mean_expectation_by_output": {
                key: mean(values) for key, values in by_output.items()
            },
        }
    payload = checkpoint("complete")
    payload["summaries"] = summaries
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"complete; output={output_path}; account cost still requires Resonance ledger check")


if __name__ == "__main__":
    main()
