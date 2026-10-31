#!/usr/bin/env python3
"""Run a small matched direct/cone QML workload on IQM Garnet."""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

from qml_workload_support import make_batch  # noqa: E402


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if hasattr(value, "to_dict"):
        return json_safe(value.to_dict())
    return repr(value)


def lower_task(task):
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(task.width, len(task.outputs))
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
            raise ValueError(f"IQM pilot has no Qiskit lowering for {gate.name}")

    output_qubits = []
    for classical_bit, (_, observable) in enumerate(task.outputs):
        if len(observable.factors) != 1 or observable.factors[0][1] != "Z":
            raise ValueError("Pilot expects single-qubit Z observables")
        wire = observable.factors[0][0]
        output_qubits.append(wire)
        circuit.measure(mapping[wire], classical_bit)
    return circuit, output_qubits


def expectation_from_counts(counts, classical_bit, width):
    ones = 0
    shots = 0
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        bits = format(int(bits, 2), f"0{width}b")
        ones += int(bits[-1 - classical_bit]) * count
        shots += count
    return 1.0 - 2.0 * ones / shots if shots else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=100)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("full", "separate", "grouped"),
        default=("full", "separate", "grouped"),
    )
    parser.add_argument("--seed-offset", type=int, default=202)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the prepared jobs to real Garnet hardware.",
    )
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")

    from iqm.qiskit_iqm import IQMProvider
    from qiskit import transpile
    from lazyqc import plan_all

    manifest = json.loads((ROOT / "benchmarks/qml_workload_manifest.json").read_text())
    cfg = manifest["large_cone_probe"]
    family, n, depth = "hardware_efficient_local", 20, cfg["depth"]
    seed = manifest["seed"] + args.seed_offset
    requests = make_batch(family, n, depth, "same_state_local_outputs", seed)
    provider = IQMProvider(url="https://resonance.iqm.tech", quantum_computer="garnet")
    backend = provider.get_backend()

    prepared = {}
    for strategy in args.strategies:
        plan = plan_all(
            requests,
            strategy=strategy,
            optimization="none",
            max_state_qubits=cfg["max_state_qubits"],
            max_state_bytes=cfg["max_state_bytes"],
        ).selected
        circuits, circuit_metadata = [], []
        for task_index, task in enumerate(plan.tasks):
            circuit, output_qubits = lower_task(task)
            compiled = transpile(
                circuit,
                backend=backend,
                optimization_level=1,
                seed_transpiler=508,
            )
            ops = compiled.count_ops()
            circuits.append(compiled)
            circuit_metadata.append(
                {
                    "task_index": task_index,
                    "logical_width": task.width,
                    "logical_wires": list(task.wires),
                    "output_qubits": output_qubits,
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

    order = list(args.strategies)
    random.Random(508).shuffle(order)
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "backend": str(backend.name),
            "family": family,
            "qubits": n,
            "depth": depth,
            "seed": seed,
            "shots_per_circuit": args.shots,
            "strategy_order": order,
            "submitted": bool(args.submit),
            "account_cost_quote_available": False,
            "job_limit": None,
            "note": "No live credit quote is exposed by the inspected IQM client API.",
        },
        "plans": {},
    }

    for strategy in order:
        item = prepared[strategy]
        row = {
            "circuit_count": len(item["circuits"]),
            "total_shots": len(item["circuits"]) * args.shots,
            "circuits": item["circuit_metadata"],
        }
        if args.submit:
            started = datetime.now(timezone.utc).isoformat()
            start = perf_counter()
            job = backend.run(item["circuits"], shots=args.shots)
            job_id = str(job.job_id())
            result = job.result()
            elapsed = perf_counter() - start
            counts = result.get_counts()
            if isinstance(counts, dict):
                counts = [counts]
            outputs = []
            for task, task_counts in zip(item["tasks"], counts, strict=True):
                for classical_bit, (program_id, observable) in enumerate(task.outputs):
                    q, pauli = observable.factors[0]
                    outputs.append(
                        {
                            "program_id": program_id,
                            "qubit": q,
                            "pauli": pauli,
                            "expectation": expectation_from_counts(
                                task_counts, classical_bit, len(task.outputs)
                            ),
                        }
                    )
            row.update(
                {
                    "job_id": job_id,
                    "submitted_at_utc": started,
                    "submit_to_result_seconds": elapsed,
                    "timeline": json_safe(getattr(result, "timeline", None)),
                    "outputs": outputs,
                    "counts": json_safe(counts),
                }
            )
        payload["plans"][strategy] = row

    output = args.output or ROOT / "results/qml_workloads/iqm_garnet_pilot.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for strategy in order:
        row = payload["plans"][strategy]
        print(
            strategy,
            "circuits=", row["circuit_count"],
            "total_shots=", row["total_shots"],
            "compiled_depths=", [c["compiled_depth"] for c in row["circuits"]],
            "job_id=", row.get("job_id", "not submitted"),
        )
    print("output=", output)


if __name__ == "__main__":
    main()
