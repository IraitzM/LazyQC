#!/usr/bin/env python3
"""Compare full and cone plans on IonQ's Forte-1 noisy simulator."""

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from benchmark_support import source_hash
from qml_workload_support import make_batch, numpy_oracle_expectations
from run_ionq_simulator_probe import api, circuit_input, load_token

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "qml_workload_manifest.json"


def distribution_from_response(result):
    """Normalize v1 flat or v2 register-nested probability responses."""
    if "probabilities" in result:
        result = result["probabilities"]
        if "registers" in result:
            result = next(iter(result["registers"].values()))
    return {str(state): float(probability) for state, probability in result.items()}


def submit_measurement(
    task, wire_map, pauli_basis, logical_qubit, token, model, noise_seed, shots, label
):
    """Submit one Pauli basis measurement and return its sampled expectation."""
    input_data, task_wire_map = circuit_input(task)
    if task_wire_map != wire_map:
        raise RuntimeError("Wire mapping changed while preparing measurement")
    qubit = wire_map[logical_qubit]
    basis = pauli_basis
    if basis == "X":
        input_data["circuit"].append({"gate": "h", "target": qubit})
    elif basis == "Y":
        input_data["circuit"].extend(
            [
                {"gate": "rz", "target": qubit, "rotation": -math.pi / 2},
                {"gate": "h", "target": qubit},
            ]
        )
    elif basis != "Z":
        raise ValueError(f"Unsupported Pauli basis: {basis}")

    payload = {
        "type": "ionq.circuit.v1",
        "name": f"distributedqc-noisy-{label}",
        "backend": "simulator",
        "shots": shots,
        "noise": {"model": model, "seed": noise_seed},
        "input": input_data,
    }
    submitted = perf_counter_ns()
    created = api("POST", "/jobs", token, payload)
    job_id = created["id"]
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        job = api("GET", f"/jobs/{job_id}", token)
        status = job["status"]
        if status == "completed":
            break
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"IonQ noisy job ended with status {status}")
        time.sleep(1.0)
    else:
        raise TimeoutError(f"IonQ noisy job {job_id} exceeded 600 seconds")
    wall_ns = perf_counter_ns() - submitted
    distribution = distribution_from_response(
        api("GET", f"/jobs/{job_id}/results/probabilities", token)
    )
    expectation = 0.0
    for state, probability in distribution.items():
        expectation += (-1.0 if (int(state) >> qubit) & 1 else 1.0) * probability
    return {
        "job_id": job_id,
        "status": status,
        "measurement_pauli": basis,
        "measurement_qubit": qubit,
        "submitted_gates": len(input_data["circuit"]),
        "submitted_1q_gate_count": sum(
            gate["gate"] != "cnot" for gate in input_data["circuit"]
        ),
        "submitted_2q_gate_count": sum(
            gate["gate"] == "cnot" for gate in input_data["circuit"]
        ),
        "returned_probability_states": len(distribution),
        "requested_shots": shots,
        "reported_shots": job.get("shots"),
        "execution_duration_ms": job.get("execution_duration_ms"),
        "stats": job.get("stats"),
        "submit_to_completion_ns": wall_ns,
        "expectation": expectation,
        "binomial_standard_error": math.sqrt(max(0.0, 1 - expectation**2) / shots),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run one family (9 jobs)")
    parser.add_argument("--families", nargs="+", help="Restrict to named QML families")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = manifest["ionq_noisy_probe"]
    shots = args.shots or config["shots"]
    if shots < 1:
        parser.error("--shots must be positive")

    from lazyqc import plan_all

    token = load_token()
    backend_list = api("GET", "/backends", token)
    simulator = next(
        (row for row in backend_list if row.get("backend") == "simulator"), None
    )
    if not simulator or simulator.get("status") != "available":
        raise RuntimeError("IonQ simulator backend is unavailable")

    if args.smoke and args.families:
        parser.error("--smoke cannot be combined with --families")
    families = (
        config["families"][:1]
        if args.smoke
        else (args.families or config["families"])
    )
    unknown = set(families) - set(config["families"])
    if unknown:
        parser.error(f"unknown family name(s): {', '.join(sorted(unknown))}")
    strategies = ("full", "separate", "grouped")
    run_rows, case_rows = [], []
    for family_index, family in enumerate(families):
        case = {
            "case_id": f"{family}_{config['qubits']}q_d{config['depth']}_"
            f"{config['pattern']}_s{config['seed_offset']}",
            "family": family,
            "qubits": config["qubits"],
            "depth": config["depth"],
            "pattern": config["pattern"],
            "seed": manifest["seed"] + config["seed_offset"],
            "seed_offset": config["seed_offset"],
        }
        requests = make_batch(
            family, case["qubits"], case["depth"], case["pattern"], case["seed"]
        )
        ideal = np.asarray(numpy_oracle_expectations(requests))
        plans, planning_ns = {}, {}
        for strategy in strategies:
            start = perf_counter_ns()
            plans[strategy] = plan_all(
                requests,
                strategy=strategy,
                optimization="none",
                max_state_qubits=24,
                max_state_bytes=256 * 2**20,
            )
            planning_ns[strategy] = perf_counter_ns() - start

        for strategy in strategies:
            plan = plans[strategy]
            task_values = {}
            output_locations = {
                output_index: (task_index, observable)
                for task_index, task in enumerate(plan.selected.tasks)
                for output_index, observable in task.outputs
            }
            for output_index in range(len(requests)):
                task_index, observable = output_locations[output_index]
                local_output = next(
                    index
                    for index, (candidate_output, _) in enumerate(
                        plan.selected.tasks[task_index].outputs
                    )
                    if candidate_output == output_index
                )
                task = plan.selected.tasks[task_index]
                _, wire_map = circuit_input(task)
                logical_qubit, pauli_basis = observable.factors[0]
                noise_seed = config["noise_seeds"][0] + 100 * family_index + output_index
                label = f"{family}-{strategy}-{pauli_basis}"
                run = submit_measurement(
                    task,
                    wire_map,
                    pauli_basis,
                    logical_qubit,
                    token,
                    config["noise_model"],
                    noise_seed,
                    shots,
                    label,
                )
                task_values[output_index] = run["expectation"]
                run_rows.append(
                    {
                        **case,
                        "strategy": strategy,
                        "output_index": output_index,
                        "task_index": task_index,
                        "task_output_index": local_output,
                        "task_width": task.width,
                        "task_operation_count": len(task.operations),
                        "planning_ns": planning_ns[strategy],
                        "noise_model": config["noise_model"],
                        "noise_seed": noise_seed,
                        "shots": shots,
                        "ideal_expectation": float(ideal[output_index]),
                        **run,
                    }
                )
            estimates = np.asarray([task_values[index] for index in range(len(requests))])
            case_rows.append(
                {
                    **case,
                    "strategy": strategy,
                    "task_count": len(plan.selected.tasks),
                    "task_widths": [task.width for task in plan.selected.tasks],
                    "task_operation_counts": [
                        len(task.operations) for task in plan.selected.tasks
                    ],
                    "planning_ns": planning_ns[strategy],
                    "ideal_expectations": ideal.tolist(),
                    "noisy_expectations": estimates.tolist(),
                    "max_abs_deviation_from_ideal": float(
                        np.max(np.abs(estimates - ideal))
                    ),
                }
            )

    output = args.output or ROOT / "results/qml_workloads/ionq_noisy_qml_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "IonQ noisy simulator",
            "backend": "simulator",
            "noise_model": config["noise_model"],
            "backend_status_at_run": simulator.get("status"),
            "shots": shots,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": source_hash(__file__),
            "ideal_runner_sha256": source_hash(
                ROOT / "benchmarks" / "run_ionq_simulator_probe.py"
            ),
            "smoke": args.smoke,
            "case_count": len(case_rows) // len(strategies),
            "note": (
                "Noisy simulation is a hardware-noise approximation; no QPU queue "
                "or hardware timing was measured. Noncommuting X/Y/Z outputs use "
                "separate measurement-basis jobs."
            ),
        },
        "cases": case_rows,
        "runs": run_rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"cases={payload['metadata']['case_count']} jobs={len(run_rows)} "
        f"noise={config['noise_model']} shots={shots} output={output}"
    )


if __name__ == "__main__":
    main()
