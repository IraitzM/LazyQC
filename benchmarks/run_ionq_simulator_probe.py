#!/usr/bin/env python3
"""Cross-check full-state and causal-cone QML plans on IonQ's ideal simulator."""

import argparse
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from benchmark_support import source_hash
from qml_workload_support import numpy_oracle_expectations, trajectory_batch

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://api.ionq.co/v0.4"
REMOTE_ABS_TOLERANCE = 1e-2


def load_token():
    """Load the exact IonQ token from the environment or the user's dotenv file."""
    token = os.environ.get("IONQ_TOKEN") or os.environ.get("IONQ_API_KEY")
    if token:
        return token
    env_path = Path.home() / ".env"
    for line in env_path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key.strip().removeprefix("export ").strip() in (
            "IONQ_TOKEN",
            "IONQ_API_KEY",
        ):
            return value.strip().strip("\"'")
    raise RuntimeError("Set IONQ_TOKEN or IONQ_API_KEY, or add one to ~/.env")


def api(method, path, token, payload=None):
    """Call the IonQ API without logging headers or response bodies on errors."""
    body = None if payload is None else json.dumps(payload).encode()
    request = Request(
        BASE_URL + path,
        data=body,
        method=method,
        headers={
            "Authorization": f"apiKey {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except HTTPError as exc:
        raise RuntimeError(f"IonQ API returned HTTP {exc.code} for {method} {path}") from None
    except URLError as exc:
        raise RuntimeError(f"IonQ API request failed: {type(exc.reason).__name__}") from None


def circuit_input(task):
    """Translate one no-cut planner task to IonQ's QIS circuit JSON."""
    from lazyqc import Gate

    wire_map = {wire: index for index, wire in enumerate(task.wires)}
    gates = []
    for operation in task.operations:
        if not isinstance(operation, Gate):
            raise ValueError("This simulator probe expects no-cut unitary tasks")
        qubits = tuple(wire_map[q] for q in operation.qubits)
        if operation.name == "CX":
            gates.append({"gate": "cnot", "control": qubits[0], "target": qubits[1]})
        elif operation.name in {"RX", "RY", "RZ"}:
            gates.append(
                {
                    "gate": operation.name.lower(),
                    "target": qubits[0],
                    "rotation": operation.angle,
                }
            )
        else:
            gates.append({"gate": operation.name.lower(), "target": qubits[0]})
    return {"qubits": task.width, "gateset": "qis", "circuit": gates}, wire_map


def run_task(task, token, label):
    """Submit one task, wait for completion, and return exact Pauli marginals."""
    input_data, wire_map = circuit_input(task)
    payload = {
        "type": "ionq.circuit.v1",
        "name": f"distributedqc-{label}",
        "backend": "simulator",
        "shots": 100,
        "input": input_data,
    }
    submitted = time.perf_counter_ns()
    created = api("POST", "/jobs", token, payload)
    job_id = created["id"]
    submission_ns = time.perf_counter_ns() - submitted

    deadline = time.monotonic() + 180
    polling_started = time.perf_counter_ns()
    while time.monotonic() < deadline:
        job = api("GET", f"/jobs/{job_id}", token)
        status = job["status"]
        if status == "completed":
            break
        if status in {"failed", "canceled"}:
            raise RuntimeError(f"IonQ job ended with status {status}")
        time.sleep(0.5)
    else:
        raise TimeoutError(f"IonQ job {job_id} did not complete within 180 seconds")
    polling_ns = time.perf_counter_ns() - polling_started
    results = api("GET", f"/jobs/{job_id}/results/probabilities", token)
    if "probabilities" in results:
        distribution = results["probabilities"]
        if "registers" in distribution:
            distribution = next(iter(distribution["registers"].values()))
    else:
        distribution = results

    values = {}
    for output_index, observable in task.outputs:
        expectation = 0.0
        for basis_state, probability in distribution.items():
            state = int(basis_state)
            parity = sum(
                (state >> wire_map[qubit]) & 1
                for qubit, pauli in observable.factors
                if pauli == "Z"
            )
            if any(pauli != "Z" for _, pauli in observable.factors):
                raise ValueError("Probe currently supports Z-only observables")
            expectation += (-1.0 if parity % 2 else 1.0) * float(probability)
        values[output_index] = expectation

    return {
        "job_id": job_id,
        "status": status,
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "execution_duration_ms": job.get("execution_duration_ms"),
        "reported_shots": job.get("shots"),
        "qubits": task.width,
        "gate_count": len(input_data["circuit"]),
        "submission_ns": submission_ns,
        "submit_to_completion_ns": time.perf_counter_ns() - submitted,
        "completion_poll_ns": polling_ns,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import numpy as np

    from lazyqc import CutOptions, Program
    from lazyqc.batch_planner import plan_batch

    manifest_path = ROOT / "benchmarks" / "qml_workload_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    token = load_token()
    backends = api("GET", "/backends", token)
    simulator = next(
        (backend for backend in backends if backend.get("backend") == "simulator"),
        None,
    )
    if not simulator or simulator.get("status") != "available":
        raise RuntimeError("IonQ ideal simulator is not available")

    case = {
        "family": "hardware_efficient_local",
        "qubits": 8,
        "depth": 4,
        "seed": manifest["seed"] + 202,
        "theta": -0.55,
    }
    batch = trajectory_batch(
        case["family"], case["qubits"], case["depth"], case["seed"], case["theta"]
    )
    expected = np.asarray(numpy_oracle_expectations(batch))
    options = CutOptions(
        max_fragment_qubits=24,
        max_cuts=0,
        max_group_size=4,
        beam_width=3,
        max_search_states=64,
        soft_timeout_ms=1000,
        search="joint",
    )
    token_free_rows, strategy_errors = [], {}
    for strategy in ("full", "separate"):
        plan = plan_batch(
            batch,
            cutting=options,
            strategy=strategy,
            optimization="none",
            max_state_qubits=24,
            max_state_bytes=256 * 2**20,
        )
        task_values = {}
        for index, task in enumerate(plan.selected.tasks):
            run = run_task(task, token, f"{strategy}-task{index}")
            token_free_rows.append(
                {
                    "strategy": strategy,
                    "task_index": index,
                    "task_width": task.width,
                    "task_outputs": [output for output, _ in task.outputs],
                    "gate_count": run["gate_count"],
                    "submission_ns": run["submission_ns"],
                    "submit_to_completion_ns": run["submit_to_completion_ns"],
                    "completion_poll_ns": run["completion_poll_ns"],
                    "execution_duration_ms": run["execution_duration_ms"],
                    "reported_shots": run["reported_shots"],
                    "values": {str(k): v for k, v in run["values"].items()},
                    "job_id": run["job_id"],
                }
            )
            task_values[index] = run["values"]
        strategy_values = {}
        for reconstruction in plan.selected.reconstruction:
            value = 0.0
            for term in reconstruction.terms:
                product = term.coefficient
                for task_index, task_output in term.factors:
                    product *= task_values[task_index][task_output]
                value += product
            strategy_values[reconstruction.output] = value
        obtained = np.asarray([strategy_values[index] for index in range(len(batch))])
        error = float(np.max(np.abs(obtained - expected)))
        strategy_errors[strategy] = error

    calibration_program = Program(1).ry(0, 0.55)
    calibration_request = [calibration_program.expectation({0: "Z"})]
    calibration_plan = plan_batch(
        calibration_request,
        cutting=options,
        strategy="full",
        optimization="none",
        max_state_qubits=24,
        max_state_bytes=256 * 2**20,
    )
    calibration_run = run_task(
        calibration_plan.selected.tasks[0], token, "single-ry-calibration"
    )
    calibration_value = calibration_run["values"][0]
    calibration_expected = math.cos(0.55)
    calibration_error = abs(calibration_value - calibration_expected)
    errors = {**strategy_errors, "single_ry_calibration": calibration_error}
    failures = [
        {"strategy": strategy, "max_abs_error": error}
        for strategy, error in errors.items()
        if error > REMOTE_ABS_TOLERANCE
    ]

    output = args.output or ROOT / "results/qml_workloads/ionq_simulator_probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provider": "IonQ ideal simulator",
            "backend_status_at_run": simulator.get("status"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "runner_sha256": source_hash(__file__),
            "case": case,
            "request_count": len(batch),
            "remote_absolute_tolerance": REMOTE_ABS_TOLERANCE,
            "max_abs_error_by_strategy": errors,
            "single_ry_calibration": {
                "theta_radians": 0.55,
                "expected_z": calibration_expected,
                "ionq_z": calibration_value,
                "absolute_error": calibration_error,
                "job_id": calibration_run["job_id"],
            },
            "failures": failures,
            "shots_note": (
                "IonQ docs say ideal simulator ignores shots. The calibration result "
                "was quantized to 0.01 probabilities and the job record reported no "
                "shot count; preserve this as a service precision limitation."
            ),
            "interpretation": (
                "correctness cross-check only; simulator API timings are not "
                "QPU sequencing evidence"
            ),
        },
        "runs": token_free_rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"jobs={len(token_free_rows) + 1} failures_over_tolerance={len(failures)} "
        f"max_error={max(errors.values()):.2e} "
        f"output={output}"
    )
    if failures:
        raise SystemExit("IonQ simulator oracle mismatch")


if __name__ == "__main__":
    main()
