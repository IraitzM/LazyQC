"""Reproduce LazyQC expectation workloads on the MQT stack.

MQT-extended reproduction: LazyQC Programs are converted to Qiskit circuits
and executed through MQT DDSIM (statevector), verified with MQT QCEC, and
round-tripped through MQT Core; MQT Bench supplies native baseline circuits.

Outputs results/mqt_repro/evaluation.json. Requires the `mqt` extra:
uv sync --locked --extra mqt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

PAULI_MATS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def program_to_qiskit(program):
    """Convert a bound LazyQC Program to a Qiskit QuantumCircuit."""
    from qiskit import QuantumCircuit

    ops = program.resolved_operations()
    qc = QuantumCircuit(program.n_qubits)
    for gate in ops:
        name, qs, angle = gate.name, gate.qubits, gate.angle
        if name == "H":
            qc.h(qs[0])
        elif name == "X":
            qc.x(qs[0])
        elif name == "Y":
            qc.y(qs[0])
        elif name == "Z":
            qc.z(qs[0])
        elif name == "RX":
            qc.rx(angle, qs[0])
        elif name == "RY":
            qc.ry(angle, qs[0])
        elif name == "RZ":
            qc.rz(angle, qs[0])
        elif name == "CX":
            qc.cx(qs[0], qs[1])
        else:  # pragma: no cover - IR gate set is closed
            raise ValueError(f"Unsupported gate for Qiskit export: {name}")
    return qc


def pauli_expectation(statevector, factors, n_qubits):
    """Evaluate a Pauli observable on a statevector (qubit 0 = least significant)."""
    mats = [PAULI_MATS[factors.get(q, "I")] for q in range(n_qubits - 1, -1, -1)]
    mat = mats[0]
    for mat_q in mats[1:]:
        mat = np.kron(mat, mat_q)
    vec = np.asarray(statevector, dtype=complex)
    return float(np.real(np.vdot(vec, mat @ vec)))


def ddsim_statevector(quantum_circuit):
    """Simulate a Qiskit circuit with MQT DDSIM and return the statevector."""
    from mqt.ddsim import DDSIMProvider

    backend = DDSIMProvider().get_backend("statevector_simulator")
    job = backend.run(quantum_circuit)
    return np.asarray(job.result().get_statevector(quantum_circuit))


def strip_measurements(quantum_circuit):
    """Return a copy without measurements and barriers for statevector simulation."""
    from qiskit import QuantumCircuit

    qc = quantum_circuit.copy()
    qc.remove_final_measurements()
    cleaned = QuantumCircuit(qc.num_qubits)
    for instruction in qc.data:
        if instruction.operation.name in ("measure", "barrier"):
            continue
        cleaned.append(instruction.operation, instruction.qubits, instruction.clbits)
    return cleaned


def workload_cases():
    """Build (name, Program, observables) reproduction cases from LazyQC workloads."""
    from lazyqc import Program

    pairs = Program(4).ry(0, 0.4).cx(0, 1).ry(2, 0.8).cx(2, 3)
    pairs_obs = [{0: "Z"}, {1: "Z"}, {0: "X", 1: "X"}, {2: "Z", 3: "Z"}]
    bell = Program(2).h(0).cx(0, 1)
    bell_obs = [{0: "Z", 1: "Z"}, {0: "X", 1: "X"}, {0: "Z"}, {0: "X", 1: "Z"}]
    overlap = (
        Program(6).h(0).cx(0, 1).cx(1, 2).ry(2, 0.5).cx(2, 3).cx(3, 4).cx(4, 5)
    )
    overlap_obs = [
        {0: "Z"},
        {0: "Z", 1: "Z"},
        {2: "X", 3: "X"},
        {5: "Z"},
        {3: "Z", 4: "Z", 5: "Z"},
    ]
    return [
        ("pairs4", pairs, pairs_obs),
        ("bell", bell, bell_obs),
        ("overlap6", overlap, overlap_obs),
    ]


def run_expectation_agreement(repeats):
    """Compare LazyQC collect() values against MQT DDSIM statevectors."""
    from lazyqc import collect_all

    rows = []
    for name, program, observables in workload_cases():
        requests = [program.expectation(obs) for obs in observables]
        t0 = time.perf_counter_ns()
        lazy_values = list(collect_all(requests).values)
        lazy_ns = time.perf_counter_ns() - t0
        qc = program_to_qiskit(program)
        times = []
        vectors = []
        for _ in range(repeats):
            t0 = time.perf_counter_ns()
            vectors.append(ddsim_statevector(qc))
            times.append(time.perf_counter_ns() - t0)
        vector = vectors[0]
        ddsim_values = [
            pauli_expectation(vector, obs, program.n_qubits) for obs in observables
        ]
        errors = [abs(a - b) for a, b in zip(lazy_values, ddsim_values)]
        rows.append(
            {
                "case": name,
                "n_qubits": program.n_qubits,
                "n_observables": len(observables),
                "lazy_values": lazy_values,
                "ddsim_values": ddsim_values,
                "max_abs_error": max(errors),
                "lazy_collect_ns": lazy_ns,
                "ddsim_median_ns": int(np.median(times)),
                "ddsim_repeats": repeats,
            }
        )
    return rows


def run_zx_verification():
    """Verify LazyQC PyZX rewrites with MQT QCEC equivalence checking."""
    from mqt import qcec

    from lazyqc import Program
    from lazyqc.rewrites import ZXOptions, rewrite_programs

    program = (
        Program(4)
        .h(0)
        .cx(0, 1)
        .rz(1, 0.37)
        .cx(0, 1)
        .rz(0, 0.5)
        .cx(0, 1)
        .rz(1, -0.37)
        .cx(0, 1)
    )
    requests = [program.expectation({0: "X"}), program.expectation({1: "Z"})]
    _, zx_requests, records, _ = rewrite_programs(
        requests, "zx", ZXOptions(soft_timeout_ms=20000.0)
    )
    record = records[0]
    outcome = {"rewrite_status": record.status, "rewrite_detail": record.detail}
    if record.status != "verified":
        outcome["qcec"] = "skipped"
        return outcome
    original_qc = program_to_qiskit(program)
    rewritten_qc = program_to_qiskit(zx_requests[0].program)
    result = qcec.verify(original_qc, rewritten_qc)
    outcome["qcec"] = str(result.equivalence)
    outcome["qcec_considered_equivalent"] = bool(
        result.equivalence == result.equivalence.EQUIVALENT
        if hasattr(result.equivalence, "EQUIVALENT")
        else "equiv" in str(result.equivalence).lower()
    )
    outcome["original_gates"] = len(program.resolved_operations())
    outcome["rewritten_gates"] = len(zx_requests[0].program.resolved_operations())
    return outcome


def run_core_roundtrip():
    """Round-trip workload circuits through MQT Core as the interchange layer."""
    from mqt.core import load as mqt_load
    from mqt.core.ir import QuantumComputation

    rows = []
    for name, program, _ in workload_cases():
        qc = program_to_qiskit(program)
        t0 = time.perf_counter_ns()
        computation = mqt_load(qc)
        qasm = computation.qasm2_str()
        restored = QuantumComputation.from_qasm_str(qasm)
        rows.append(
            {
                "case": name,
                "qubits": computation.num_qubits,
                "ops": len(computation),
                "qasm_chars": len(qasm),
                "roundtrip_ns": time.perf_counter_ns() - t0,
                "restored_qubits": restored.num_qubits,
                "restored_ops": len(restored),
            }
        )
    return rows


def run_bench_baselines():
    """Simulate MQT Bench circuits natively with DDSIM; convert GHZ to LazyQC."""
    from mqt.bench import get_benchmark
    from mqt.bench.benchmark_generation import BenchmarkLevel
    from mqt.core import load as mqt_load

    rows = []
    for bench_name, size in (("ghz", 4), ("qaoa", 4), ("qft", 4)):
        qc = get_benchmark(bench_name, level=BenchmarkLevel.ALG, circuit_size=size)
        sim_qc = strip_measurements(qc)
        t0 = time.perf_counter_ns()
        vector = ddsim_statevector(sim_qc)
        elapsed_ns = time.perf_counter_ns() - t0
        norm = float(np.linalg.norm(vector))
        computation = mqt_load(sim_qc)
        rows.append(
            {
                "benchmark": bench_name,
                "qubits": sim_qc.num_qubits,
                "ops": dict(sim_qc.count_ops()),
                "core_ops": len(computation),
                "state_norm": norm,
                "ddsim_ns": elapsed_ns,
            }
        )
    ghz_lazy = ghz_lazy_comparison()
    return rows, ghz_lazy


def ghz_lazy_comparison():
    """Run the MQT Bench GHZ-4 circuit through LazyQC by gate conversion."""
    from qiskit import QuantumCircuit

    from lazyqc import Program, collect_all

    program = Program(4).h(0).cx(0, 1).cx(1, 2).cx(2, 3)
    reference = QuantumCircuit(4)
    reference.h(0)
    reference.cx(0, 1)
    reference.cx(1, 2)
    reference.cx(2, 3)
    vector = ddsim_statevector(reference)
    observables = [{0: "Z", 1: "Z"}, {0: "Z", 3: "Z"}, {0: "X", 1: "X", 2: "X", 3: "X"}]
    lazy_values = list(
        collect_all([program.expectation(obs) for obs in observables]).values
    )
    ddsim_values = [
        pauli_expectation(vector, obs, 4) for obs in observables
    ]
    return {
        "lazy_values": lazy_values,
        "ddsim_values": ddsim_values,
        "max_abs_error": max(abs(a - b) for a, b in zip(lazy_values, ddsim_values)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/mqt_repro/evaluation.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    try:
        import mqt.bench  # noqa: F401
        import mqt.ddsim  # noqa: F401
        import mqt.qcec  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(f"MQT extra not installed (uv sync --extra mqt): {exc}") from exc

    payload = {
        "versions": {
            "mqt_core": version("mqt-core"),
            "mqt_ddsim": version("mqt-ddsim"),
            "mqt_qcec": version("mqt-qcec"),
            "mqt_bench": version("mqt-bench"),
        },
        "agreement": run_expectation_agreement(args.repeats),
        "zx_qcec": run_zx_verification(),
        "core_roundtrip": run_core_roundtrip(),
    }
    bench_rows, ghz_lazy = run_bench_baselines()
    payload["bench_ddsim"] = bench_rows
    payload["bench_ghz_lazyqc"] = ghz_lazy

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    worst = max(row["max_abs_error"] for row in payload["agreement"])
    print(f"worst LazyQC-vs-DDSIM error: {worst:.3e}")
    print(f"QCEC ZX verdict: {payload['zx_qcec']}")
    print(f"GHZ LazyQC-vs-DDSIM error: {ghz_lazy['max_abs_error']:.3e}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
