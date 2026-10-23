"""Qiskit circuits as a frontend language for the batch planner.

Write plain Qiskit code; the importer translates it to the LazyQC IR, and
that is where the planner kicks in: ``plan_all`` sees the translated
program, never the Qiskit object. MQT Core may sit upstream
(``QuantumComputation`` exported to QASM2 and parsed back) without changing
the contract below.

Scope: bound unitary circuits over the IR gate set after basis translation
(H, X, Y, Z, RX, RY, RZ, CX). Measurements are rejected unless explicitly
stripped; barriers are dropped as scheduling directives; anything else
without a basis translation (reset, delay, custom non-unitary ops) is an
error, not a silent skip. The program starts in the all-zero product state,
matching Qiskit statevector semantics up to global phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

from ..ir import GATES, Gate, Program

_BASIS_IR = {name.lower(): name for name in sorted(GATES)}
_DROPPED = ("barrier",)


@dataclass(frozen=True)
class QiskitImport:
    """Translated program plus deferred requests and translation provenance."""

    program: Program
    requests: tuple
    source_ops: tuple[str, ...]
    translated_ops: tuple[str, ...]
    global_phase: float
    dropped: tuple[str, ...]


def _require_qiskit():
    """Import Qiskit lazily so the core package has no Qiskit dependency."""
    try:
        import qiskit
    except ImportError as exc:
        raise ImportError(
            "The Qiskit frontend requires Qiskit: "
            "uv sync --extra cutting (or --extra mqt)"
        ) from exc
    return qiskit


def _bind(circuit, bindings):
    """Substitute numeric bindings keyed by parameter name."""
    if not circuit.num_parameters:
        if bindings:
            raise ValueError("Circuit has no parameters; bindings are unused")
        return circuit
    if not bindings:
        names = sorted(p.name for p in circuit.parameters)
        raise ValueError(f"Circuit has free parameters {names}; pass bindings={{name: value}}")
    try:
        mapping = {p: bindings[p.name] for p in circuit.parameters}
    except KeyError as exc:
        raise ValueError(f"Missing binding for parameter {exc}") from exc
    unbound = [p.name for p, v in mapping.items() if v is None]
    if unbound:
        raise ValueError(f"Missing binding for parameters {unbound}")
    return circuit.assign_parameters(mapping)


def _strip_measurements(circuit, strip):
    """Remove final measurements when explicitly requested; else reject."""
    ops = {inst.operation.name for inst in circuit.data}
    if "measure" not in ops:
        return circuit
    if not strip:
        raise ValueError("Circuit contains measurements; pass strip_measurements=True to drop them")
    cleaned = circuit.copy()
    cleaned.remove_final_measurements()  # Qiskit 2.x strips in place
    if not cleaned.data:
        raise ValueError("Circuit has only measurements; nothing to translate")
    remaining = {inst.operation.name for inst in cleaned.data}
    if "measure" in remaining:
        raise ValueError("Only terminal measurements can be stripped")
    return cleaned


def _to_basis(circuit):
    """Translate to the IR basis without optimization or layout changes."""
    qiskit = _require_qiskit()
    translated = qiskit.transpile(
        circuit, basis_gates=list(_BASIS_IR), optimization_level=0
    )
    problems = sorted(
        {
            inst.operation.name
            for inst in translated.data
            if inst.operation.name not in _BASIS_IR and inst.operation.name not in _DROPPED
        }
    )
    if problems:
        raise ValueError(
            f"Instructions without IR translation: {problems}; "
            "only unitary basis gates survive translation"
        )
    return translated


def from_qiskit(circuit, observables, *, bindings=None, strip_measurements=False):
    """Translate a Qiskit circuit plus Pauli observables into IR requests.

    Args:
        circuit: A Qiskit QuantumCircuit (parameters allowed with bindings).
        observables: An iterable of Pauli mappings ({qubit_index: axis}).
        bindings: Optional {parameter_name: radians} for parameterized circuits.
        strip_measurements: Drop terminal measurements instead of rejecting.

    Returns:
        QiskitImport with the translated Program, one Request per
        observable, and provenance (source/translated op names,
        global phase, dropped directives).
    """
    qiskit = _require_qiskit()
    if not isinstance(circuit, qiskit.QuantumCircuit):
        raise ValueError("Expected a Qiskit QuantumCircuit")
    observables = tuple(observables)
    if not observables:
        raise ValueError("Provide at least one observable")

    bound = _bind(circuit, bindings or {})
    cleaned = _strip_measurements(bound, strip_measurements)
    source_ops = tuple(inst.operation.name for inst in cleaned.data)
    translated = _to_basis(cleaned)

    index_of = {bit: i for i, bit in enumerate(translated.qubits)}
    operations, dropped = [], []
    for inst in translated.data:
        name = inst.operation.name
        if name in _DROPPED:
            dropped.append(name)
            continue
        qubits = tuple(index_of[q] for q in inst.qubits)
        params = getattr(inst.operation, "params", ())
        angle = float(params[0]) % (2 * pi) if params else None
        if angle is not None and angle > pi:
            angle -= 2 * pi
        operations.append(Gate(_BASIS_IR[name], qubits, angle))

    program = Program(len(translated.qubits), tuple(operations))
    requests = tuple(program.expectation(dict(obs)) for obs in observables)
    return QiskitImport(
        program=program,
        requests=requests,
        source_ops=source_ops,
        translated_ops=tuple(g.name for g in operations),
        global_phase=float(translated.global_phase),
        dropped=tuple(dropped),
    )


def from_mqt_core(computation, observables, **options):
    """Translate an MQT Core QuantumComputation via QASM2 into IR requests.

    The computation is exported to OpenQASM 2, parsed with Qiskit, and
    delegated to :func:`from_qiskit`. Accepts the same keyword options.
    """
    _require_qiskit()
    try:
        qasm = computation.qasm2_str()
    except AttributeError as exc:
        raise ValueError("Expected an MQT Core QuantumComputation") from exc
    from qiskit import qasm2

    return from_qiskit(qasm2.loads(qasm), observables, **options)
