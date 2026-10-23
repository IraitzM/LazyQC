"""Optional unitary rewrites for terminal expectations; global phase is unobservable."""

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from math import isfinite, pi
from pathlib import Path
from time import perf_counter_ns

from .ir import ROTATIONS, Gate


@dataclass(frozen=True)
class ZXOptions:
    max_qubits: int = 16
    max_gates: int = 256
    max_circuits: int = 8
    soft_timeout_ms: float = 1000.0

    def __post_init__(self):
        """Validate rewrite options and normalize their numeric limits."""
        for name in ("max_qubits", "max_gates", "max_circuits"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.soft_timeout_ms, bool)
            or not isfinite(self.soft_timeout_ms)
            or self.soft_timeout_ms <= 0
        ):
            raise ValueError("soft_timeout_ms must be finite and positive")


@dataclass(frozen=True)
class RewriteRecord:
    qubits: int
    input_gates: int
    local_gates: int
    zx_gates: int | None
    status: str
    detail: str
    elapsed_ns: int


@lru_cache(maxsize=2)
def optimizer_identity(mode):
    """Return the versioned identity of a requested optimizer mode."""
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if mode == "zx":
        try:
            return f"{digest}/pyzx-{version('pyzx')}"
        except PackageNotFoundError as exc:
            raise ImportError(
                "ZX optimization requires the optional extra: uv sync --extra zx"
            ) from exc
    return digest


def local_simplify(operations):
    """Apply the deterministic local gate simplifications to operations.

    Cancel adjacent inverses and exact zero rotations; no approximate angle merging.
    """
    stack = []
    for gate in operations:
        if gate.name in ROTATIONS and gate.angle == 0:
            continue
        if stack and stack[-1].qubits == gate.qubits and stack[-1].name == gate.name:
            if gate.name not in ROTATIONS or stack[-1].angle == -gate.angle:
                stack.pop()
                continue
        stack.append(gate)
    return tuple(stack)


def to_pyzx(qubits, operations):
    """Convert IR operations to a PyZX circuit, requiring the optional dependency."""
    from pyzx import Circuit
    from pyzx.circuit.gates import CNOT, HAD, XPhase, ZPhase

    circuit = Circuit(qubits)
    for gate in operations:
        q = gate.qubits[0]
        if gate.name == "H":
            circuit.add_gate(HAD(q))
        elif gate.name == "CX":
            circuit.add_gate(CNOT(*gate.qubits))
        elif gate.name in ("X", "Y", "Z"):
            if gate.name in ("Y", "Z"):
                circuit.add_gate(ZPhase(q, 1))
            if gate.name in ("X", "Y"):
                circuit.add_gate(XPhase(q, 1))
        else:
            # Exact ratio of the input binary floats; never snap to Clifford+T.
            if abs(gate.angle) > 8 * pi:
                raise ValueError("rotation magnitude exceeds supported 8*pi conversion bound")
            phase = Fraction(gate.angle) / Fraction(pi)
            if gate.name == "RY":
                # Chronological RZ(-pi/2), RX(theta), RZ(+pi/2).
                circuit.add_gate(ZPhase(q, Fraction(-1, 2)))
            circuit.add_gate((ZPhase if gate.name == "RZ" else XPhase)(q, phase))
            if gate.name == "RY":
                circuit.add_gate(ZPhase(q, Fraction(1, 2)))
    return circuit


def from_pyzx(circuit, qubits):
    """Convert a PyZX circuit back into IR operations and preserve qubit names."""
    from pyzx.circuit.gates import CNOT, CZ, HAD, SWAP, XPhase, ZPhase

    if circuit.qubits != qubits or circuit.bits:
        raise ValueError("ZX extraction changed the circuit interface")
    operations = []
    for g in circuit.to_basic_gates().gates:
        if isinstance(g, HAD):
            operations.append(Gate("H", (g.target,)))
        elif isinstance(g, CNOT):
            operations.append(Gate("CX", (g.control, g.target)))
        elif isinstance(g, SWAP):  # SWAP subclasses CZ in PyZX.
            a, b = g.control, g.target
            operations.extend((Gate("CX", (a, b)), Gate("CX", (b, a)), Gate("CX", (a, b))))
        elif isinstance(g, CZ):
            operations.extend(
                (Gate("H", (g.target,)), Gate("CX", (g.control, g.target)), Gate("H", (g.target,)))
            )
        elif isinstance(g, (ZPhase, XPhase)):
            phase = g.phase % 2
            if phase > 1:
                phase -= 2
            if phase:
                name = "RZ" if isinstance(g, ZPhase) else "RX"
                operations.append(Gate(name, (g.target,), float(phase) * pi))
        else:
            raise ValueError(f"Unsupported extracted gate: {type(g).__name__}")
    if any(q >= qubits for g in operations for q in g.qubits):
        raise ValueError("ZX extraction returned an invalid qubit")
    return local_simplify(operations)


def zx_rewrite(qubits, operations, deadline_ns, max_output_gates):
    """Rewrite operations with PyZX until the deadline or gate limit is reached.

    A cooperative deadline: a PyZX call already running cannot be preempted.
    """
    import pyzx

    def deadline():
        """Return whether the rewrite deadline has elapsed."""
        if perf_counter_ns() >= deadline_ns:
            raise TimeoutError("soft ZX budget exhausted; active PyZX calls are not preempted")

    deadline()
    original = to_pyzx(qubits, operations)
    graph = original.to_graph()
    pyzx.full_reduce(graph)
    deadline()
    extracted = pyzx.extract_circuit(graph, up_to_perm=False)
    deadline()
    output = from_pyzx(extracted, qubits)
    if len(output) > max_output_gates:
        raise ValueError("extracted circuit exceeds output gate limit")
    if original.verify_equality(extracted, up_to_swaps=False, up_to_global_phase=True) is not True:
        raise ValueError("ZX equality verification did not succeed")
    deadline()
    return output


def rewrite_programs(requests, mode, options):
    """Build original, locally rewritten, and optionally ZX-rewritten requests.

    Return local and ZX request batches plus immutable diagnostics; no simulation.
    """
    from .ir import Program, Request

    start = perf_counter_ns()
    deadline = start + int(options.soft_timeout_ms * 1e6)
    resolved, rewritten, programs = {}, {}, {}
    local_requests, zx_requests, records = [], [], []
    attempts = 0
    for request in requests:
        p = request.program
        if p not in resolved:
            resolved[p] = p.resolved_operations()
        operations = resolved[p]
        key = (p.n_qubits, operations)
        if key not in rewritten:
            mark = perf_counter_ns()
            local = local_simplify(operations)
            output, count = local, None
            status, detail = "local", "adjacent inverse and zero-rotation cleanup"
            if mode == "zx":
                if p.n_qubits > options.max_qubits or len(operations) > options.max_gates:
                    status, detail = "skipped", "ZX input width or gate limit"
                elif attempts >= options.max_circuits:
                    status, detail = "skipped", "ZX circuit-attempt limit"
                elif not local:
                    status, detail = "skipped", "local circuit is already empty"
                else:
                    attempts += 1
                    try:
                        output = zx_rewrite(p.n_qubits, local, deadline, 4 * options.max_gates)
                        count = len(output)
                        status, detail = (
                            "verified",
                            "fixed wire order; equivalent up to global phase",
                        )
                    except Exception as exc:
                        status, detail = "fallback", f"{type(exc).__name__}: {exc}"
            rewritten[key] = (local, output)
            records.append(
                RewriteRecord(
                    p.n_qubits,
                    len(operations),
                    len(local),
                    count,
                    status,
                    detail,
                    perf_counter_ns() - mark,
                )
            )
        local, output = rewritten[key]
        # Reuse the original logical naming; parameters have been resolved explicitly.
        if p not in programs:
            programs[p] = (
                p if local == operations else Program(p.qubits, local),
                p if output == operations else Program(p.qubits, output),
            )
        local_program, zx_program = programs[p]
        local_requests.append(Request(local_program, request.observable))
        zx_requests.append(Request(zx_program, request.observable))
    return tuple(local_requests), tuple(zx_requests), tuple(records), perf_counter_ns() - start
