"""Restricted Rhyme-style phase/interference/permutation tables, never Python callbacks.

The inspected Rhyme repositories do not expose a stable circuit export. This
explicit interchange subset preserves logical basis semantics without importing
or evaluating live qobject/lqint instances or callable bytecode.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from numbers import Real

from ..ir import GATES, ROTATIONS, Gate, Program


@dataclass(frozen=True)
class ProbabilityQuery:
    requests: tuple
    coefficients: tuple[float, ...]

    def evaluate(self, values):
        """Evaluate the probability query against a mapping of variable values."""
        values = tuple(values)
        if len(values) != len(self.coefficients) or any(not math.isfinite(v) for v in values):
            raise ValueError("Provide every finite expectation in query order")
        return math.fsum(a * b for a, b in zip(self.coefficients, values, strict=True))


@dataclass(frozen=True)
class ImportedProgram:
    program: Program
    subsystems: tuple[tuple[str, tuple, tuple[int, ...]], ...]
    source_sha256: str
    global_phase: float

    def probability(self, subsystem, value):
        """Return the probability of one encoded subsystem value."""
        entries = {name: (basis, wires) for name, basis, wires in self.subsystems}
        if subsystem not in entries:
            raise ValueError("Unknown logical subsystem")
        basis, wires = entries[subsystem]
        if value not in basis:
            raise ValueError("Value is not in the declared logical basis")
        return self._code_probability(wires, basis.index(value))

    def _code_probability(self, wires, code):
        """Compute the probability associated with one binary wire code."""
        requests, weights = [], []
        for mask in range(1 << len(wires)):
            requests.append(
                self.program.expectation({q: "Z" for i, q in enumerate(wires) if (mask >> i) & 1})
            )
            weights.append((-1.0 if (mask & code).bit_count() % 2 else 1.0) / (1 << len(wires)))
        return ProbabilityQuery(tuple(requests), tuple(weights))

    def invalid_probabilities(self, subsystem):
        """Return probabilities that are invalid for the requested subsystem."""
        for name, basis, wires in self.subsystems:
            if name == subsystem:
                return tuple(
                    self._code_probability(wires, i) for i in range(len(basis), 1 << len(wires))
                )
        raise ValueError("Unknown logical subsystem")


def from_rhyme_ir(data):
    """Convert a Rhyme IR mapping into an immutable imported program.

    Compile JSON-compatible finite basis tables (dimensions 2..4; local width <=4).

    Local gate matrices are compiler descriptions, not allocated quantum states.
    All padded basis states remain fixed. Native execution is deferred to collect.
    """
    import numpy as np
    from qiskit import QuantumCircuit, transpile

    if (
        not isinstance(data, dict)
        or set(data) != {"schema_version", "subsystems", "operations"}
        or data["schema_version"] != 1
    ):
        raise ValueError("Use the version-1 declarative Rhyme interchange schema")
    try:
        encoded = json.dumps(data, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Rhyme import accepts finite data tables, not callables or live objects"
        ) from exc
    specs, names, offset = [], set(), 0
    for spec in data["subsystems"]:
        if not isinstance(spec, dict) or set(spec) != {"name", "basis"}:
            raise ValueError("Subsystems require name and explicit basis")
        name, basis = spec["name"], tuple(spec["basis"])
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Subsystem names must be distinct nonempty strings")
        if (
            not 2 <= len(basis) <= 4
            or any(type(v) not in (str, int) for v in basis)
            or len(set(basis)) != len(basis)
        ):
            raise ValueError("Declare two to four distinct string/integer basis values")
        width = (len(basis) - 1).bit_length()
        specs.append((name, basis, tuple(range(offset, offset + width))))
        names.add(name)
        offset += width
    if not specs or offset > 16:
        raise ValueError("Use nonempty logical subsystems requiring at most 16 encoded qubits")
    circuit = QuantumCircuit(offset)
    lookup = {name: (basis, wires) for name, basis, wires in specs}
    for operation in data["operations"]:
        if not isinstance(operation, dict) or operation.get("kind") not in (
            "phase",
            "interfere",
            "permutation",
        ):
            raise ValueError("Supported declarative operations: phase, interfere, permutation")
        kind = operation["kind"]
        payload = {"phase": "angles", "interfere": "pairs", "permutation": "mapping"}[kind]
        if set(operation) != {"kind", "targets", payload}:
            raise ValueError("Unexpected operation fields")
        targets = operation["targets"]
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(t not in lookup for t in targets)
        ):
            raise ValueError("Invalid logical targets")
        selected = [lookup[t] for t in targets]
        wires = tuple(q for _, ws in selected for q in ws)
        if len(wires) > 4:
            raise ValueError("Table operations are limited to four encoded qubits")
        # First logical target is least significant, matching our circuit wire order.
        codes = []
        for digits in product(*(range(len(b)) for b, _ in reversed(selected))):
            code, shift = 0, 0
            for digit, (_, ws) in zip(reversed(digits), selected, strict=True):
                code |= digit << shift
                shift += len(ws)
            codes.append(code)
        matrix = np.eye(1 << len(wires), dtype=complex)
        values = operation[payload]
        if kind == "phase":
            if len(values) != len(codes) or any(
                isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v)
                for v in values
            ):
                raise ValueError(
                    "Supply one finite phase angle per valid logical basis combination"
                )
            for code, angle in zip(codes, values, strict=True):
                matrix[code, code] = np.exp(1j * angle)
        elif kind == "interfere":
            used = set()
            for pair in values:
                if (
                    len(pair) != 2
                    or any(type(v) is not int or not 0 <= v < len(codes) for v in pair)
                    or pair[0] == pair[1]
                    or used.intersection(pair)
                ):
                    raise ValueError("Interference pairs must be disjoint valid basis ordinals")
                used.update(pair)
                a, b = (codes[i] for i in pair)
                matrix[a, a] = matrix[a, b] = matrix[b, a] = 1 / math.sqrt(2)
                matrix[b, b] = -1 / math.sqrt(2)
        else:
            if (
                len(values) != len(codes)
                or any(type(v) is not int for v in values)
                or sorted(values) != list(range(len(codes)))
            ):
                raise ValueError("Permutation mapping must be a bijection of valid basis ordinals")
            for src, dst in enumerate(values):
                matrix[codes[src], codes[src]] = 0
            for src, dst in enumerate(values):
                matrix[codes[dst], codes[src]] = 1
        circuit.unitary(matrix, list(wires))
    lowered = transpile(
        circuit,
        basis_gates=sorted(g.lower() for g in GATES - {"CZ", "RZZ"}),
        optimization_level=1,
        seed_transpiler=0,
    )
    gates = []
    for inst in lowered.data:
        name = inst.operation.name.upper()
        gates.append(
            Gate(
                name,
                tuple(lowered.find_bit(q).index for q in inst.qubits),
                float(inst.operation.params[0]) if name in ROTATIONS else None,
            )
        )
    return ImportedProgram(
        Program(offset, tuple(gates)),
        tuple(specs),
        hashlib.sha256(encoded.encode()).hexdigest(),
        float(lowered.global_phase),
    )
