"""Optional quimb adapter: evaluate tasks by tensor-network contraction.

quimb is not a core LazyQC dependency (install the ``tensor`` extra). Each task
becomes a quimb circuit whose requested expectations are contracted with
quimb's own reverse light-cone simplification; outputs on the same wires share
one contraction. quimb caches simplified networks on a circuit object, so a
fresh circuit is built at every lowering and nothing carries over between
collections.
"""

from dataclasses import dataclass
from math import isfinite

from ..instruments import Projector
from ..ir import ROTATIONS, Parameter
from .qulacs import Counters


@dataclass(frozen=True)
class TensorLowered:
    circuit: object
    groups: tuple[tuple[tuple[int, ...], tuple[str, ...], tuple[int, ...]], ...]
    observables: tuple[int, ...]


def pauli_groups(outputs, mapping):
    """Group outputs by their wires: (wires, Pauli strings, output positions)."""
    grouped = {}
    for position, (_, pauli) in enumerate(outputs):
        wires = tuple(mapping[q] for q, _ in pauli.factors)
        letters = "".join(p for _, p in pauli.factors)
        grouped.setdefault(wires, ([], []))
        grouped[wires][0].append(letters)
        grouped[wires][1].append(position)
    return tuple((w, tuple(ls), tuple(ps)) for w, (ls, ps) in grouped.items())


def contract_groups(circuit, groups, count, optimize):
    """Contract every group of same-wire Paulis; return values in output order."""
    import quimb as qu

    values = [None] * count
    for wires, letters, positions in groups:
        operators = [qu.kron(*(qu.pauli(p) for p in word)) for word in letters]
        result = circuit.local_expectation(
            operators if len(operators) > 1 else operators[0], wires, optimize=optimize
        )
        results = result if len(operators) > 1 else (result,)
        for position, value in zip(positions, results, strict=True):
            values[position] = value
    return values


class QuimbBackend:
    def __init__(self, optimize="auto-hq"):
        """Create a tensor-network adapter with a contraction path optimizer."""
        import quimb.tensor  # noqa: F401  (fail early if the extra is missing)

        self.optimize = optimize
        self.counters = Counters()

    def capabilities(self):
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": "quimb",
            "gates": ("H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"),
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex128",
        }

    def lower(self, task):
        """Lower a unitary task into a fresh quimb circuit and grouped Pauli outputs."""
        import quimb.tensor as qtn

        if any(isinstance(g, Projector) for g in task.operations):
            raise ValueError("QuimbBackend executes unitary tasks only")
        mapping = {q: i for i, q in enumerate(task.wires)}
        circuit = qtn.Circuit(task.width)
        for gate in task.operations:
            if isinstance(gate.angle, Parameter):
                raise ValueError("Cannot lower unbound parameters")
            wires = [mapping[q] for q in gate.qubits]
            if gate.name in ROTATIONS:
                # quimb shares the public convention R_P(theta) = exp(-i theta P / 2).
                circuit.apply_gate(gate.name, gate.angle, *wires)
            else:
                circuit.apply_gate(gate.name, *wires)
        groups = pauli_groups(task.outputs, mapping)
        return TensorLowered(circuit, groups, tuple(range(len(task.outputs))))

    def evolve(self, lowered):
        """Contract the requested expectations; the 'state' is their values."""
        values = contract_groups(
            lowered.circuit, lowered.groups, len(lowered.observables), self.optimize
        )
        self.counters.allocations += 1
        self.counters.evolutions += 1
        return values

    def expectations(self, state, observables):
        """Read contracted expectations and reject invalid values."""
        values = []
        for index in observables:
            value = complex(state[index])
            if abs(value.imag) > 1e-10:
                raise ArithmeticError(f"Complex Pauli expectation: {value}")
            self.counters.expectations += 1
            if not isfinite(value.real) or abs(value.real) > 1.0 + 1e-10:
                raise ArithmeticError(f"Invalid Pauli expectation: {value.real}")
            values.append(value.real)
        return tuple(values)


@dataclass(frozen=True)
class MPSLowered:
    gates: tuple
    width: int
    outputs: tuple[tuple[tuple[int, ...], str], ...]
    observables: tuple[int, ...]


class QuimbMPSBackend:
    """Exact matrix-product-state execution of one whole-program task.

    ``cutoff=0.0`` keeps every singular value, so results are exact and the bond
    follows ``lazyqc.tensor_cost.mps_bond_profile``. All outputs of a task read
    the same MPS, which is applied once.
    """

    def __init__(self):
        """Create a backend with zeroed counters."""
        import quimb.tensor  # noqa: F401  (fail early if the extra is missing)

        self.counters = Counters()

    def capabilities(self):
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": "quimb-mps",
            "gates": ("H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"),
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex128",
        }

    def lower(self, task):
        """Validate a unitary task and record its gates and Pauli outputs."""
        if any(isinstance(g, Projector) for g in task.operations):
            raise ValueError("QuimbMPSBackend executes unitary tasks only")
        mapping = {q: i for i, q in enumerate(task.wires)}
        gates = []
        for gate in task.operations:
            if isinstance(gate.angle, Parameter):
                raise ValueError("Cannot lower unbound parameters")
            wires = tuple(mapping[q] for q in gate.qubits)
            gates.append((gate.name, gate.angle if gate.name in ROTATIONS else None, wires))
        outputs = tuple(
            (
                tuple(mapping[q] for q, _ in pauli.factors),
                "".join(p for _, p in pauli.factors),
            )
            for _, pauli in task.outputs
        )
        return MPSLowered(tuple(gates), task.width, outputs, tuple(range(len(outputs))))

    def evolve(self, lowered):
        """Apply the circuit to an exact MPS and return it."""
        import quimb.tensor as qtn

        mps = qtn.CircuitMPS(lowered.width, cutoff=0.0)
        for name, angle, wires in lowered.gates:
            if angle is None:
                mps.apply_gate(name, *wires)
            else:
                # quimb shares the public convention R_P(theta) = exp(-i theta P / 2).
                mps.apply_gate(name, angle, *wires)
        self.counters.allocations += 1
        self.counters.evolutions += 1
        return (mps, lowered.outputs)

    def expectations(self, state, observables):
        """Read Pauli expectations from the MPS and reject invalid values."""
        import quimb as qu

        mps, outputs = state
        values = []
        for index in observables:
            wires, letters = outputs[index]
            operator = qu.kron(*(qu.pauli(p) for p in letters))
            value = complex(mps.local_expectation(operator, wires))
            if abs(value.imag) > 1e-10:
                raise ArithmeticError(f"Complex Pauli expectation: {value}")
            self.counters.expectations += 1
            if not isfinite(value.real) or abs(value.real) > 1.0 + 1e-10:
                raise ArithmeticError(f"Invalid Pauli expectation: {value.real}")
            values.append(value.real)
        return tuple(values)
