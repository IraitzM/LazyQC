"""Optional Qiskit Aer adapter for placing wide unitary tasks on a GPU statevector.

Qiskit Aer is not a LazyQC dependency. GPU execution needs the ``qiskit-aer-gpu``
wheel, which provides the same ``qiskit_aer`` module as the CPU wheel; install it
in an environment without ``qiskit-aer``. Nothing is imported or allocated until
the adapter is constructed.
"""

from dataclasses import dataclass
from math import isfinite

from ..instruments import Projector
from ..ir import ROTATIONS, Parameter
from .qulacs import Counters


@dataclass(frozen=True)
class AerLowered:
    circuit: object
    observables: tuple[str, ...]
    width: int


class AerBackend:
    def __init__(self, device="GPU", precision="double", **options):
        """Create an Aer statevector simulator on `device` with zeroed counters."""
        from qiskit_aer import AerSimulator

        self.device = device
        self.simulator = AerSimulator(
            method="statevector", device=device, precision=precision, **options
        )
        self.counters = Counters()

    def capabilities(self):
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": f"aer-{self.device.lower()}",
            "gates": ("H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"),
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex128",
        }

    def lower(self, task):
        """Lower a unitary task into an Aer circuit that saves each requested expectation."""
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp

        if any(isinstance(g, Projector) for g in task.operations):
            raise ValueError("AerBackend executes unitary tasks only")
        circuit = QuantumCircuit(task.width)
        mapping = {q: i for i, q in enumerate(task.wires)}
        for gate in task.operations:
            if isinstance(gate.angle, Parameter):
                raise ValueError("Cannot lower unbound parameters")
            qs = [mapping[q] for q in gate.qubits]
            if gate.name == "CX":
                circuit.cx(*qs)
            elif gate.name in ROTATIONS:
                # Qiskit shares the public convention R_P(theta) = exp(-i theta P / 2).
                getattr(circuit, gate.name.lower())(gate.angle, qs[0])
            else:
                getattr(circuit, gate.name.lower())(qs[0])
        labels = []
        for k, (_, pauli) in enumerate(task.outputs):
            operator = SparsePauliOp.from_sparse_list(
                [("".join(p for _, p in pauli.factors), [mapping[q] for q, _ in pauli.factors], 1)],
                task.width,
            )
            labels.append(f"o{k}")
            circuit.save_expectation_value(operator, range(task.width), label=labels[-1])
        return AerLowered(circuit, tuple(labels), task.width)

    def evolve(self, lowered):
        """Run the circuit; the simulator allocates, evolves and evaluates on its device."""
        result = self.simulator.run(lowered.circuit).result()
        self.counters.allocations += 1
        self.counters.evolutions += 1
        if not result.success:
            raise ArithmeticError(f"Aer execution failed: {result.status}")
        return result.data(0)

    def expectations(self, state, observables):
        """Read saved expectations and reject invalid values."""
        values = []
        for label in observables:
            value = state[label]
            if abs(complex(value).imag) > 1e-10:
                raise ArithmeticError(f"Complex Pauli expectation: {value}")
            value = float(complex(value).real)
            self.counters.expectations += 1
            if not isfinite(value) or abs(value) > 1.0 + 1e-10:
                raise ArithmeticError(f"Invalid Pauli expectation: {value}")
            values.append(value)
        return tuple(values)
