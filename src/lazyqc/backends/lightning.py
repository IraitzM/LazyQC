"""Optional PennyLane Lightning adapter for running tasks on lightning.gpu or lightning.qubit.

PennyLane is not a LazyQC dependency. Nothing is imported or allocated until the
adapter is constructed. Devices are created per task width on first use and
reused afterwards, so repeated collections do not pay for device creation.
"""

from dataclasses import dataclass
from math import isfinite

from ..instruments import Projector
from ..ir import GATES, ROTATIONS, Parameter
from .qulacs import Counters


@dataclass(frozen=True)
class LightningLowered:
    tape: object
    observables: tuple[int, ...]
    width: int


class LightningBackend:
    def __init__(self, device="lightning.gpu"):
        """Create an adapter for a Lightning device name, with zeroed counters."""
        import pennylane as qml

        self.device = device
        self._qml = qml
        self._devices = {}
        self._fixed = {"H": qml.Hadamard, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
        self.counters = Counters()

    def capabilities(self):
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": self.device,
            "gates": tuple(sorted(GATES)),
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex128",
        }

    def close(self):
        """Release every device and its state memory."""
        self._devices.clear()

    def _device_for(self, width):
        device = self._devices.get(width)
        if device is None:
            device = self._devices[width] = self._qml.device(self.device, wires=width)
        return device

    def lower(self, task):
        """Lower a unitary task into a PennyLane tape measuring every requested Pauli."""
        qml = self._qml
        if any(isinstance(g, Projector) for g in task.operations):
            raise ValueError("LightningBackend executes unitary tasks only")
        mapping = {q: i for i, q in enumerate(task.wires)}
        operations = []
        for gate in task.operations:
            if isinstance(gate.angle, Parameter):
                raise ValueError("Cannot lower unbound parameters")
            wires = [mapping[q] for q in gate.qubits]
            if gate.name == "CX":
                operations.append(qml.CNOT(wires=wires))
            elif gate.name == "CZ":
                operations.append(qml.CZ(wires=wires))
            elif gate.name == "RZZ":
                # IsingZZ(phi) = exp(-i phi Z Z / 2), the public RZZ convention.
                operations.append(qml.IsingZZ(gate.angle, wires=wires))
            elif gate.name in ROTATIONS:
                # PennyLane shares the public convention R_P(theta) = exp(-i theta P / 2).
                operations.append(getattr(qml, gate.name)(gate.angle, wires=wires[0]))
            else:
                operations.append(self._fixed[gate.name](wires=wires[0]))
        measurements = []
        for _, pauli in task.outputs:
            factors = [self._fixed[p](wires=mapping[q]) for q, p in pauli.factors]
            observable = factors[0] if len(factors) == 1 else qml.prod(*factors)
            measurements.append(qml.expval(observable))
        tape = qml.tape.QuantumScript(operations, measurements)
        return LightningLowered(tape, tuple(range(len(measurements))), task.width)

    def evolve(self, lowered):
        """Execute the tape on the device for its width and keep its expectation values."""
        values = self._device_for(lowered.width).execute(lowered.tape)
        self.counters.allocations += 1
        self.counters.evolutions += 1
        return [values] if len(lowered.observables) == 1 else list(values)

    def expectations(self, state, observables):
        """Read the executed expectations and reject invalid values."""
        values = []
        for index in observables:
            value = float(state[index])
            self.counters.expectations += 1
            if not isfinite(value) or abs(value) > 1.0 + 1e-10:
                raise ArithmeticError(f"Invalid Pauli expectation: {value}")
            values.append(value)
        return tuple(values)
