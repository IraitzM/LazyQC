"""Qulacs adapter. All native allocation happens after collection starts."""

from dataclasses import dataclass
from math import isfinite

from ..instruments import Projector
from ..ir import GATES, ROTATIONS, Parameter


@dataclass
class Counters:
    allocations: int = 0
    evolutions: int = 0
    expectations: int = 0


@dataclass(frozen=True)
class Lowered:
    circuit: object
    observables: tuple
    width: int
    projected: bool = False
    # (parameter index in the circuit, parameter name, offset) per symbolic gate.
    slots: tuple = ()


class QulacsBackend:
    def __init__(self):
        """Create a backend with zeroed allocation and operation counters."""
        self.counters = Counters()

    @staticmethod
    def capabilities():
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": "qulacs",
            "gates": GATES,
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex128",
        }

    def lower(self, task, *, parametric=False):
        """Lower a backend-independent task into Qulacs objects.

        With `parametric`, symbolic gates become Qulacs parametric gates and the result
        records where each one lives, so `rebind` can set new angles without lowering again.
        """
        from qulacs import Observable, ParametricQuantumCircuit, QuantumCircuit

        circuit = ParametricQuantumCircuit(task.width) if parametric else QuantumCircuit(task.width)
        slots = []
        mapping = {q: i for i, q in enumerate(task.wires)}
        for gate in task.operations:
            if isinstance(gate, Projector):
                from qulacs.gate import P0, P1

                circuit.add_gate((P0 if gate.outcome == 0 else P1)(mapping[gate.qubits[0]]))
                continue
            qs = [mapping[q] for q in gate.qubits]
            if isinstance(gate.angle, Parameter):
                if not parametric:
                    raise ValueError("Cannot lower unbound parameters")
                slots.append((circuit.get_parameter_count(), gate.angle.name, gate.angle.offset))
                if gate.name == "RZZ":
                    circuit.add_parametric_multi_Pauli_rotation_gate(qs, [3, 3], 0.0)
                else:
                    getattr(circuit, f"add_parametric_{gate.name}_gate")(qs[0], 0.0)
                continue
            if gate.name == "CX":
                circuit.add_CNOT_gate(*qs)
            elif gate.name == "CZ":
                circuit.add_CZ_gate(*qs)
            elif gate.name == "RZZ":
                # Same sign convention as the single-qubit rotations above.
                circuit.add_multi_Pauli_rotation_gate(qs, [3, 3], -gate.angle)
            elif gate.name in ROTATIONS:
                # Qulacs uses exp(+i theta P/2); the public IR uses the opposite sign.
                getattr(circuit, f"add_{gate.name}_gate")(qs[0], -gate.angle)
            else:
                getattr(circuit, f"add_{gate.name}_gate")(qs[0])
        observables = []
        for _, pauli in task.outputs:
            observable = Observable(task.width)
            observable.add_operator(1.0, " ".join(f"{p} {mapping[q]}" for q, p in pauli.factors))
            observables.append(observable)
        return Lowered(
            circuit,
            tuple(observables),
            task.width,
            any(isinstance(g, Projector) for g in task.operations),
            tuple(slots),
        )

    @staticmethod
    def rebind(lowered, values):
        """Set a parametric lowering's angles in place (Qulacs rotates with the opposite sign)."""
        circuit = lowered.circuit
        for index, name, offset in lowered.slots:
            circuit.set_parameter(index, -(values[name] + offset))

    def evolve(self, lowered):
        """Allocate, evolve, and normalization-check a Qulacs state."""
        from qulacs import QuantumState

        state = QuantumState(lowered.width)
        self.counters.allocations += 1
        state.set_zero_state()
        lowered.circuit.update_quantum_state(state)
        self.counters.evolutions += 1
        norm = state.get_squared_norm()
        if (
            not isfinite(norm)
            or (not lowered.projected and abs(norm - 1.0) > 1e-10)
            or (lowered.projected and not -1e-12 <= norm <= 1.0 + 1e-10)
        ):
            raise ArithmeticError(f"State normalization failed: {norm}")
        return state

    def expectations(self, state, observables):
        """Evaluate observables and reject invalid expectation values."""
        values = []
        for observable in observables:
            value = float(observable.get_expectation_value(state))
            self.counters.expectations += 1
            if not isfinite(value) or abs(value) > 1.0 + 1e-10:
                raise ArithmeticError(f"Invalid Pauli expectation: {value}")
            values.append(value)
        return tuple(values)
