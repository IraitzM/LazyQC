"""Optional Google qsim adapter through Cirq's qsimcirq interface."""

from dataclasses import dataclass, field
from math import isfinite

from ..instruments import Projector
from ..ir import GATES, ROTATIONS, Parameter
from .qulacs import Counters

FP32_TOL = 1e-4


@dataclass
class QSimLowered:
    circuit: object
    observables: tuple
    paulis: tuple
    qubits: tuple
    slots: tuple = ()
    symbols: dict = field(default_factory=dict)
    resolver: object = None


class QSimBackend:
    """CPU statevector backend using Google's single-precision qsim."""

    def __init__(self, threads=None):
        if threads is not None and (type(threads) is not int or threads < 1):
            raise ValueError("threads must be None or a positive integer")
        import cirq
        import qsimcirq
        import sympy

        self._cirq = cirq
        self._sympy = sympy
        options = qsimcirq.QSimOptions(cpu_threads=threads) if threads is not None else None
        self._simulator = qsimcirq.QSimSimulator(qsim_options=options)
        self.counters = Counters()

    def capabilities(self):
        """Return the gates, state convention, and numerical precision supported."""
        return {
            "name": "qsim-cpu",
            "gates": tuple(sorted(GATES)),
            "initial_state": "zero",
            "evaluation": "exact_pauli",
            "precision": "complex64",
        }

    def lower(self, task, *, parametric=False):
        """Lower one unitary task into a Cirq circuit and Pauli observables."""
        if any(isinstance(gate, Projector) for gate in task.operations):
            raise ValueError("QSimBackend executes unitary tasks only")

        cirq = self._cirq
        qubits = tuple(cirq.LineQubit.range(task.width))
        mapping = {wire: qubit for wire, qubit in zip(task.wires, qubits, strict=True)}
        symbols = {}
        operations = []

        for gate in task.operations:
            wires = [mapping[wire] for wire in gate.qubits]
            angle = gate.angle
            if isinstance(angle, Parameter):
                if not parametric:
                    raise ValueError("Cannot lower unbound parameters")
                symbol = symbols.setdefault(angle.name, self._sympy.Symbol(angle.name))
                angle = symbol + angle.offset if angle.offset else symbol

            if gate.name == "CX":
                operations.append(cirq.CNOT(*wires))
            elif gate.name == "CZ":
                operations.append(cirq.CZ(*wires))
            elif gate.name == "RZZ":
                operations.extend(
                    (
                        cirq.CNOT(*wires),
                        cirq.rz(angle).on(wires[1]),
                        cirq.CNOT(*wires),
                    )
                )
            elif gate.name in ROTATIONS:
                operations.append(getattr(cirq, gate.name.lower())(angle).on(wires[0]))
            else:
                operations.append(getattr(cirq, gate.name)(wires[0]))

        observables = tuple(
            cirq.PauliString(
                {mapping[wire]: getattr(cirq, letter) for wire, letter in pauli.factors}
            )
            for _, pauli in task.outputs
        )
        return QSimLowered(
            cirq.Circuit(operations),
            tuple(range(len(observables))),
            observables,
            qubits,
            tuple(symbols),
            symbols,
        )

    def rebind(self, lowered, values):
        """Bind new values to the symbols in a previously lowered circuit."""
        lowered.resolver = self._cirq.ParamResolver(
            {symbol: values[name] for name, symbol in lowered.symbols.items()}
        )

    def evolve(self, lowered):
        """Run qsim once for every requested Pauli expectation in the task."""
        options = {}
        if lowered.resolver is not None:
            options["param_resolver"] = lowered.resolver
        values = self._simulator.simulate_expectation_values(
            lowered.circuit,
            observables=list(lowered.paulis),
            qubit_order=lowered.qubits,
            **options,
        )
        self.counters.allocations += 1
        self.counters.evolutions += 1
        return values

    def expectations(self, state, observables):
        """Read expectations and reject non-finite or invalid Pauli values."""
        values = []
        for index in observables:
            value = complex(state[index])
            if not isfinite(value.real) or not isfinite(value.imag):
                raise ArithmeticError(f"Invalid Pauli expectation: {value}")
            if abs(value.imag) > FP32_TOL:
                raise ArithmeticError(f"Complex Pauli expectation: {value}")
            real = float(value.real)
            self.counters.expectations += 1
            if not isfinite(real) or abs(real) > 1.0 + FP32_TOL:
                raise ArithmeticError(f"Invalid Pauli expectation: {real}")
            values.append(real)
        return tuple(values)
