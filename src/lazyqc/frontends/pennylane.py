"""PennyLane device plugin: run QNodes on LazyQC.

``qml.device("lazyqc.qubit", wires=n)`` plans every tape of a batch together,
so the ``2P + 1`` shifted tapes built by ``diff_method="parameter-shift"`` (or
the device's own derivatives, ``diff_method="best"``) share one plan.

Scope: analytic ``expval`` of Pauli words and weighted Pauli sums over the IR
gate set (H, X, Y, Z, RX, RY, RZ, CX); other operations are decomposed.

Device derivatives run the two-term shift rule inside LazyQC and emit shifted
requests only for parameters inside the causal cone of an observable term;
all other entries are structural zeros and are counted in ``last_stats``.
"""

from dataclasses import replace
from math import pi

import numpy as np

try:
    import pennylane as qml
    from pennylane.devices import Device, ExecutionConfig
    from pennylane.devices.preprocess import (
        decompose,
        validate_device_wires,
        validate_measurements,
        validate_observables,
    )
    from pennylane.measurements import ExpectationMP
    from pennylane.transforms.core import CompilePipeline
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The LazyQC PennyLane device needs PennyLane: pip install 'lazyqc[pennylane]'"
    ) from exc

from ..ir import Gate, Program
from ..planner import plan_all

_PL_TO_IR = {
    "Hadamard": "H",
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "RX": "RX",
    "RY": "RY",
    "RZ": "RZ",
    "CNOT": "CX",
    "CZ": "CZ",
    "IsingZZ": "RZZ",
}
_IGNORED = frozenset({"Identity", "GlobalPhase"})
_SUPPORTED = frozenset(_PL_TO_IR) | _IGNORED
_ROTATIONS = frozenset({"RX", "RY", "RZ", "IsingZZ"})


def _supports_operation(op) -> bool:
    return op.name in _SUPPORTED


def _supports_observable(obs) -> bool:
    return getattr(obs, "pauli_rep", None) is not None


def _supports_measurement(mp) -> bool:
    return isinstance(mp, ExpectationMP) and mp.obs is not None and _supports_observable(mp.obs)


def _scalar(value) -> float:
    return float(np.asarray(qml.math.unwrap(value)).reshape(()))


def _in_cone(ops, support):
    """Flags for operations in the backward causal cone of ``support`` wires."""
    active, flags = set(support), [False] * len(ops)
    for i in range(len(ops) - 1, -1, -1):
        wires = set(ops[i][1])
        if wires & active:
            flags[i] = True
            active |= wires
    return flags


class LazyQubit(Device):
    """Exact analytic PennyLane device backed by the LazyQC batch planner.

    Args:
        wires: Device wires (labels or a count); inferred from each tape if omitted.
        shots: Must be ``None``; the device is analytic.
        **plan_options: Keyword options forwarded to :func:`lazyqc.plan_all`.
    """

    name = "lazyqc.qubit"

    def __init__(self, wires=None, shots=None, **plan_options):
        if shots is not None:
            raise ValueError("lazyqc.qubit is analytic; finite shots are not supported")
        super().__init__(wires=wires, shots=shots)
        self._plan_options = plan_options
        self.last_stats = {}

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        """Device derivatives (cone-restricted shift rule) for first-order gradients."""
        if execution_config is None:
            return True
        return execution_config.gradient_method in ("device", "best") and (
            execution_config.derivative_order == 1
        )

    def setup_execution_config(self, config=None, circuit=None):
        """Select device derivatives for ``diff_method="best"``/``"device"``."""
        config = config or ExecutionConfig()
        if config.gradient_method in (None, "best", "device"):
            return replace(
                config,
                gradient_method="device",
                use_device_gradient=True,
                use_device_jacobian_product=False,
                grad_on_execution=False,
                convert_to_numpy=True,
            )
        return config

    def preprocess_transforms(self, execution_config=None):
        """Decompose to the IR gate set and validate wires, measurements, observables."""
        program = CompilePipeline()
        program.add_transform(
            decompose, stopping_condition=_supports_operation, name=self.name
        )
        program.add_transform(qml.transforms.broadcast_expand)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(
            validate_measurements,
            analytic_measurements=_supports_measurement,
            name=self.name,
        )
        program.add_transform(
            validate_observables,
            stopping_condition=_supports_observable,
            name=self.name,
        )
        return program

    # -- translation -----------------------------------------------------
    def _wire_index(self, tape):
        wires = self.wires if self.wires is not None else tape.wires
        return {w: i for i, w in enumerate(wires)}, len(wires)

    def _translate(self, tape):
        """Return (n_qubits, ops, terms, offset-free parameter map)."""
        index, n = self._wire_index(tape)
        ops, slots, count = [], {}, 0
        for op in tape.operations:
            if op.name in _IGNORED:
                count += len(op.data)
                continue
            angle = None
            if op.name in _ROTATIONS:
                slots[count] = len(ops)
                angle = _scalar(op.data[0])
            count += len(op.data)
            ops.append((_PL_TO_IR[op.name], tuple(index[w] for w in op.wires), angle))
        measurements = []
        for mp in tape.measurements:
            terms = []
            for word, coeff in mp.obs.pauli_rep.items():
                coeff = complex(coeff)
                terms.append((tuple((index[w], p) for w, p in word.items()), coeff.real))
            measurements.append(terms)
        return n, ops, measurements, slots

    @staticmethod
    def _program(n, ops, shift=None):
        gates = []
        for i, (name, qubits, angle) in enumerate(ops):
            if shift is not None and shift[0] == i:
                angle += shift[1]
            gates.append(Gate(name, qubits, angle))
        return Program(n, tuple(gates))

    def _collect(self, requests):
        if not requests:
            return []
        return list(plan_all(requests, **self._plan_options).collect().values)

    # -- execution -------------------------------------------------------
    def execute(self, circuits, execution_config=None):
        """Evaluate every tape of the batch in a single plan."""
        single = not isinstance(circuits, (list, tuple))
        circuits = [circuits] if single else list(circuits)
        requests, layout = [], []
        for tape in circuits:
            n, ops, measurements, _ = self._translate(tape)
            program = self._program(n, ops)
            per_measurement = []
            for terms in measurements:
                per_term = []
                for factors, coeff in terms:
                    per_term.append((len(requests), coeff))
                    requests.append(program.expectation(dict(factors)))
                per_measurement.append(per_term)
            layout.append(per_measurement)
        values = self._collect(requests)
        self.last_stats = {"tapes": len(circuits), "requests": len(requests)}
        results = []
        for per_measurement in layout:
            out = tuple(sum(c * values[i] for i, c in terms) for terms in per_measurement)
            results.append(out[0] if len(out) == 1 else out)
        return results[0] if single else tuple(results)

    # -- derivatives -----------------------------------------------------
    def compute_derivatives(self, circuits, execution_config=None):
        """Jacobians via the shift rule, skipping parameters outside every cone."""
        single = not isinstance(circuits, (list, tuple))
        circuits = [circuits] if single else list(circuits)
        requests, layout, skipped, total = [], [], 0, 0
        for tape in circuits:
            n, ops, measurements, slots = self._translate(tape)
            trainable = list(tape.trainable_params)
            cones = [
                [_in_cone(ops, [q for q, _ in factors]) for factors, _ in terms]
                for terms in measurements
            ]
            per_param = []
            for p in trainable:
                if p not in slots:
                    raise ValueError(
                        "lazyqc.qubit differentiates only RX, RY and RZ parameters; "
                        "other trainable parameters must be decomposed first"
                    )
                position = slots[p]
                plus = self._program(n, ops, (position, pi / 2))
                minus = self._program(n, ops, (position, -pi / 2))
                per_measurement = []
                for terms, term_cones in zip(measurements, cones):
                    entries = []
                    for (factors, coeff), cone in zip(terms, term_cones):
                        total += 1
                        if not cone[position]:
                            skipped += 1
                            continue
                        obs = dict(factors)
                        entries.append(
                            (
                                len(requests),
                                coeff,
                            )
                        )
                        requests.append(plus.expectation(obs))
                        requests.append(minus.expectation(obs))
                    per_measurement.append(entries)
                per_param.append(per_measurement)
            layout.append((len(measurements), per_param))
        values = self._collect(requests)
        self.last_stats = {
            "tapes": len(circuits),
            "requests": len(requests),
            "shift_terms": total,
            "skipped_structural_zeros": skipped,
        }
        results = []
        for n_meas, per_param in layout:
            by_measurement = []
            for m in range(n_meas):
                grads = []
                for per_measurement in per_param:
                    grads.append(
                        sum(
                            (c * 0.5 * (values[i] - values[i + 1]) for i, c in per_measurement[m]),
                            0.0,
                        )
                    )
                by_measurement.append(tuple(grads))
            results.append(by_measurement[0] if n_meas == 1 else tuple(by_measurement))
        return results[0] if single else tuple(results)
