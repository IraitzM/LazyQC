"""PennyLane device plugin: run QNodes on LazyQC.

``qml.device("lazyqc.qubit", wires=n)`` plans every tape of a batch together,
so the ``2P + 1`` shifted tapes built by ``diff_method="parameter-shift"`` (or
the device's own derivatives, ``diff_method="best"``) share one plan.

Scope: analytic ``expval`` of Pauli words and weighted Pauli sums over the IR
gate set (H, X, Y, Z, RX, RY, RZ, CX); other operations are decomposed.

Device derivatives run the two-term shift rule inside LazyQC and emit shifted
requests only for parameters inside the causal cone of an observable term;
all other entries are structural zeros and are counted in ``last_stats``.

A QNode evaluated repeatedly (a training loop) sends the same tape structure with new
angles. From the second sight of a structure the device keeps a symbolic plan
(`plan_all(symbolic=True)`) and passes the angles as ``bindings``, so later calls skip
program construction, cone analysis, planning and lowering. Values equal the fresh-plan
path exactly; ``last_stats["plan_reused"]`` reports it. Options that need numeric angles
(``optimization``, ``cutting``, ``workers``, ``total_state_bytes``, ``cost_model``) keep
the fresh-plan path.
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

from ..ir import Gate, Parameter, Program
from ..passes import commuting_cone
from ..planner import plan_all, symbolic_compatible

_PLAN_CACHE = 4

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


def _in_cone(ops, factors):
    """Flags for operations in the backward cone of the Pauli `factors`."""
    _, retained = commuting_cone([(name, qubits) for name, qubits, _ in ops], factors)
    flags = [False] * len(ops)
    for i in retained:
        flags[i] = True
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
        self._plans = {}  # tape-batch structure -> (symbolic plan, layout, stats), oldest first
        self._once = {}  # structures seen once, oldest first; a repeat earns a symbolic plan

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
    def _program(n, ops, shift=None, prefix=None):
        """Build the IR program; with `prefix`, rotation angles become named parameters."""
        gates = []
        for i, (name, qubits, angle) in enumerate(ops):
            if prefix is not None and angle is not None:
                angle = Parameter(f"{prefix}{i}")
            if shift is not None and shift[0] == i:
                angle = angle.shifted(shift[1]) if prefix is not None else angle + shift[1]
            gates.append(Gate(name, qubits, angle))
        return Program(n, tuple(gates))

    @staticmethod
    def _structure(translated, extra=()):
        """Return a hashable key for everything about a tape batch except its angles."""
        return (
            tuple(
                (
                    n,
                    tuple((name, qubits, angle is not None) for name, qubits, angle in ops),
                    tuple(tuple(terms) for terms in measurements),
                )
                for n, ops, measurements, _ in translated
            ),
            extra,
        )

    @staticmethod
    def _bindings(translated):
        """Return ``{name: angle}`` for the rotation angles of every tape."""
        return {
            f"{t}.{i}": angle
            for t, (_, ops, _, _) in enumerate(translated)
            for i, (_, _, angle) in enumerate(ops)
            if angle is not None
        }

    def _evaluate(self, translated, key, build):
        """Return ``(values, layout, stats)`` for the requests ``build(prefixes)`` creates.

        `build(symbolic)` returns ``(requests, layout, stats)``. The first sight of a structure
        takes the fresh-plan path; a repeat builds a symbolic plan that later calls reuse.
        """
        if not symbolic_compatible(self._plan_options):
            requests, layout, stats = build(False)
            return self._collect(requests), layout, {**stats, "plan_reused": False}
        entry = self._plans.pop(key, None)
        if entry is not None:
            self._plans[key] = entry  # most recently used goes last
            plan, layout, stats = entry
            values = list(plan.collect(bindings=self._bindings(translated)).values) if plan else []
            return values, layout, {**stats, "plan_reused": True}
        if key not in self._once:
            self._once[key] = None
            while len(self._once) > 4 * _PLAN_CACHE:
                self._once.pop(next(iter(self._once)))
            requests, layout, stats = build(False)
            return self._collect(requests), layout, {**stats, "plan_reused": False}
        self._once.pop(key)
        requests, layout, stats = build(True)
        plan = plan_all(requests, symbolic=True, **self._plan_options) if requests else None
        self._plans[key] = (plan, layout, stats)
        while len(self._plans) > _PLAN_CACHE:
            self._plans.pop(next(iter(self._plans)))
        bindings = self._bindings(translated)
        values = list(plan.collect(bindings=bindings).values) if plan else []
        return values, layout, {**stats, "plan_reused": False}

    def _collect(self, requests):
        if not requests:
            return []
        return list(plan_all(requests, **self._plan_options).collect().values)

    # -- execution -------------------------------------------------------
    def execute(self, circuits, execution_config=None):
        """Evaluate every tape of the batch in a single plan."""
        single = not isinstance(circuits, (list, tuple))
        circuits = [circuits] if single else list(circuits)
        translated = [self._translate(tape) for tape in circuits]

        def build(symbolic):
            requests, layout = [], []
            for t, (n, ops, measurements, _) in enumerate(translated):
                program = self._program(n, ops, prefix=f"{t}." if symbolic else None)
                per_measurement = []
                for terms in measurements:
                    per_term = []
                    for factors, coeff in terms:
                        per_term.append((len(requests), coeff))
                        requests.append(program.expectation(dict(factors)))
                    per_measurement.append(per_term)
                layout.append(per_measurement)
            return requests, layout, {"tapes": len(circuits), "requests": len(requests)}

        values, layout, stats = self._evaluate(translated, self._structure(translated), build)
        self.last_stats = stats
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
        translated = [self._translate(tape) for tape in circuits]
        trainable = tuple(tuple(tape.trainable_params) for tape in circuits)

        def build(symbolic):
            requests, layout, skipped, total = [], [], 0, 0
            for t, (n, ops, measurements, slots) in enumerate(translated):
                prefix = f"{t}." if symbolic else None
                cones = [
                    [_in_cone(ops, factors) for factors, _ in terms]
                    for terms in measurements
                ]
                per_param = []
                for p in trainable[t]:
                    if p not in slots:
                        raise ValueError(
                            "lazyqc.qubit differentiates only RX, RY and RZ parameters; "
                            "other trainable parameters must be decomposed first"
                        )
                    position = slots[p]
                    plus = self._program(n, ops, (position, pi / 2), prefix)
                    minus = self._program(n, ops, (position, -pi / 2), prefix)
                    per_measurement = []
                    for terms, term_cones in zip(measurements, cones):
                        entries = []
                        for (factors, coeff), cone in zip(terms, term_cones):
                            total += 1
                            if not cone[position]:
                                skipped += 1
                                continue
                            obs = dict(factors)
                            entries.append((len(requests), coeff))
                            requests.append(plus.expectation(obs))
                            requests.append(minus.expectation(obs))
                        per_measurement.append(entries)
                    per_param.append(per_measurement)
                layout.append((len(measurements), per_param))
            stats = {
                "tapes": len(circuits),
                "requests": len(requests),
                "shift_terms": total,
                "skipped_structural_zeros": skipped,
            }
            return requests, layout, stats

        # Parameter indices map to operations through the slots, which ignored operations shift.
        slot_map = tuple(tuple(sorted(slots.items())) for _, _, _, slots in translated)
        key = self._structure(translated, ("derivatives", trainable, slot_map))
        values, layout, stats = self._evaluate(translated, key, build)
        self.last_stats = stats
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
