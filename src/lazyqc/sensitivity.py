"""Opt-in occurrence gradients and bounded zero-angle candidate planning.

Profiling can use parameter shifts or cone-local adjoint backpropagation. Ordinary
plan_all remains exact and lazy. Bounds concern norm-one terminal Paulis and assume
the caller's evaluation error allowance; they are not interval-arithmetic
certificates. A profile is valid only at its stored program, bindings and observables,
never across optimizer steps.
"""

from dataclasses import dataclass, replace
from itertools import combinations, islice
from math import fsum, isfinite, pi
from time import perf_counter_ns

from .ir import ROTATIONS, Parameter, Program, finite_angle
from .planner import NoFeasiblePlan, plan_all, validate_limits


def _nonnegative(value, name):
    """Convert a value to a finite nonnegative angle or raise `ValueError`."""
    value = finite_angle(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _cone_indices(operations, observable):
    """Return operation indices in the observable's backward structural cone."""
    active = {q for q, _ in observable.factors}
    indices = set()
    for i in reversed(range(len(operations))):
        if active.intersection(operations[i].qubits):
            indices.add(i)
            active.update(operations[i].qubits)
    return indices


@dataclass(frozen=True)
class SensitivityProfile:
    program: Program
    observables: tuple
    gate_indices: tuple[int, ...]
    jacobian: tuple[tuple[float, ...], ...]
    relevant: tuple[tuple[bool, ...], ...]
    evaluation_error: float
    shifted_requests: int
    elapsed_ns: int
    methods: tuple[str, ...] = ()

    def parameter_gradient(self, name):
        """Sum occurrences of a bound symbolic parameter; require complete coverage."""
        occurrences = {
            i
            for i, g in enumerate(self.program.operations)
            if isinstance(g.angle, Parameter) and g.angle.name == name
        }
        if not occurrences or not occurrences.issubset(self.gate_indices):
            raise ValueError("Profile must include every occurrence of the parameter")
        return tuple(
            fsum(row[j] for j, i in enumerate(self.gate_indices) if i in occurrences)
            for row in self.jacobian
        )

    def bounds(self, deltas):
        """Bound output change, including two endpoint evaluation allowances.

        Each first and mixed second derivative with respect to Pauli-rotation
        occurrences has magnitude <= 1. Taylor's integral remainder is <= L^2/2.
        Shift-derived derivatives use the caller's per-evaluation error allowance;
        for adjoint-derived entries, evaluation_error is the assumed derivative error.
        Gates outside the original structural cone cannot affect the output after
        changing angles alone. Identity/unmodified outputs have zero change.
        """
        deltas = tuple(finite_angle(d) for d in deltas)
        if len(deltas) != len(self.gate_indices):
            raise ValueError("One delta is required per profiled gate occurrence")
        result = []
        for row, mask in zip(self.jacobian, self.relevant, strict=True):
            ds = tuple(d if relevant else 0.0 for d, relevant in zip(deltas, mask, strict=True))
            length = fsum(abs(d) for d in ds)
            if not length:
                result.append(0.0)
                continue
            linear = abs(fsum(g * d for g, d in zip(row, ds, strict=True)))
            # Avoid overflow in the remainder for deliberately huge perturbations.
            taylor = linear + self.evaluation_error * length + 0.5 * min(length, 2.0) ** 2
            result.append(min(2.0, length, taylor) + 2 * self.evaluation_error)
        return tuple(result)


def profile_sensitivity(
    program,
    observables,
    gate_indices,
    *,
    evaluation_error,
    method="shift",
    session=None,
    **planning_options,
):
    """Profile occurrence gradients using shifts or cone-local adjoint backpropagation.

    evaluation_error is the assumed absolute error allowance for gradient entries.
    For shifts this derives from the expectation-evaluation error; for adjoint
    entries the caller supplies an absolute derivative-error allowance.
    max_state_* and workers options are forwarded to the ordinary planner. Callers
    own any persistent ExecutionSession and must account for its startup/shutdown.
    ``auto`` selects adjoint when its estimated three state sweeps are cheaper than
    the two-shift sweeps for the selected rotations in each observable cone.
    """
    start = perf_counter_ns()
    evaluation_error = _nonnegative(evaluation_error, "evaluation_error")
    if method not in ("shift", "adjoint", "auto"):
        raise ValueError("method must be 'shift', 'adjoint' or 'auto'")
    indices = tuple(gate_indices)
    operations = program.resolved_operations()
    if len(set(indices)) != len(indices) or any(
        type(i) is not int or not 0 <= i < len(operations) or operations[i].name not in ROTATIONS
        for i in indices
    ):
        raise ValueError("gate_indices must be distinct rotation occurrence indices")
    requests = tuple(program.expectation(o) for o in observables)
    if not requests:
        raise ValueError("At least one observable is required")
    observables = tuple(r.observable for r in requests)
    cones = tuple(_cone_indices(operations, o) for o in observables)
    relevant = tuple(tuple(i in cone for i in indices) for cone in cones)
    shifted, positions = [], []
    methods = []
    jacobian = [[0.0] * len(indices) for _ in observables]
    for k, observable in enumerate(observables):
        selected = sum(relevant[k])
        use_adjoint = method == "adjoint" or (
            method == "auto" and selected > 0 and 2 * selected > 3
        )
        selected_method = "adjoint" if use_adjoint else "shift"
        methods.append(selected_method)
        if use_adjoint and selected:
            wires = _cone_wires(operations, observable, cones[k])
            _validate_adjoint_limits(wires, planning_options)
            row = _adjoint_task_gradient(
                operations,
                wires,
                cones[k],
                ((1.0, observable),),
            )
            for j, i in enumerate(indices):
                jacobian[k][j] = row.get(i, 0.0)
    for j, i in enumerate(indices):
        outputs = tuple(
            k for k in range(len(observables))
            if methods[k] == "shift" and relevant[k][j]
        )
        if not outputs:
            continue
        for sign in (1, -1):
            gates = list(operations)
            angle = gates[i].angle
            if abs(angle) > 8 * pi:
                raise ValueError("Sensitivity profiling supports angles within 8*pi")
            gates[i] = replace(gates[i], angle=angle + sign * pi / 2)
            variant = Program(program.qubits, tuple(gates))
            for k in outputs:
                shifted.append(variant.expectation(observables[k]))
                positions.append((k, j, sign))
    if shifted:
        plan = plan_all(shifted, **planning_options)
        values = plan.collect(**({"session": session} if session is not None else {})).values
        for (k, j, sign), value in zip(positions, values, strict=True):
            if not isfinite(value):
                raise ValueError("Nonfinite sensitivity evaluation")
            jacobian[k][j] += sign * value / 2
    return SensitivityProfile(
        program,
        observables,
        indices,
        tuple(map(tuple, jacobian)),
        relevant,
        evaluation_error,
        len(shifted),
        perf_counter_ns() - start,
        tuple(methods),
    )


def _validate_adjoint_limits(wires, planning_options):
    """Reject adjoint state-vector tasks that exceed planner resource limits."""
    max_state_qubits = planning_options.get("max_state_qubits", 24)
    max_state_bytes = planning_options.get("max_state_bytes", 256 * 2**20)
    validate_limits(planning_options.get("workers", 1), max_state_qubits, max_state_bytes)
    total_state_bytes = planning_options.get("total_state_bytes")
    if total_state_bytes is not None and (
        type(total_state_bytes) is not int or total_state_bytes < 1
    ):
        raise ValueError("total_state_bytes must be a positive integer")
    state_bytes = 16 * (1 << len(wires))
    byte_limit = min(
        max_state_bytes,
        total_state_bytes if total_state_bytes is not None else max_state_bytes,
    )
    if len(wires) > max_state_qubits or state_bytes > byte_limit:
        raise NoFeasiblePlan(
            f"Adjoint task width {len(wires)} requires {state_bytes} bytes, exceeding "
            f"limits of {max_state_qubits} qubits and {byte_limit} bytes"
        )


def _cone_wires(operations, observable, cone):
    """Return all logical wires retained by a backward structural cone."""
    wires = {q for q, _ in observable.factors}
    for i in cone:
        wires.update(operations[i].qubits)
    return tuple(sorted(wires))


def _adjoint_task_gradient(operations, wires, cone, weighted_observables):
    """Backpropagate a weighted Pauli sum through one compact local cone."""
    from qulacs import Observable, ParametricQuantumCircuit

    if not any(weight and observable.factors for weight, observable in weighted_observables):
        return {}
    mapping = {q: i for i, q in enumerate(wires)}
    circuit = ParametricQuantumCircuit(len(wires))
    parameter_gates = []
    for i, gate in enumerate(operations):
        if i not in cone:
            continue
        qs = [mapping[q] for q in gate.qubits]
        if gate.name == "CX":
            circuit.add_CNOT_gate(*qs)
        elif gate.name == "CZ":
            circuit.add_CZ_gate(*qs)
        elif gate.name == "RZZ":
            circuit.add_parametric_multi_Pauli_rotation_gate(qs, [3, 3], -gate.angle)
            parameter_gates.append(i)
        elif gate.name in ROTATIONS:
            # Qulacs rotates with the opposite sign to LazyQC's IR convention.
            getattr(circuit, f"add_parametric_{gate.name}_gate")(qs[0], -gate.angle)
            parameter_gates.append(i)
        else:
            getattr(circuit, f"add_{gate.name}_gate")(qs[0])
    combined = Observable(len(wires))
    for weight, observable in weighted_observables:
        if weight and observable.factors:
            pauli = " ".join(f"{axis} {mapping[q]}" for q, axis in observable.factors)
            combined.add_operator(weight, pauli)
    gradients = circuit.backprop(combined)
    return {i: -gradient for i, gradient in zip(parameter_gates, gradients, strict=True)}


def adjoint_gradient(program, observables, weights):
    """Return bound symbolic-parameter gradients of a weighted expectation loss.

    Observables with identical structural cones share one local parametric circuit
    and one Qulacs backpropagation. Contributions are summed across cones and across
    every occurrence of a reused symbolic parameter.
    """
    operations = program.resolved_operations()
    requests = tuple(program.expectation(observable) for observable in observables)
    weights = tuple(finite_angle(weight) for weight in weights)
    if not requests or len(requests) != len(weights):
        raise ValueError("A finite weight is required for each observable")
    parameters = {
        gate.angle.name for gate in program.operations if isinstance(gate.angle, Parameter)
    }
    if not parameters:
        raise ValueError("Program must contain at least one symbolic parameter")
    contributions = {name: [] for name in parameters}
    tasks = {}
    for request, weight in zip(requests, weights, strict=True):
        if not weight or not request.observable.factors:
            continue
        cone = _cone_indices(operations, request.observable)
        wires = _cone_wires(operations, request.observable, cone)
        key = (wires, tuple(sorted(cone)))
        tasks.setdefault(key, []).append((weight, request.observable))
    for (wires, cone), terms in tasks.items():
        gradients = _adjoint_task_gradient(operations, wires, set(cone), terms)
        for i, gradient in gradients.items():
            angle = program.operations[i].angle
            if isinstance(angle, Parameter):
                contributions[angle.name].append(gradient)
    contributions = {name: fsum(values) for name, values in contributions.items()}
    if any(not isfinite(value) for value in contributions.values()):
        raise ArithmeticError("Nonfinite adjoint gradient")
    return contributions


@dataclass(frozen=True)
class ApproximationPlan:
    exact_plan: object
    plan: object
    program: Program
    zeroed_indices: tuple[int, ...]
    error_bounds: tuple[float, ...]
    error_budgets: tuple[float, ...]
    candidates_checked: int
    search_truncated: bool
    planning_ns: int
    profile_ns: int


def plan_approximate(profile, *, error_budget, max_candidates=64, **planning_options):
    """Select exact/local-rewritten zero-angle subsets under per-output bounds.

    Search is bounded: test the full subset, then sensitivity-ranked subsets of
    increasing cardinality. Rank admissible plans by their existing execution cost
    (a heuristic without calibration), retaining exact execution on cost ties.
    Profiling/search overhead is exposed, not included in the execution score;
    callers must establish workload-specific lifetime break-even.
    """
    start = perf_counter_ns()
    if type(max_candidates) is not int or not 1 <= max_candidates <= 4096:
        raise ValueError("max_candidates must be between 1 and 4096")
    if isinstance(error_budget, (list, tuple)):
        budgets = tuple(_nonnegative(b, "error_budget") for b in error_budget)
    else:
        budgets = (_nonnegative(error_budget, "error_budget"),) * len(profile.observables)
    if len(budgets) != len(profile.observables):
        raise ValueError("One error budget is required per observable")
    options = dict(planning_options)
    if options.get("optimization", "local") != "local":
        raise ValueError("Approximation requires optimization='local'")
    options["optimization"] = "local"
    original = profile.program
    operations = original.resolved_operations()
    exact = plan_all([original.expectation(o) for o in profile.observables], **options)

    def score(plan):
        """Return calibrated prediction when available, otherwise heuristic score."""
        predicted = getattr(plan.selected, "predicted_execution_ns", None)
        return predicted if predicted is not None else plan.selected.score

    best, best_program, zeroed = exact, original, ()
    best_bounds = (0.0,) * len(budgets)
    ranked = sorted(
        range(len(profile.gate_indices)),
        key=lambda j: (
            abs(operations[profile.gate_indices[j]].angle)
            * max((abs(row[j]) for row in profile.jacobian), default=0.0)
        ),
    )
    full = tuple(ranked)

    def subsets():
        """Yield candidate zeroed subsets from largest to increasing cardinality."""
        if full:
            yield full
        for size in range(1, len(ranked)):
            yield from combinations(ranked, size)

    checked = 1
    for subset in islice(subsets(), max_candidates - 1):
        checked += 1
        chosen = {profile.gate_indices[j] for j in subset}
        deltas = tuple(-operations[i].angle if i in chosen else 0.0 for i in profile.gate_indices)
        bounds = profile.bounds(deltas)
        if any(b > budget for b, budget in zip(bounds, budgets, strict=True)):
            continue
        gates = tuple(replace(g, angle=0.0) if i in chosen else g for i, g in enumerate(operations))
        variant = Program(original.qubits, gates)
        try:
            plan = plan_all([variant.expectation(o) for o in profile.observables], **options)
        except NoFeasiblePlan:
            continue
        if score(plan) < score(best):
            best, best_program, zeroed, best_bounds = plan, variant, tuple(sorted(chosen)), bounds
    return ApproximationPlan(
        exact,
        best,
        best_program,
        zeroed,
        best_bounds,
        budgets,
        checked,
        len(ranked) >= max_candidates.bit_length() or (1 << len(ranked)) > checked,
        perf_counter_ns() - start,
        profile.elapsed_ns,
    )
