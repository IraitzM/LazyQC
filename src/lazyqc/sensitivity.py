"""Opt-in, occurrence-wise sensitivity and bounded zero-angle candidate planning.

Profiling explicitly executes shifted circuits. Ordinary plan_all remains exact and
lazy. Bounds concern norm-one terminal Paulis and assume the caller's evaluation
error allowance; they are not interval-arithmetic certificates. A profile is valid
only at its stored program, bindings and observables, never across optimizer steps.
"""

from dataclasses import dataclass, replace
from itertools import combinations, islice
from math import fsum, isfinite, pi
from time import perf_counter_ns

from .ir import ROTATIONS, Parameter, Program, finite_angle
from .planner import NoFeasiblePlan, plan_all


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
        Each two-shift derivative has error <= evaluation_error, conditionally.
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
    program, observables, gate_indices, *, evaluation_error, session=None, **planning_options
):
    """Execute two shifts per relevant rotation occurrence, batched across outputs.

    evaluation_error is an explicit assumed absolute error per evaluated expectation.
    max_state_* and workers options are forwarded to the ordinary planner. Callers
    own any persistent ExecutionSession and must account for its startup/shutdown.
    """
    start = perf_counter_ns()
    evaluation_error = _nonnegative(evaluation_error, "evaluation_error")
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
    for j, i in enumerate(indices):
        for sign in (1, -1):
            gates = list(operations)
            angle = gates[i].angle
            if abs(angle) > 8 * pi:
                raise ValueError("Sensitivity profiling supports angles within 8*pi")
            gates[i] = replace(gates[i], angle=angle + sign * pi / 2)
            variant = Program(program.qubits, tuple(gates))
            for k, observable in enumerate(observables):
                if relevant[k][j]:
                    shifted.append(variant.expectation(observable))
                    positions.append((k, j, sign))
    jacobian = [[0.0] * len(indices) for _ in observables]
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
    )


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
