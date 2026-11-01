"""Plan once, rebind at every step (issue #8)."""

import math
import random

import numpy as np
import pytest
from reference_numpy import expectation, statevector

from lazyqc import (
    Parameter,
    Pauli,
    PauliSum,
    Program,
    adjoint_gradient,
    parameter_shift_requests,
    plan_all,
)
from lazyqc.backends.qulacs import QulacsBackend
from lazyqc.cache import PlanningCache


def ansatz(n, layers, seed):
    """Random layered circuit with every rotation kind, including RZZ."""
    rng = random.Random(seed)
    program = Program(n)
    for _ in range(layers):
        for q in range(n):
            program = getattr(program, rng.choice(["rx", "ry", "rz"]))(q, rng.uniform(-3, 3))
        for q in range(n - 1):
            program = (
                program.cx(q, q + 1)
                if rng.random() < 0.5
                else program.rzz(q, q + 1, rng.uniform(-3, 3))
            )
    return program


def dense(program, observable):
    operations = [(g.name, g.qubits, g.angle) for g in program.resolved_operations()]
    return expectation(statevector(program.n_qubits, operations), observable.factors)


class NoRebind:
    """Backend without `rebind`: symbolic plans fall back to binding each task."""

    def __init__(self):
        self.inner = QulacsBackend()
        self.counters = self.inner.counters

    def lower(self, task):
        return self.inner.lower(task)

    def evolve(self, lowered):
        return self.inner.evolve(lowered)

    def expectations(self, state, observables):
        return self.inner.expectations(state, observables)


def test_parameter_offsets_resolve_and_accumulate():
    theta = Parameter("t")
    assert theta.shifted(0.5).shifted(0.25) == Parameter("t", 0.75)
    program = Program(1).ry(0, theta.shifted(math.pi / 2)).bind(t=0.1)
    assert program.resolved_operations()[0].angle == pytest.approx(0.1 + math.pi / 2)
    with pytest.raises(ValueError):
        Parameter("t", float("nan"))


def test_parametrize_returns_values_and_keeps_structure():
    program = ansatz(3, 2, 1)
    indices = [i for i, g in enumerate(program.operations) if g.name == "RY"]
    symbolic, values = program.parametrize(indices, prefix="w")
    assert list(values) == [f"w{k}" for k in range(len(indices))]
    assert [symbolic.operations[i].angle.name for i in indices] == list(values)
    assert symbolic.bind(**values).resolved_operations() == program.resolved_operations()
    with pytest.raises(ValueError):
        program.parametrize([0, 0])
    with pytest.raises(ValueError):
        program.parametrize(
            [program.operations.index(next(g for g in program.operations if g.name == "CX"))]
        )


@pytest.mark.parametrize("strategy", ["auto", "full", "separate", "grouped"])
def test_symbolic_plan_matches_dense_at_every_step(strategy):
    n = 4
    program = ansatz(n, 3, 7)
    symbolic, values = program.parametrize()
    observables = [Pauli(((q, "Z"),)) for q in range(n)] + [Pauli(((0, "X"), (2, "Y")))]
    plan = plan_all(
        [symbolic.expectation(o) for o in observables], symbolic=True, strategy=strategy
    )
    rng = np.random.default_rng(3)
    for step in range(4):
        current = {k: float(v + 0.3 * step * rng.normal()) for k, v in values.items()}
        got = plan.collect(bindings=current).values
        expected = [dense(symbolic.bind(**current), o) for o in observables]
        assert got == pytest.approx(expected, abs=1e-12)


def test_rebinding_equals_planning_from_scratch():
    program = ansatz(5, 3, 11)
    symbolic, values = program.parametrize()
    obs = [Pauli(((q, "Z"),)) for q in range(5)]
    plan = plan_all([symbolic.expectation(o) for o in obs], symbolic=True)
    for shift in (0.0, 0.4, -1.1):
        current = {k: v + shift for k, v in values.items()}
        fresh = plan_all([symbolic.bind(**current).expectation(o) for o in obs]).collect()
        assert plan.collect(bindings=current).values == pytest.approx(fresh.values, abs=1e-12)


def test_unbound_programs_still_fail_without_the_flag():
    symbolic, _ = Program(1).ry(0, 0.2).parametrize()
    with pytest.raises(ValueError, match="Unbound"):
        plan_all([symbolic.expectation({0: "Z"})])


def test_collect_reports_missing_bindings_and_ignores_extras():
    symbolic, values = Program(2).ry(0, 0.2).cx(0, 1).rx(1, 0.5).parametrize()
    plan = plan_all([symbolic.expectation({1: "Z"})], symbolic=True)
    assert plan.parameters == ("p0", "p1")
    with pytest.raises(ValueError, match="Unbound parameters: p1"):
        plan.collect(bindings={"p0": 0.1})
    with pytest.raises(ValueError, match="Unbound parameters"):
        plan.collect()
    with pytest.raises(ValueError):
        plan.collect(bindings={"p0": 0.1, "p1": float("inf")})
    plan.collect(bindings=values | {"unused": 1.0})


def test_parameters_outside_every_cone_are_not_required():
    program = Program(2).ry(0, Parameter("a")).ry(1, Parameter("b"))
    plan = plan_all([program.expectation({0: "Z"})], symbolic=True)
    assert plan.parameters == ("a",)
    assert plan.collect(bindings={"a": 0.3}).values == pytest.approx([math.cos(0.3)])


def test_numeric_plans_ignore_bindings():
    plan = plan_all([Program(1).ry(0, 0.3).expectation({0: "Z"})])
    assert plan.parameters == ()
    assert plan.collect(bindings={"x": 1.0}).values == pytest.approx([math.cos(0.3)])


def test_symbolic_planning_rejects_options_that_need_numbers():
    request = Program(1).ry(0, Parameter("a")).expectation({0: "Z"})
    for options in ({"optimization": "local"}, {"workers": 2}, {"analysis": "nope"}):
        with pytest.raises(ValueError):
            plan_all([request], symbolic=True, **options)


def test_lowering_happens_once_per_task_and_backends_without_rebind_still_work():
    class Counting(QulacsBackend):
        lowerings = 0

        def lower(self, task, **kwargs):
            Counting.lowerings += 1
            return super().lower(task, **kwargs)

    symbolic, values = ansatz(4, 2, 5).parametrize()
    obs = [Pauli(((q, "Z"),)) for q in range(4)]
    plan = plan_all([symbolic.expectation(o) for o in obs], symbolic=True, strategy="separate")
    first = plan.collect(backend=Counting(), bindings=values).values
    assert Counting.lowerings == len(plan.selected.tasks)
    second = plan.collect(backend=Counting(), bindings=values).values
    assert Counting.lowerings == len(plan.selected.tasks)
    assert first == second
    plain = plan.collect(backend=NoRebind(), bindings=values).values
    assert plain == pytest.approx(first, abs=1e-12)


def test_cached_lowering_follows_new_angles_not_old_ones():
    symbolic, values = Program(1).ry(0, 0.0).parametrize()
    plan = plan_all([symbolic.expectation({0: "Z"})], symbolic=True)
    for angle in (0.2, 1.3, -0.7, 0.2):
        assert plan.collect(bindings={"p0": angle}).values == pytest.approx([math.cos(angle)])


def test_shift_requests_give_the_gradient():
    n = 4
    program = ansatz(n, 2, 13)
    indices = [i for i, g in enumerate(program.operations) if g.name in ("RX", "RY", "RZ", "RZZ")]
    symbolic, values = program.parametrize(indices)
    obs = [Pauli(((q, "Z"),)) for q in range(n)]
    requests = parameter_shift_requests(symbolic, obs, indices)
    assert len(requests) == (1 + 2 * len(indices)) * n
    values_ = plan_all(requests, symbolic=True).collect(bindings=values).values
    table = np.asarray(values_).reshape(1 + 2 * len(indices), n)
    jacobian = 0.5 * (table[1::2] - table[2::2]).T
    weights = [0.7, -0.2, 0.4, 1.0]
    bound = symbolic.bind(**values)
    expected = adjoint_gradient(bound, obs, weights)
    got = jacobian.T @ np.asarray(weights)
    for k, name in enumerate(values):
        assert got[k] == pytest.approx(expected[name], abs=1e-10)


def test_shift_requests_validate_inputs():
    program = Program(1).ry(0, 0.1)
    with pytest.raises(ValueError, match="symbolic"):
        parameter_shift_requests(program, [Pauli(((0, "Z"),))], [0])
    with pytest.raises(ValueError):
        parameter_shift_requests(program, [Pauli(((0, "Z"),))], [5])


def test_sums_take_bindings_through_the_wrapper():
    symbolic, values = Program(2).ry(0, 0.4).cx(0, 1).parametrize()
    total = symbolic.expectation(PauliSum(((0.5, Pauli(((0, "Z"),))), (2.0, Pauli(((1, "Z"),))))))
    plan = plan_all([total], symbolic=True)
    for angle in (0.4, 1.0):
        got = plan.collect(bindings={"p0": angle}).values
        assert got == pytest.approx([2.5 * math.cos(angle)], abs=1e-12)


def test_cache_keeps_symbolic_and_numeric_plans_apart():
    cache = PlanningCache()
    symbolic, _ = Program(1).ry(0, 0.2).parametrize()
    request = symbolic.expectation({0: "Z"})
    first = plan_all([request], symbolic=True, cache=cache)
    again = plan_all([request], symbolic=True, cache=cache)
    assert (first.cache_status, again.cache_status) == ("miss", "hit")
    with pytest.raises(ValueError, match="Unbound"):
        plan_all([request], cache=cache)
