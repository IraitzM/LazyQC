"""Commutation-aware observable cones (issue #1)."""

import math
import random

import pytest
from reference_numpy import expectation, statevector

from lazyqc import Gate, Parameter, PlanningCache, Program, plan_all, profile_sensitivity
from lazyqc.passes import commuting_cone


def cone(ops, observable):
    """Return the retained gate names for the cone of a Pauli given as {wire: letter}."""
    shape = tuple((name, qubits) for name, qubits in ops)
    wires, retained = commuting_cone(shape, tuple(sorted(observable.items())))
    return wires, [ops[i][0] for i in retained]


def test_the_issue_example_keeps_only_the_hadamard():
    ops = [("H", (0,)), ("RY", (1,)), ("CX", (1, 2)), ("CX", (0, 1))]
    # Z0 meets a CX whose control carries only Z, then nothing else touches wire 0.
    assert cone(ops, {0: "Z"}) == ((0,), ["H"])


@pytest.mark.parametrize(
    "ops,observable,kept",
    [
        ([("CX", (0, 1))], {0: "Z"}, []),  # control carries Z
        ([("CX", (0, 1))], {1: "X"}, []),  # target carries X
        ([("CX", (0, 1))], {1: "Z"}, ["CX"]),  # Z on the target spreads to the control
        ([("CX", (0, 1))], {0: "X"}, ["CX"]),  # X on the control spreads to the target
        ([("CX", (0, 1))], {0: "Y"}, ["CX"]),
        ([("CZ", (0, 1))], {0: "Z", 1: "Z"}, []),
        ([("CZ", (0, 1))], {1: "X"}, ["CZ"]),
        ([("RZZ", (0, 1))], {0: "Z"}, []),
        ([("RZZ", (0, 1))], {0: "X"}, ["RZZ"]),
        ([("RZ", (0,))], {0: "Z"}, []),
        ([("RZ", (0,))], {0: "X"}, ["RZ"]),
        ([("RX", (0,))], {0: "X"}, []),
        ([("RY", (0,))], {0: "Y"}, []),
        ([("RY", (0,))], {0: "Z"}, ["RY"]),
        ([("Z", (0,))], {0: "Z"}, []),
        ([("Z", (0,))], {0: "X"}, ["Z"]),  # a sign flip still changes the value
        ([("H", (0,))], {0: "Z"}, ["H"]),
        ([("RY", (1,))], {0: "Z"}, []),  # off the support
    ],
)
def test_gate_rules(ops, observable, kept):
    assert cone(ops, observable)[1] == kept


def test_letters_travel_through_retained_gates():
    # H turns the X on wire 0 into Z, so the CX before it has a control carrying only Z.
    assert cone([("CX", (0, 1)), ("H", (0,))], {0: "X"})[1] == ["H"]
    # A rotation about X widens Z into {Y, Z}; the CX control no longer carries only Z.
    assert cone([("CX", (0, 1)), ("RX", (0,))], {0: "Z"})[1] == ["CX", "RX"]
    # Two CX in a row: the second one spreads X to wire 1, so the first must stay.
    assert cone([("CX", (0, 1)), ("CX", (0, 1))], {0: "X"})[1] == ["CX", "CX"]


def random_circuit(rng, n, size):
    gates = []
    for _ in range(size):
        kind = rng.random()
        if kind < 0.3:
            gates.append(
                Gate(rng.choice(("RX", "RY", "RZ")), (rng.randrange(n),), rng.uniform(-3, 3))
            )
        elif kind < 0.4:
            gates.append(Gate(rng.choice(("H", "X", "Y", "Z")), (rng.randrange(n),)))
        elif kind < 0.7:
            gates.append(Gate("CX", tuple(rng.sample(range(n), 2))))
        elif kind < 0.85:
            gates.append(Gate("CZ", tuple(rng.sample(range(n), 2))))
        else:
            gates.append(Gate("RZZ", tuple(rng.sample(range(n), 2)), rng.uniform(-3, 3)))
    return gates


def dense_value(n, gates, factors):
    return expectation(statevector(n, [(g.name, g.qubits, g.angle) for g in gates]), factors)


@pytest.mark.parametrize("seed", range(150))
def test_the_cone_keeps_the_expectation_and_is_never_wider(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    gates = random_circuit(rng, n, rng.randint(1, 18))
    wires = rng.sample(range(n), rng.randint(1, min(3, n)))
    factors = tuple(sorted((q, rng.choice("XYZ")) for q in wires))
    shape = tuple((g.name, g.qubits) for g in gates)
    live, retained = commuting_cone(shape, factors)
    # Same value as the full circuit, computed on the cone wires alone.
    index = {q: i for i, q in enumerate(live)}
    reduced = [
        Gate(gates[i].name, tuple(index[q] for q in gates[i].qubits), gates[i].angle)
        for i in retained
    ]
    local = tuple((index[q], p) for q, p in factors)
    got = dense_value(len(live), reduced, local)
    assert got == pytest.approx(dense_value(n, gates, factors), abs=1e-10)
    # Never wider than the support cone.
    support = set(q for q, _ in factors)
    kept = set(support)
    kept_gates = set()
    for i in range(len(gates) - 1, -1, -1):
        if kept.intersection(gates[i].qubits):
            kept.update(gates[i].qubits)
            kept_gates.add(i)
    assert set(live) <= kept and set(retained) <= kept_gates


def test_planning_with_commuting_cones_gives_the_same_values():
    rng = random.Random(3)
    n = 6
    program = Program(n, tuple(random_circuit(rng, n, 24)))
    requests = [program.expectation({q: p}) for q in range(n) for p in "XYZ"]
    for strategy in ("full", "separate", "grouped", "auto"):
        want = plan_all(requests, strategy=strategy).collect().values
        got = plan_all(requests, strategy=strategy, cones="commuting").collect().values
        assert got == pytest.approx(want, abs=1e-10)


def test_commuting_cones_can_be_narrower():
    program = Program(4).ry(0, 0.4).cx(0, 1).cx(1, 2).cx(2, 3)
    [request] = [program.expectation({0: "Z"})]
    narrow = plan_all([request], strategy="separate", cones="commuting").selected.tasks[0]
    wide = plan_all([request], strategy="separate").selected.tasks[0]
    assert narrow.wires == (0,) and [g.name for g in narrow.operations] == ["RY"]
    assert wide.width == 2 and len(wide.operations) == 2  # the CX only touches the control


def test_outputs_on_one_support_still_share_a_task():
    program = Program(3).ry(0, 0.4).cx(0, 1).ry(1, 0.2).cx(1, 2)
    requests = [program.expectation({1: p}) for p in "XYZ"]
    plan = plan_all(requests, strategy="grouped", cones="commuting")
    assert len(plan.selected.tasks) == 1
    assert plan.selected.tasks[0].outputs[0][0] == 0
    support = plan_all(requests, strategy="grouped")
    assert plan.selected.tasks[0].width <= support.selected.tasks[0].width


def test_options_are_validated_and_kept_apart_in_the_cache():
    program = Program(2).ry(0, 0.3).cx(0, 1)
    request = program.expectation({0: "Z"})
    with pytest.raises(ValueError, match="cones"):
        plan_all([request], cones="narrow")
    with pytest.raises(ValueError, match="commuting"):
        plan_all([request], cones="commuting", workers=2)
    cache = PlanningCache()
    first = plan_all([request], strategy="separate", cache=cache)
    second = plan_all([request], strategy="separate", cache=cache, cones="commuting")
    assert first.cache_status == "miss" and second.cache_status == "miss"
    assert first.selected.tasks[0].width == 2 and second.selected.tasks[0].width == 1


def test_symbolic_plans_take_the_option():
    theta = Parameter("t")
    program = Program(3).ry(0, theta).cx(0, 1).rz(1, theta.shifted(0.5)).cx(1, 2)
    requests = [program.expectation({q: p}) for q in range(3) for p in "ZX"]
    plan = plan_all(requests, symbolic=True, cones="commuting")
    for value in (0.3, -1.1):
        want = plan_all(requests, symbolic=True).collect(bindings={"t": value}).values
        assert plan.collect(bindings={"t": value}).values == pytest.approx(want, abs=1e-10)


def test_sensitivity_finds_exact_structural_zeros():
    # RZ before a Z measurement commutes with it, so its derivative is exactly zero.
    program = Program(2).ry(0, 0.4).cx(0, 1).rz(0, 0.7).rx(1, 0.3)
    profile = profile_sensitivity(program, [{0: "Z"}], [0, 2, 3], evaluation_error=1e-12)
    assert profile.relevant == ((True, False, False),)
    assert profile.jacobian[0][1] == 0.0 and profile.jacobian[0][2] == 0.0
    step = 1e-6
    ops = [(g.name, g.qubits, g.angle) for g in program.operations]
    plus, minus = list(ops), list(ops)
    plus[0] = ("RY", (0,), 0.4 + step)
    minus[0] = ("RY", (0,), 0.4 - step)
    numeric = (
        expectation(statevector(2, plus), ((0, "Z"),))
        - expectation(statevector(2, minus), ((0, "Z"),))
    ) / (2 * step)
    assert profile.jacobian[0][0] == pytest.approx(numeric, abs=1e-8)
    assert math.isfinite(numeric)
