from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
from reference_numpy import expectation, statevector

from lazyqc import Gate, Parameter, Pauli, Program, plan_all
from lazyqc.backends.qulacs import QulacsBackend
from lazyqc.fixtures import build_requests, load_fixture

ROOT = Path(__file__).resolve().parents[1]
STRATEGIES = ("full", "separate", "grouped", "auto")


def reference(program, observables):
    gates = [(g.name, g.qubits, g.angle) for g in program.resolved_operations()]
    state = statevector(program.n_qubits, gates)
    return [expectation(state, p.factors) for p in observables]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("fixture", ["initial.json", "uncomputation.json"])
def test_golden_fixtures(strategy, fixture):
    data, cases = load_fixture(ROOT / "tests/fixtures" / fixture)
    for case in cases:
        requests = build_requests(case)
        result = plan_all(requests, strategy=strategy).collect()
        expected = [r["expected"] for r in case["requests"]]
        np.testing.assert_allclose(result.values, expected, rtol=0, atol=data["absolute_tolerance"])
        if "expected_data_11_probability" in case:
            program = requests[0].program
            observables = [{0: "Z"}, {1: "Z"}, {0: "Z", 1: "Z"}]
            zs = (
                plan_all([program.expectation(o) for o in observables], strategy=strategy)
                .collect()
                .values
            )
            probability = (1 - zs[0] - zs[1] + zs[2]) / 4
            assert probability == pytest.approx(case["expected_data_11_probability"], abs=1e-10)


@pytest.mark.parametrize("seed", range(8))
def test_random_circuits_against_independent_reference(seed):
    rng = np.random.default_rng(seed)
    program = Program(4)
    names = ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"]
    for _ in range(24):
        name = str(rng.choice(names))
        if name == "CX":
            wires = tuple(int(q) for q in rng.choice(4, 2, replace=False))
            program = program.gate(name, *wires)
        else:
            angle = float(rng.uniform(-3, 3)) if name.startswith("R") else None
            program = program.gate(name, int(rng.integers(4)), angle=angle)
    observables = [
        Pauli(tuple((q, str(rng.choice(["I", "X", "Y", "Z"]))) for q in range(4)))
        for _ in range(12)
    ]
    expected = reference(program, observables)
    for strategy in STRATEGIES:
        results = plan_all(
            [program.expectation(o) for o in observables], strategy=strategy
        ).collect()
        np.testing.assert_allclose(results.values, expected, rtol=0, atol=1e-10)


def test_bridge_changes_dependencies():
    program = Program(4).ry(0, 0.41).cx(0, 1).ry(2, 0.77).cx(2, 3)
    base = plan_all([program.expectation({2: "Z"})], strategy="grouped")
    assert base.selected.max_width == 2
    bridged = program.cx(1, 2)
    observables = [Pauli(((2, "Z"),)), Pauli(((0, "X"), (2, "X"))), Pauli(((1, "Y"), (3, "Y")))]
    new = plan_all([bridged.expectation(o) for o in observables], strategy="grouped")
    assert new.selected.max_width == 4
    expected = reference(bridged, observables)
    np.testing.assert_allclose(new.collect().values, expected, rtol=0, atol=1e-10)
    assert abs(new.collect().values[0] - base.collect().values[0]) > 1e-3


def test_binding_is_immutable_and_coherence_is_sign_sensitive():
    program = Program(("a", "b")).ry("a", Parameter("theta")).cx("a", "b")
    positive = program.bind(theta=0.43)
    negative = positive.bind(theta=-0.43)
    for p, theta in [(positive, 0.43), (negative, -0.43)]:
        values = (
            plan_all([p.expectation({"a": "X", "b": "X"}), p.expectation({"a": "Z"})])
            .collect()
            .values
        )
        np.testing.assert_allclose(values, [np.sin(theta), np.cos(theta)], atol=1e-12)
    assert positive.bindings == (("theta", 0.43),)
    assert not program.bindings
    with pytest.raises(FrozenInstanceError):
        program.operations = ()
    with pytest.raises(ValueError, match="Unbound"):
        plan_all([program.expectation({"a": "Z"})])
    with pytest.raises(ValueError, match="Binding"):
        program.bind(typo=1)


def test_parameter_bindings_cannot_share_incorrect_states():
    program = Program(2).ry(0, Parameter("t")).cx(0, 1)
    requests = [program.bind(t=t).expectation({0: "X", 1: "X"}) for t in [0.2, -0.2, 0.2]]
    plan = plan_all(requests, strategy="grouped")
    assert len(plan.selected.tasks) == 2
    np.testing.assert_allclose(plan.collect().values, np.sin([0.2, -0.2, 0.2]), atol=1e-12)


def test_remapping_named_qubits_and_order():
    p = Program(tuple(f"logical{i}" for i in range(10))).ry("logical8", 0.63).cx(8, 9)
    requests = [p.expectation({"logical8": "X", "logical9": "X"}), p.expectation({9: "Z"})]
    requests = [requests[1], requests[0], requests[1]]
    plan = plan_all(requests, strategy="grouped")
    assert plan.explain()["groups"][0]["logical_to_local"] == {"8": 0, "9": 1}
    np.testing.assert_allclose(
        plan.collect().values, [np.cos(0.63), np.sin(0.63), np.cos(0.63)], atol=1e-12
    )


@pytest.mark.parametrize("name", ["H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"])
def test_inverse_and_normalization(name):
    gate = Gate(name, (0, 1) if name == "CX" else (0,), 0.7 if name.startswith("R") else None)
    p = Program(2).h(0).ry(1, 0.4)
    p = Program(p.qubits, (*p.operations, gate, gate.inverse()))
    task = plan_all([p.expectation({0: "X"})], strategy="full").selected.tasks[0]
    backend = QulacsBackend()
    state = backend.evolve(backend.lower(task))
    assert state.get_squared_norm() == pytest.approx(1.0, abs=1e-12)
    expected = statevector(2, [("H", (0,), None), ("RY", (1,), 0.4)])
    np.testing.assert_allclose(state.get_vector(), expected, atol=1e-12)


def test_version_metadata_preserves_dependencies():
    p = Program(("x", "temp")).h("x").cx("x", "temp").z("temp").cx("x", "temp")
    steps = p.versioned_operations
    assert steps[1].inputs == ((0, 1), (1, 0))
    assert steps[3].inputs == ((0, 2), (1, 2))
    assert steps[0].gate.effects.measurement_free
    assert not steps[0].gate.effects.basis_preserving


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Program(0),
        lambda: Program(True),
        lambda: Program(("x", "x")),
        lambda: Program(2).cx(0, 0),
        lambda: Program(2).x(2),
        lambda: Program(2).x("absent"),
        lambda: Program(2).ry(0, float("nan")),
        lambda: Program(2).rx(0, float("inf")),
        lambda: Program(2).gate("measure", 0),
        lambda: Program(2).gate("H", 0, angle=0.1),
        lambda: Program(2).expectation({2: "Z"}),
        lambda: Pauli(((0, "X"), (0, "Y"))),
        lambda: Pauli(((0, "A"),)),
    ],
)
def test_invalid_programs_fail_before_execution(factory):
    with pytest.raises(ValueError):
        factory()
