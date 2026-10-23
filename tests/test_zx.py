"""Independent unitary checks at conversion boundaries; no timing assertions."""

import random
from dataclasses import replace
from math import pi
from time import perf_counter_ns

import numpy as np
import pytest
from reference_numpy import expectation, statevector
from test_cost import profile

from lazyqc import Gate, Parameter, PlanningCache, Program, ZXOptions, plan_all
from lazyqc.fixtures import build_requests, load_fixture
from lazyqc.rewrites import from_pyzx, local_simplify, to_pyzx, zx_rewrite

pyzx = pytest.importorskip("pyzx")


def matrix(qubits, operations):
    gates = [(g.name, g.qubits, g.angle) for g in operations]
    return np.column_stack(
        [
            statevector(
                qubits, [("X", (q,), None) for q in range(qubits) if (column >> q) & 1] + gates
            )
            for column in range(2**qubits)
        ]
    )


def assert_unitary_equal(qubits, before, after):
    original, output = matrix(qubits, before), matrix(qubits, after)
    overlap = np.vdot(original, output) / (1 << qubits)
    assert abs(overlap) == pytest.approx(1, abs=1e-10)
    np.testing.assert_allclose(original * overlap / abs(overlap), output, atol=1e-10, rtol=0)


def interference(n=4, theta=0.37):
    p = Program(n).h(0).ry(n - 1, 0.41)
    # Nonadjacent phase gadgets commute and cancel; local adjacent cleanup cannot see it.
    return p.cx(0, 1).rz(1, theta).cx(0, 1).rz(0, 0.5).cx(0, 1).rz(1, -theta).cx(0, 1)


@pytest.mark.parametrize(
    "gate",
    [
        Gate("H", (2,)),
        Gate("X", (2,)),
        Gate("Y", (2,)),
        Gate("Z", (2,)),
        Gate("RX", (2,), 0.73),
        Gate("RY", (2,), -0.43),
        Gate("RZ", (2,), 0.29),
        Gate("CX", (2, 0)),
    ],
)
def test_conversion_all_supported_gates(gate):
    converted = to_pyzx(3, (gate,))
    restored = from_pyzx(converted, 3)
    assert_unitary_equal(3, (gate,), restored)


def test_swap_cz_and_output_order():
    circuit = pyzx.Circuit(3)
    circuit.add_gate("SWAP", 0, 2)
    circuit.add_gate("CZ", 2, 1)
    expected = (
        Gate("CX", (0, 2)),
        Gate("CX", (2, 0)),
        Gate("CX", (0, 2)),
        Gate("H", (1,)),
        Gate("CX", (2, 1)),
        Gate("H", (1,)),
    )
    assert_unitary_equal(3, expected, from_pyzx(circuit, 3))


@pytest.mark.parametrize("seed", range(6))
def test_random_unitaries_after_zx(seed):
    rng = random.Random(seed)
    gates = []
    for _ in range(24):
        name = rng.choice(("H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"))
        qs = tuple(rng.sample(range(3), 2)) if name == "CX" else (rng.randrange(3),)
        angle = rng.choice((rng.uniform(-2, 2), pi / 4, -pi / 2)) if name.startswith("R") else None
        gates.append(Gate(name, qs, angle))
    output = zx_rewrite(3, tuple(gates), perf_counter_ns() + 10**10, 1024)
    assert_unitary_equal(3, gates, output)


def test_nonlocal_zx_reduction_and_calibrated_selection():
    p = interference()
    assert local_simplify(p.operations) == p.operations
    requests = [p.expectation({q: axis}) for q in range(4) for axis in "XYZ"]
    model = profile(gates=1000, tasks=1, state_amplitudes=1)
    plan = plan_all(requests, optimization="zx", cost_model=model)
    assert plan.selected.optimization == "zx"
    assert plan.rewrite_records[0].status == "verified"
    assert plan.rewrite_records[0].zx_gates < plan.rewrite_records[0].local_gates
    assert plan.selected.score < min(c.score for c in plan.alternatives if c.optimization == "none")
    expected = statevector(4, [(g.name, g.qubits, g.angle) for g in p.operations])
    assert plan.collect().values == pytest.approx(
        [expectation(expected, r.observable.factors) for r in requests], abs=1e-10
    )
    assert plan.collect().selected_optimization == "zx"


def test_expanding_rewrite_keeps_original_candidate():
    p = Program(2).ry(0, 0.37).cx(0, 1)
    plan = plan_all([p.expectation({0: "Z"})], strategy="full", optimization="zx")
    assert any(c.optimization == "zx" for c in plan.alternatives)
    assert plan.selected.optimization == "none"


def test_lazy_optimization_and_cache_identity(monkeypatch):
    import qulacs

    import lazyqc.planner as planner
    import lazyqc.rewrites as rewrites

    def forbidden(*args, **kwargs):
        pytest.fail("Optimizer allocated a simulation object or was rerun on cache hit")

    monkeypatch.setattr(qulacs, "QuantumState", forbidden)
    monkeypatch.setattr(qulacs, "QuantumCircuit", forbidden)
    monkeypatch.setattr(pyzx.Circuit, "to_matrix", forbidden)
    monkeypatch.setattr(pyzx.Circuit, "to_tensor", forbidden)
    cache = PlanningCache()
    p = interference()
    requests = [p.expectation({0: "X"})]
    for mode in ("none", "local", "zx"):
        assert plan_all(requests, optimization=mode, cache=cache).cache_status == "miss"
    monkeypatch.setattr(rewrites, "zx_rewrite", forbidden)
    hit = plan_all(requests, optimization="zx", cache=cache)
    assert hit.cache_status == "hit"
    assert hit.optimization_ns == 0 < hit.optimization_build_ns
    hit.explain()
    # Change implementation identity/options: old analysis cannot hit. Skip ZX by width
    # so the patched forbidden optimizer is never entered during this explicit miss.
    assert (
        plan_all(
            requests, optimization="zx", cache=cache, zx_options=ZXOptions(max_qubits=1)
        ).cache_status
        == "miss"
    )
    monkeypatch.setattr(planner, "optimizer_identity", lambda mode: "different-implementation")
    assert (
        plan_all(
            requests, optimization="zx", cache=cache, zx_options=ZXOptions(max_qubits=1)
        ).cache_status
        == "miss"
    )


def test_options_limits_fallback_and_binding():
    p = interference()
    r = [p.expectation({0: "X"})]
    for options in (
        ZXOptions(max_qubits=1),
        ZXOptions(max_gates=1),
        ZXOptions(soft_timeout_ms=1e-9),
    ):
        plan = plan_all(r, optimization="zx", zx_options=options)
        assert plan.rewrite_records[0].status in ("skipped", "fallback")
        assert plan.collect().values == pytest.approx(plan_all(r).collect().values, abs=1e-10)
    cache = PlanningCache()
    for theta in (0.27, 0.89):
        p = Program(4).ry(3, Parameter("t")).cx(3, 1).bind(t=theta)
        r = [p.expectation({3: "Z"}), p.expectation({1: "X"}), p.expectation({})]
        plan = plan_all(r, cache=cache, optimization="zx", max_state_qubits=2)
        assert plan.cache_status == "miss"
        assert plan.collect().values == pytest.approx(plan_all(r).collect().values, abs=1e-10)
        assert plan_all(r, cache=cache, optimization="zx", max_state_qubits=2).cache_status == "hit"
    batches = [interference(theta=t).expectation({0: "X"}) for t in (0.3, 0.5, 0.7)]
    plan = plan_all(batches, optimization="zx", zx_options=ZXOptions(max_circuits=1))
    assert sum(r.status == "verified" for r in plan.rewrite_records) == 1
    assert sum(r.detail == "ZX circuit-attempt limit" for r in plan.rewrite_records) == 2


@pytest.mark.parametrize("failure", ["extract", "verify", "large_angle"])
def test_explicit_failure_falls_back(monkeypatch, failure):
    p = interference()
    if failure == "extract":

        def fail(*args, **kwargs):
            raise ValueError("test extraction failure")

        monkeypatch.setattr(pyzx, "extract_circuit", fail)
    elif failure == "verify":
        monkeypatch.setattr(pyzx.Circuit, "verify_equality", lambda *a, **kw: False)
    else:
        p = p.rz(0, 100 * pi)
    r = [p.expectation({0: "Y"})]
    plan = plan_all(r, optimization="zx")
    assert plan.rewrite_records[0].status == "fallback"
    assert plan.selected.optimization != "zx"
    assert plan.collect().values == pytest.approx(plan_all(r).collect().values, abs=1e-10)


def test_explicit_uncomputation_controls_and_empty():
    from pathlib import Path

    _, cases = load_fixture(Path(__file__).parent / "fixtures/uncomputation.json")
    values = []
    for case in cases:
        requests = build_requests(case)
        result = plan_all(requests, optimization="zx").collect()
        assert result.values == pytest.approx([r["expected"] for r in case["requests"]], abs=1e-10)
        values.append(result.values)
    assert values[0] != values[1]
    assert plan_all([], optimization="zx").collect().values == ()
    assert plan_all([Program(1).h(0).h(0).expectation({})], optimization="zx").collect().values == (
        1,
    )


def test_local_cleanup_and_invalid_options():
    p = Program(2).h(0).h(0).ry(1, 0.4).ry(1, -0.4).rz(0, 0)
    assert local_simplify(p.operations) == ()
    assert (
        plan_all([p.expectation({0: "Z"})], optimization="local").selected.optimization == "local"
    )
    for kwargs in (
        {"optimization": "bad"},
        {"zx_options": ZXOptions()},
        {"optimization": "zx", "zx_options": {}},
    ):
        with pytest.raises(ValueError):
            plan_all([], **kwargs)
    for kwargs in (
        {"max_qubits": 0},
        {"max_gates": True},
        {"max_circuits": 1.2},
        {"soft_timeout_ms": float("inf")},
    ):
        with pytest.raises(ValueError):
            replace(ZXOptions(), **kwargs)


def test_noop_local_rewrite_does_not_repeat_cone_analysis(monkeypatch):
    import lazyqc.planner as planner

    original = planner._analyze
    calls = []

    def count(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(planner, "_analyze", count)
    p = interference()
    plan_all([p.expectation({0: "X"})], optimization="local")
    assert len(calls) == 1
