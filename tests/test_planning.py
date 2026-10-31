from pathlib import Path

import pytest

from lazyqc import NoFeasiblePlan, Program, collect_all, plan_all
from lazyqc.backends.qulacs import QulacsBackend
from lazyqc.fixtures import build_requests, load_fixture

ROOT = Path(__file__).resolve().parents[1]


def initial_requests():
    _, cases = load_fixture(ROOT / "tests/fixtures/initial.json")
    return build_requests(cases[0])


def test_planning_and_explain_do_not_touch_native_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Backend was touched before collection")

    monkeypatch.setattr(QulacsBackend, "lower", forbidden)
    monkeypatch.setattr(QulacsBackend, "evolve", forbidden)
    import qulacs

    monkeypatch.setattr(qulacs, "QuantumState", forbidden)
    monkeypatch.setattr(qulacs, "QuantumCircuit", forbidden)
    requests = initial_requests()
    backend = QulacsBackend()
    for strategy in ("full", "separate", "grouped", "auto"):
        plan_all(requests, strategy=strategy).explain()
    assert backend.counters.allocations == backend.counters.evolutions == 0


@pytest.mark.parametrize(
    "strategy,count,width",
    [("full", 1, 12), ("separate", 24, 2), ("grouped", 6, 2), ("auto", 6, 2)],
)
def test_structural_counts_and_fresh_collections(strategy, count, width):
    plan = plan_all(initial_requests(), strategy=strategy)
    backend = QulacsBackend()
    for _ in range(2):
        result = plan.collect(backend=backend)
        assert result.stats.evolutions == result.stats.allocations == count
        assert result.stats.max_state_qubits == width
        assert result.stats.estimated_state_bytes == 16 * 2**width
        assert result.stats.expectations == 24
    assert backend.counters.evolutions == 2 * count


def test_partial_batch_and_disjoint_requests():
    requests = initial_requests()
    assert len(plan_all(requests[:4], strategy="grouped").selected.tasks) == 1
    assert len(plan_all(requests[::4], strategy="grouped").selected.tasks) == 6


def test_limits_reject_full_but_allow_small_cones():
    requests = initial_requests()
    plan = plan_all(requests, max_state_qubits=2, max_state_bytes=64)
    assert plan.selected.strategy == "grouped"
    assert plan.alternatives[0].rejection is not None
    assert plan.collect().stats.max_state_qubits == 2
    with pytest.raises(NoFeasiblePlan, match="full"):
        plan_all(requests, strategy="full", max_state_qubits=2)
    with pytest.raises(NoFeasiblePlan):
        plan_all(requests, max_state_bytes=63)


def test_identity_and_empty_batch_do_not_allocate(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A constant-only plan should not allocate")

    monkeypatch.setattr(QulacsBackend, "evolve", forbidden)
    result = collect_all([Program(2).h(0).expectation({})])
    assert result.values == (1.0,)
    assert result.stats.evolutions == 0
    assert collect_all([]).values == ()


def test_full_candidate_shares_equal_bound_programs():
    p = Program(2).ry(0, 0.4).cx(0, 1)
    requests = [p.expectation({0: "Z"}), p.expectation({1: "Z"})]
    plan = plan_all(requests, strategy="full")
    assert len(plan.selected.tasks) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"strategy": "unknown"},
        {"max_state_qubits": 0},
        {"max_state_qubits": True},
        {"max_state_bytes": 0},
        {"max_state_bytes": 1.5},
        {"max_state_qubits": 31},
    ],
)
def test_bad_options(options):
    with pytest.raises(ValueError):
        plan_all(initial_requests(), **options)


def test_forced_full_baseline_does_not_run_cone_optimizer(monkeypatch):
    import lazyqc.planner as planner

    def forbidden(*args, **kwargs):
        pytest.fail("The full-state baseline must not pay for cone optimization")

    monkeypatch.setattr(planner, "cone_indices", forbidden)
    plan = plan_all(initial_requests(), strategy="full")
    assert len(plan.alternatives) == 1
    assert plan.collect().stats.evolutions == 1
