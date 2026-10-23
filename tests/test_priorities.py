"""Fallback coverage, planning reuse, persistent workers and resource controls."""

import json
import multiprocessing
import resource
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

import pytest

from lazyqc import (
    BatchCostModel,
    CutOptions,
    ExecutionSession,
    PlanningCache,
    Program,
    plan_all,
    plan_isolated,
)
from lazyqc.batch_cost import batch_environment
from lazyqc.cache import object_bytes
from lazyqc.fixtures import build_requests

pytest.importorskip("qiskit_addon_cutting")
ROOT = Path(__file__).resolve().parents[1]


def local_batch():
    fixture = json.loads((ROOT / "benchmarks/p4_p5.json").read_text())
    case = next(c for c in fixture["cases"] if c["name"] == "local_pauli_batch16")
    return build_requests(case), CutOptions(max_fragment_qubits=3, **fixture["cut_limits"])


def test_frozen_local_batch_now_feasible_and_reuses_compilation():
    pytest.importorskip("pyzx")
    requests, options = local_batch()
    plan = plan_all(requests, cutting=options, optimization="zx")
    assert plan.collect().values == pytest.approx(
        plan_all(requests, strategy="full").collect().values, abs=1e-10
    )
    assert dict(plan.analysis_stats)["recipe_hits"] > 0
    assert any(c.strategy == "fixed-fallback" for c in plan.alternatives)


def test_forced_full_independent_search_considers_rewritten_variants():
    pytest.importorskip("pyzx")
    p = Program(2).h(0).cx(0, 1).rz(0, 0.2).cx(0, 1)
    requests = [p.expectation({0: "X"})]
    plan = plan_all(
        requests,
        strategy="full",
        optimization="zx",
        cutting=CutOptions(max_fragment_qubits=1, max_cuts=0, search="independent"),
    )
    assert plan.selected.optimization == "zx"
    assert plan.collect().values == pytest.approx(plan_all(requests).collect().values, abs=1e-10)


def test_timeout_after_fixed_baseline_retains_complete_result(monkeypatch):
    pytest.importorskip("pyzx")
    import lazyqc.batch_planner as planner

    requests, options = local_batch()
    expired = False
    original = planner.price

    def price(candidate, *args, **kwargs):
        nonlocal expired
        result = original(candidate, *args, **kwargs)
        if candidate.strategy == "fixed-fallback":
            expired = True
        return result

    monkeypatch.setattr(planner, "price", price)
    monkeypatch.setattr(
        planner, "perf_counter_ns", lambda: perf_counter_ns() + (10**12 if expired else 0)
    )
    plan = plan_all(requests, cutting=options, optimization="zx")
    assert plan.search_exhausted
    assert plan.collect().values == pytest.approx(
        plan_all(requests, strategy="full").collect().values, abs=1e-10
    )


def test_reuse_budget_stops_optional_exploration_without_changing_semantics():
    p = Program(4).ry(0, 0.3).cx(0, 1).cx(1, 2).cx(2, 3)
    requests = [p.expectation({q: "Z"}) for q in range(4)]
    # Tiny predicted work makes further exploration economically unjustified.
    model = BatchCostModel((0,) * 8, 0, 0, batch_environment(), 4, 0)
    options = CutOptions(max_cuts=0, expected_collections=1)
    limited = plan_all(requests, cutting=options, cost_model=model)
    broad = plan_all(
        requests, cutting=replace(options, expected_collections=None), cost_model=model
    )
    assert limited.search_states < broad.search_states
    assert limited.collect().values == pytest.approx(broad.collect().values, abs=1e-10)
    cache = PlanningCache()
    assert plan_all(requests, cutting=options, cache=cache).cache_status == "miss"
    assert (
        plan_all(
            requests, cutting=replace(options, expected_collections=10), cache=cache
        ).cache_status
        == "miss"
    )


def test_cache_byte_cap_lru_rejections_and_clear():
    size = object_bytes(("first", bytes(200)))
    cache = PlanningCache(8, max_bytes=size + 50)
    cache._put("first", bytes(200))
    cache._put("second", bytes(200))
    assert cache._get("first") is None
    assert cache._get("second") == bytes(200)
    info = cache.info()
    assert info.evictions == 1 and 0 < info.accounted_bytes <= info.max_bytes
    cache._put("huge", bytes(size * 2))
    assert cache.info().oversized_rejections == 1
    cache.clear()
    assert cache.info().accounted_bytes == 0
    for value in (0, -1, True):
        with pytest.raises(ValueError):
            PlanningCache(max_bytes=value)


def bell_plan(**kwargs):
    p = Program(2).h(0).cx(0, 1)
    return plan_all(
        [p.expectation({0: a, 1: a}) for a in "XYZ"],
        workers=2,
        cutting=CutOptions(max_fragment_qubits=1),
        total_state_bytes=64,
        **kwargs,
    )


def test_persistent_workers_recompute_results_and_close():
    plan = bell_plan()
    with ExecutionSession(2, policy="parallel") as session:
        first = plan.collect(session=session)
        second = plan.collect(session=session)
        assert first.values == second.values == pytest.approx((1, -1, 1), abs=1e-10)
        assert not first.stats.pool_reused and first.stats.pool_startup_ns > 0
        assert second.stats.pool_reused and second.stats.pool_startup_ns == 0
        assert set(first.stats.worker_pids) & set(second.stats.worker_pids)
        assert second.stats.allocations == len(plan.selected.tasks)
        assert second.stats.peak_active_state_bytes <= 64
        changed = Program(2).ry(0, 0.8).cx(0, 1)
        request = [changed.expectation({1: "Z"})]
        result = plan_all(request, workers=2, cutting=CutOptions(max_fragment_qubits=1)).collect(
            session=session
        )
        assert result.values == pytest.approx(plan_all(request).collect().values, abs=1e-10)
    with pytest.raises(RuntimeError, match="closed"):
        plan.collect(session=session)


def test_auto_session_avoids_unmeasured_startup_and_expensive_dispatch(monkeypatch):
    for key in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(key, "1")
    model = BatchCostModel((1,) * 8, 1, 1, batch_environment(), 2, 0)
    plan = bell_plan(cost_model=model)
    with ExecutionSession(2) as session:
        assert plan.collect(session=session).stats.execution_mode == "serial"
        assert session._pool is None
        profile = session.warmup()
        assert profile.startup_ns > 0 and profile.roundtrip_ns > 0
        result = plan.collect(session=session)
        assert result.stats.execution_mode == "serial"
        assert "dispatch" in result.stats.dispatch_reason


def test_session_rejects_concurrent_use_and_worker_mismatch():
    with ExecutionSession(2) as session:
        with session._claim():
            with pytest.raises(RuntimeError, match="active"):
                bell_plan().collect(session=session)
    with ExecutionSession(3) as session:
        with pytest.raises(ValueError, match="worker counts"):
            bell_plan().collect(session=session)


def test_session_worker_address_space_limit_is_local():
    before = resource.getrlimit(resource.RLIMIT_AS)
    limit = 2 * 2**30
    with ExecutionSession(2, policy="parallel", max_address_space_bytes=limit) as session:
        session.warmup()
        remote = session._pool.submit(resource.getrlimit, resource.RLIMIT_AS).result()
        assert remote[0] <= limit and remote[1] <= limit
        assert bell_plan().collect(session=session).values == pytest.approx((1, -1, 1), abs=1e-10)
    assert resource.getrlimit(resource.RLIMIT_AS) == before


def test_isolated_planning_preserves_result_and_parent_limits():
    before = resource.getrlimit(resource.RLIMIT_AS)
    p = Program(2).h(0).cx(0, 1)
    requests = [p.expectation({0: "Y", 1: "Y"})]
    plan = plan_isolated(
        requests,
        timeout_s=10,
        max_address_space_bytes=4 * 2**30,
        cutting=CutOptions(max_fragment_qubits=1),
    )
    assert plan.collect().values == pytest.approx((-1,), abs=1e-10)
    for group in plan.selected.groups:
        assert hash(group) == hash(replace(group))
        for formula in group.formulas:
            for term in formula.terms:
                for ref in term.factors:
                    assert hash(ref.recipe) == hash(replace(ref.recipe))
    assert resource.getrlimit(resource.RLIMIT_AS) == before


def test_isolated_deadline_terminates_child_and_propagates_validation():
    children = {p.pid for p in multiprocessing.active_children()}
    requests = [Program(2).h(0).cx(0, 1).expectation({0: "X"})]
    with pytest.raises(TimeoutError):
        plan_isolated(requests, timeout_s=0.000001, optimization="zx")
    assert {p.pid for p in multiprocessing.active_children()} == children
    with pytest.raises(ValueError, match="strategy"):
        plan_isolated(requests, strategy="invalid")
    with pytest.raises(ValueError, match="PlanningCache"):
        plan_isolated(requests, cache=PlanningCache())
