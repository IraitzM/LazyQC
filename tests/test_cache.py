"""Classical reuse must preserve binding, request order, policy and execution semantics."""

from dataclasses import replace

import pytest
from test_cost import profile

from lazyqc import NoFeasiblePlan, Parameter, PlanningCache, Program, plan_all
from lazyqc.backends.qulacs import QulacsBackend


def batch(theta=0.37):
    p = Program(6).ry(0, Parameter("t")).cx(0, 1).ry(4, 0.81).cx(4, 5).bind(t=theta)
    return tuple(p.expectation({q: axis}) for q in (0, 1, 4, 5) for axis in ("X", "Z"))


def test_cache_hit_skips_analysis_and_features(monkeypatch):
    import lazyqc.plan as plan_module
    import lazyqc.planner as planner

    cache = PlanningCache()
    model = profile(tasks=100, state_amplitudes=1)
    first = plan_all(batch(), cache=cache, cost_model=model)
    assert first.cache_status == "miss"

    def forbidden(*args, **kwargs):
        pytest.fail("Cache hit recomputed structural analysis/features or invoked backend")

    monkeypatch.setattr(planner, "_analyze", forbidden)
    monkeypatch.setattr(plan_module, "_compute_features", forbidden)
    monkeypatch.setattr(QulacsBackend, "lower", forbidden)
    monkeypatch.setattr(QulacsBackend, "evolve", forbidden)
    import qulacs

    monkeypatch.setattr(qulacs, "QuantumCircuit", forbidden)
    monkeypatch.setattr(qulacs, "QuantumState", forbidden)
    second = plan_all(batch(), cache=cache, cost_model=model)
    assert second.cache_status == second.explain()["cache_status"] == "hit"
    assert first.selected == second.selected
    assert cache.info().hits == cache.info().misses == 1


def test_repeated_collection_allocates_fresh_states():
    cache = PlanningCache()
    backend = QulacsBackend()
    plans = [plan_all(batch(), cache=cache) for _ in range(2)]
    values = []
    for plan in plans:
        result = plan.collect(backend=backend)
        assert result.stats.allocations == result.stats.evolutions == len(plan.selected.tasks)
        values.append(result.values)
    assert values[0] == values[1]
    assert backend.counters.allocations == sum(len(p.selected.tasks) for p in plans)


def test_changed_binding_gate_order_and_duplicates():
    cache = PlanningCache()
    initial = batch()
    changes = [batch(0.92), initial[::-1], initial + initial[:2], initial[:-1]]
    changed = initial[0].program.cx(1, 4)
    changes.append(tuple(changed.expectation(r.observable) for r in initial))
    expected = plan_all(initial, strategy="full").collect().values
    plan_all(initial, cache=cache)
    for requests in changes:
        plan = plan_all(requests, cache=cache)
        assert plan.cache_status == "miss"
        assert plan.collect().values == pytest.approx(
            plan_all(requests, strategy="full").collect().values, abs=1e-10
        )
        assert plan_all(requests, cache=cache).cache_status == "hit"
    assert plan_all(initial, cache=cache).collect().values == pytest.approx(expected, abs=1e-10)
    assert plan_all(initial[::-1], cache=cache).collect().values == pytest.approx(
        expected[::-1], abs=1e-10
    )


def test_equal_programs_with_different_object_sharing_hit():
    cache = PlanningCache()
    original = batch()
    plan_all(original, cache=cache)
    reconstructed = tuple(replace(r, program=replace(r.program)) for r in original)
    assert plan_all(reconstructed, cache=cache).cache_status == "hit"


def test_cache_rechecks_limits_models_and_domain():
    cache = PlanningCache()
    initial = plan_all(batch(), cache=cache, cost_model=profile(tasks=10000))
    assert initial.selected.strategy == "full"
    # Reuse structural analysis; selection must use the current profile and limits.
    grouped = plan_all(batch(), cache=cache, cost_model=profile(state_amplitudes=1000))
    assert grouped.cache_status == "hit"
    assert grouped.selected.strategy == "grouped"
    limited = plan_all(batch(), cache=cache, max_state_qubits=2, max_state_bytes=64)
    assert limited.cache_status == "hit"
    assert limited.selected.strategy == "grouped"
    assert limited.alternatives[0].rejection is not None
    with pytest.raises(NoFeasiblePlan):
        plan_all(batch(), cache=cache, max_state_bytes=63)
    assert (
        plan_all(batch(), cache=cache, cost_model=profile(tasks=10000)).selected == initial.selected
    )
    narrow = replace(profile(tasks=10), max_width=2)
    assert "fallback" in plan_all(batch(), cache=cache, cost_model=narrow).cost_model_status
    assert (
        plan_all(batch(), cache=cache, cost_model=narrow, max_state_qubits=2).cost_model_status
        == "calibrated"
    )


def test_cache_cannot_hide_stale_profile_or_invalid_input(monkeypatch):
    cache, model = PlanningCache(), profile(tasks=1)
    requests = batch()
    plan = plan_all(requests, cache=cache, cost_model=model)
    monkeypatch.setenv("QULACS_NUM_THREADS", "100")
    with pytest.raises(ValueError, match="environment mismatch"):
        plan_all(requests, cache=cache, cost_model=model)
    with pytest.raises(ValueError, match="environment mismatch"):
        plan.collect()
    with pytest.raises(ValueError, match="Requests"):
        plan_all([object()], cache=cache)
    unbound = Program(1).ry(0, Parameter("unbound"))
    with pytest.raises(ValueError, match="Unbound"):
        plan_all([unbound.expectation({})], cache=cache)


def test_strategy_is_part_of_cache_key():
    cache = PlanningCache()
    for strategy in ("auto", "full", "separate", "grouped"):
        plan = plan_all(batch(), strategy=strategy, cache=cache)
        assert plan.cache_status == "miss"
        assert plan_all(batch(), strategy=strategy, cache=cache).cache_status == "hit"
        if strategy != "auto":
            assert len(plan.alternatives) == 1
            assert plan.selected.strategy == strategy


def test_lru_eviction_and_clear():
    cache = PlanningCache(max_entries=2)
    plan = plan_all(batch(0.1), cache=cache)
    plan_all(batch(0.2), cache=cache)
    assert plan_all(batch(0.1), cache=cache).cache_status == "hit"
    plan_all(batch(0.3), cache=cache)
    assert cache.info().entries == 2
    assert cache.info().evictions == 1
    assert plan_all(batch(0.2), cache=cache).cache_status == "miss"
    cache.clear()
    assert cache.info().entries == cache.info().hits == cache.info().misses == 0
    assert cache.info().evictions == 0
    assert plan.collect().stats.allocations > 0


def test_empty_identity_and_equal_hash_collisions(monkeypatch):
    cache = PlanningCache()
    for requests in ((), (Program(2).h(0).expectation({}),)):
        plan_all(requests, cache=cache)
        plan = plan_all(requests, cache=cache)
        assert plan.cache_status == "hit"
        assert plan.collect().stats.allocations == 0
    # Hash equality is never semantic equality.
    monkeypatch.setattr(Program, "__hash__", lambda self: 0)
    plan_all(batch(0.1), cache=cache)
    plan = plan_all(batch(0.9), cache=cache)
    assert plan.cache_status == "miss"
    assert plan.collect().values == plan_all(batch(0.9)).collect().values


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_capacity(value):
    with pytest.raises(ValueError, match="positive integer"):
        PlanningCache(value)


def test_invalid_cache():
    with pytest.raises(ValueError, match="PlanningCache"):
        plan_all(batch(), cache={})
