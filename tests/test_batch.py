"""Batch search, bounded workers and instrument-aware calibration contracts."""

from dataclasses import replace

import pytest

from lazyqc import (
    BatchCostModel,
    CutOptions,
    NoFeasiblePlan,
    PlanningCache,
    Program,
    calibrate_batch,
    plan_all,
)
from lazyqc.batch import schedule_estimate
from lazyqc.batch_cost import batch_environment
from lazyqc.ir import Gate, Parameter, Pauli
from lazyqc.plan import Task

pytest.importorskip("qiskit_addon_cutting")


def workload():
    p = Program(4).ry(0, 0.3).h(1).cx(0, 1).ry(2, 0.7).cx(1, 2).cx(2, 3)
    return tuple(
        p.expectation(o)
        for o in (
            {0: "Z"},
            {2: "X"},
            {0: "Z", 3: "Z"},
            {1: "Y", 2: "Y"},
        )
    )


@pytest.mark.parametrize("optimization", ["none", "local", "zx"])
def test_joint_and_independent_match_reference_and_retain_baseline(optimization):
    if optimization == "zx":
        pytest.importorskip("pyzx")
    requests = workload()
    opts = CutOptions(max_fragment_qubits=2, max_cuts=2)
    reference = plan_all(requests, strategy="full").collect().values
    independent = plan_all(
        requests, cutting=replace(opts, search="independent"), optimization=optimization
    )
    joint = plan_all(requests, cutting=opts, optimization=optimization)
    assert joint.selected.score <= independent.selected.score
    for plan in (independent, joint):
        assert plan.collect().values == pytest.approx(reference, abs=1e-10)
        assert plan.selected.max_width <= 2
        assert plan.explain()["dag"]["experiment_nodes"]


def test_cross_request_fragment_reuse_and_changed_parameters():
    p = Program(2).ry(0, Parameter("t")).cx(0, 1)
    requests = [p.bind(t=t).expectation({0: "Z", 1: "Z"}) for t in (0.2, 0.2, 0.9)]
    opts = CutOptions(max_fragment_qubits=1, search="independent")
    shared = plan_all(requests, cutting=opts)
    separate = plan_all(requests, cutting=replace(opts, reuse=False))
    assert len(shared.selected.tasks) < len(separate.selected.tasks)
    assert shared.collect().values == pytest.approx(separate.collect().values, abs=1e-10)


def test_cut_seeds_handle_repeated_boundary_and_report_budget():
    p = Program(4).h(0).cx(0, 1).cx(1, 2).ry(2, 0.2).cx(1, 2).cx(2, 3)
    requests = [p.expectation({0: "X", 3: "Z"})]
    opts = CutOptions(max_fragment_qubits=2, max_cut_candidates=2)
    plan = plan_all(requests, strategy="full", cutting=opts)
    assert plan.search_exhausted
    assert plan.collect().values == pytest.approx(
        plan_all(requests, strategy="full").collect().values, abs=1e-10
    )


def test_search_budget_preserves_independent_fallback():
    opts = CutOptions(max_fragment_qubits=2, max_search_states=1)
    plan = plan_all(workload(), cutting=opts)
    assert plan.search_exhausted and plan.search_states == 1
    assert plan.collect().values == pytest.approx(
        plan_all(workload(), strategy="full").collect().values, abs=1e-10
    )


@pytest.mark.parametrize("memory,expected_concurrency", [(32, 1), (64, 2)])
def test_spawn_workers_obey_aggregate_payload(memory, expected_concurrency):
    p = Program(2).h(0).cx(0, 1)
    requests = [p.expectation({0: "Y", 1: "Y"}), p.expectation({0: "Z", 1: "Z"})]
    opts = CutOptions(max_fragment_qubits=1)
    plan = plan_all(requests, cutting=opts, workers=2, total_state_bytes=memory)
    result = plan.collect()
    assert result.values == pytest.approx((-1, 1), abs=1e-10)
    assert result.stats.peak_active_state_bytes <= memory
    assert result.stats.peak_in_flight == expected_concurrency
    assert 1 <= result.stats.worker_processes <= 2
    assert result.stats.worker_peak_rss_bytes > 0


def test_worker_failure_is_propagated():
    plan = plan_all(workload(), workers=2)
    bad_task = Task((0,), (Gate("RY", (0,), Parameter("unbound")),), ((0, Pauli()),))
    broken = replace(plan, selected=replace(plan.selected, tasks=(bad_task,)))
    with pytest.raises(ValueError, match="unbound"):
        broken.collect()


def test_worker_configuration_is_cached_separately():
    cache = PlanningCache()
    first = plan_all(workload(), workers=2, cache=cache)
    assert first.cache_status == "miss"
    assert plan_all(workload(), workers=2, cache=cache).cache_status == "hit"
    assert plan_all(workload(), workers=3, cache=cache).cache_status == "miss"
    with pytest.raises(NoFeasiblePlan):
        plan_all(workload(), workers=2, total_state_bytes=31, cache=cache)
    for workers in (0, True, 33):
        with pytest.raises(ValueError):
            plan_all(workload(), workers=workers)


def test_scheduler_prediction_respects_memory():
    tasks = [Task((0,), (), ((0, Pauli()),))] * 3
    assert schedule_estimate(tasks, 2, 64, [3, 2, 1]) == (3, 64)
    assert schedule_estimate(tasks, 2, 32, [3, 2, 1]) == (6, 32)


def test_batch_calibration_roundtrip_and_environment(tmp_path, monkeypatch):
    model, raw = calibrate_batch(widths=(1, 2), repeats=1)
    assert raw["runs"] and model.calibration_ns > 0
    path = tmp_path / "batch.json"
    model.save(path)
    assert BatchCostModel.load(path) == model
    opts = CutOptions(max_fragment_qubits=2)
    plan = plan_all(workload(), cutting=opts, cost_model=model)
    assert plan.selected.cost_units == "estimated ns"
    assert plan.collect().values == pytest.approx(
        plan_all(workload(), strategy="full").collect().values, abs=1e-10
    )
    monkeypatch.setenv("QULACS_NUM_THREADS", "7")
    with pytest.raises(ValueError, match="environment mismatch"):
        plan.collect()


def test_parallel_profile_requires_single_native_threads(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    model = BatchCostModel((1,) * 8, 1, 1, batch_environment(), 4, 0)
    with pytest.raises(ValueError, match="native thread"):
        plan_all(workload(), workers=2, cost_model=model)
