"""Exact channel reconstruction against independent dense calculations."""

import random
from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from reference_numpy import expectation, statevector

from lazyqc import CutOptions, NoFeasiblePlan, Parameter, PlanningCache, Program, plan_all
from lazyqc.batch import BatchPlan, assemble, price
from lazyqc.batch_planner import plan_batch
from lazyqc.cutting import compile_group, cut_choices, cx_branches
from lazyqc.instruments import Projector

pytest.importorskip("qiskit_addon_cutting")


def dense(requests):
    states = {}
    for r in requests:
        p = r.program
        if p not in states:
            states[p] = statevector(
                p.n_qubits, [(g.name, g.qubits, g.angle) for g in p.resolved_operations()]
            )
    return tuple(expectation(states[r.program], r.observable.factors) for r in requests)


def bell():
    p = Program(2).h(0).cx(0, 1)
    return tuple(
        p.expectation(o)
        for o in ({0: "Z"}, {1: "Z"}, {0: "Z", 1: "Z"}, {0: "X", 1: "X"}, {0: "Y", 1: "Y"}, {})
    )


@pytest.mark.parametrize("search", ["independent", "fixed", "joint"])
def test_bell_correlations(search):
    plan = plan_all(bell(), cutting=CutOptions(max_fragment_qubits=1, max_cuts=1, search=search))
    result = plan.collect()
    assert result.values == pytest.approx((0, 0, 1, 1, -1, 1), abs=1e-10)
    assert result.stats.max_state_qubits == 1
    assert len(cx_branches()) == 10
    assert result.stats.allocations == len(plan.selected.tasks)
    assert any(isinstance(g, Projector) for t in plan.selected.tasks for g in t.operations)
    assert any(not o.factors for t in plan.selected.tasks for _, o in t.outputs)


@pytest.mark.parametrize("seed", range(6))
def test_arbitrary_product_inputs_and_noncommuting_suffix(seed):
    rng = random.Random(seed)
    p = Program(2)
    for q in range(2):
        p = p.rx(q, rng.uniform(-2, 2)).ry(q, rng.uniform(-2, 2)).rz(q, rng.uniform(-2, 2))
    p = p.cx(seed % 2, 1 - seed % 2)
    for q in range(2):
        p = p.ry(q, rng.uniform(-2, 2)).rx(q, rng.uniform(-2, 2))
    requests = tuple(p.expectation({0: a, 1: b}) for a in "IXYZ" for b in "IXYZ")
    plan = plan_all(
        requests, strategy="full", cutting=CutOptions(max_fragment_qubits=1, max_cuts=1)
    )
    assert plan.collect().values == pytest.approx(dense(requests), abs=1e-10)


def test_multiple_cuts_probability_weights_and_branch_cap():
    p = Program(3).ry(0, 0.4).h(1).cx(0, 1).ry(1, 0.7).cx(1, 2).rx(2, 0.3)
    r = tuple(
        p.expectation(o) for o in ({0: "X", 2: "Y"}, {0: "Z", 1: "Z", 2: "Z"}, {1: "X"}, {2: "Z"})
    )
    opts = CutOptions(max_fragment_qubits=1, max_cuts=2, max_cut_candidates=16)
    plan = plan_all(r, strategy="full", cutting=opts)
    assert plan.collect().values == pytest.approx(dense(r), abs=1e-10)
    assert any(g.branches == 100 and g.qpd_terms == 36 for g in plan.selected.groups)
    with pytest.raises(NoFeasiblePlan):
        plan_all(r, strategy="full", cutting=replace(opts, max_branches=99))
    with pytest.raises(NoFeasiblePlan):
        plan_all(r, strategy="full", cutting=replace(opts, max_cuts=1))


def test_all_pauli_transfer_entries_of_cx_channel():
    # Product eigenstates of X/Y/Z span the two-qubit operator space. Check all
    # output Paulis, so a decomposition preserving only Bell marginals cannot pass.
    options = CutOptions(max_fragment_qubits=1, max_cuts=1)
    for a, b in product("XYZ", repeat=2):
        for bit0 in (0, 1):
            for bit1 in (0, 1):
                p = Program(2)
                for q, axis, bit in ((0, a, bit0), (1, b, bit1)):
                    if bit:
                        p = p.x(q)
                    if axis in "XY":
                        p = p.h(q)
                    if axis == "Y":
                        p = p.rz(q, np.pi / 2)
                p = p.cx(0, 1)
                r = tuple(p.expectation({0: x, 1: y}) for x in "IXYZ" for y in "IXYZ")
                assert plan_all(
                    r, strategy="full", cutting=options
                ).collect().values == pytest.approx(dense(r), abs=1e-10)


def test_zero_probability_branches_are_not_normalized(monkeypatch):
    from lazyqc.backends.qulacs import QulacsBackend
    from lazyqc.ir import Pauli
    from lazyqc.plan import Task

    t = Task((0,), (Projector((0,), 1),), ((0, Pauli()), (1, Pauli(((0, "Z"),)))))
    backend = QulacsBackend()
    lowered = backend.lower(t)
    state = backend.evolve(lowered)
    assert state.get_squared_norm() == 0
    assert backend.expectations(state, lowered.observables) == (0, 0)


def test_cut_planning_and_explain_are_classical(monkeypatch):
    import qulacs

    from lazyqc.backends.qulacs import QulacsBackend

    def forbidden(*a, **kw):
        pytest.fail("Simulation during planning")

    monkeypatch.setattr(qulacs, "QuantumState", forbidden)
    monkeypatch.setattr(qulacs, "QuantumCircuit", forbidden)
    monkeypatch.setattr(QulacsBackend, "evolve", forbidden)
    plan_all(bell(), cutting=CutOptions(max_fragment_qubits=1)).explain()


def test_cache_profiles_limits_bindings_and_order():
    cache = PlanningCache()
    o = CutOptions(max_fragment_qubits=1, max_cuts=1)
    r = bell()
    assert plan_all(r, cutting=o, cache=cache).cache_status == "miss"
    assert plan_all(r, cutting=o, cache=cache).cache_status == "hit"
    rev = plan_all(r[::-1], cutting=o, cache=cache)
    assert rev.cache_status == "miss"
    assert rev.collect().values == pytest.approx(dense(r)[::-1], abs=1e-10)
    assert plan_all(r, cutting=replace(o, reuse=False), cache=cache).cache_status == "miss"
    with pytest.raises(NoFeasiblePlan):
        plan_all(r, cutting=o, cache=cache, max_state_bytes=31)
    for theta in (0.13, 0.71):
        p = Program(2).ry(0, Parameter("t")).cx(0, 1).bind(t=theta)
        r = [p.expectation({0: "Z"})]
        plan = plan_all(r, cutting=o, cache=cache)
        assert plan.cache_status == "miss"
        assert plan.collect().values == pytest.approx(dense(r), abs=1e-10)


def test_repeated_factors_keep_multiplicity_and_no_value_cache():
    p = Program(2).ry(0, 0.7).ry(1, 0.7)
    r = [p.expectation({0: "Z", 1: "Z"})]
    plan = plan_all(r, cutting=CutOptions(max_fragment_qubits=1, max_cuts=0))
    assert len(plan.selected.tasks) == 1  # isomorphic local zero-input preparations
    assert len(plan.selected.reconstruction[0].terms[0].factors) == 2
    for _ in range(2):
        result = plan.collect()
        assert result.values == pytest.approx((np.cos(0.7) ** 2,), abs=1e-10)
        assert result.stats.evolutions == 1


def test_empty_identity_and_explicit_partition():
    opts = CutOptions(max_fragment_qubits=1, partition=((0,), (1,)))
    assert plan_all([], cutting=opts).collect().values == ()
    assert plan_all([Program(2).expectation({})], cutting=opts).collect().values == (1,)
    p = plan_all(bell(), cutting=opts)
    assert p.collect().values == pytest.approx(dense(bell()), abs=1e-10)
    with pytest.raises(ValueError, match="partition"):
        plan_all(bell(), cutting=replace(opts, partition=((0,),)))
    for kwargs in ({"max_cuts": 7}, {"partition": ((0,), (0,))}, {"max_branches": 0}, {"reuse": 1}):
        with pytest.raises(ValueError):
            CutOptions(**kwargs)


def test_compiled_group_never_has_cross_fragment_gate():
    task = plan_all(bell(), strategy="full").selected.tasks[0]
    options = CutOptions(max_fragment_qubits=1)
    cuts, parts = next(cut_choices(task, options))
    g = compile_group(task, cuts, parts, options)
    c = price(assemble((g,), options, "test"), 1, 64)
    assert all(len(op.qubits) == 1 for t in c.tasks for op in t.operations)
    plan = BatchPlan(c, (c,), len(bell()), ((5, 1.0),), 1, 32, 1, 64, options, 0, 0, False)
    assert plan.collect().values == pytest.approx(dense(bell()), abs=1e-10)


def test_disjoint_fastpath_skips_beam_but_keeps_winner():
    p = Program(4).ry(0, 0.4).cx(0, 1).ry(2, 0.8).cx(2, 3)
    requests = [
        p.expectation({0: "Z"}),
        p.expectation({1: "Z"}),
        p.expectation({2: "Z"}),
        p.expectation({3: "Z"}),
    ]
    opts = CutOptions(max_fragment_qubits=2, max_cuts=1, search="joint")
    joint = plan_all(requests, cutting=opts)
    assert dict(joint.analysis_stats).get("disjoint_fastpath") == 1
    assert len(joint.selected.tasks) == 2
    indep = plan_all(requests, cutting=replace(opts, search="independent"))
    assert len(indep.selected.tasks) == len(joint.selected.tasks)
    assert joint.collect().values == pytest.approx(dense(requests), abs=1e-10)

    q = Program(3).ry(0, 0.3).cx(0, 1).cx(1, 2)
    coupled = [q.expectation({0: "Z"}), q.expectation({2: "Z"})]
    overlapping = plan_all(
        coupled, cutting=CutOptions(max_fragment_qubits=2, max_cuts=1, search="joint")
    )
    assert "disjoint_fastpath" not in dict(overlapping.analysis_stats)
    assert overlapping.collect().values == pytest.approx(dense(coupled), abs=1e-10)


def test_structural_hints_replay_decisions():
    p = Program(4).ry(0, 0.4).cx(0, 1).ry(2, 0.8).cx(2, 3)
    requests = [p.expectation({q: "Z"}) for q in range(4)]
    kwargs = dict(
        strategy="auto",
        optimization="none",
        workers=1,
        max_state_qubits=2,
        max_state_bytes=64,
        total_state_bytes=128,
    )
    opts = CutOptions(max_fragment_qubits=2, max_cuts=1, search="joint")
    full = plan_batch(requests, cutting=opts, **kwargs)
    template = full.explain()["search"]["structural_template"]
    assert template["selected"]
    hinted = plan_batch(requests, cutting=opts, _structural_hints=template, **kwargs)
    assert dict(hinted.analysis_stats).get("structural_hit") == 1
    assert len(hinted.selected.tasks) == len(full.selected.tasks)
    assert hinted.selected.score == full.selected.score

    q = Program(4).ry(0, 0.1).cx(0, 1).cx(1, 2).cx(2, 3)
    other = [q.expectation({0: "Z"})]
    fallback = plan_batch(other, cutting=opts, _structural_hints=template, **kwargs)
    assert dict(fallback.analysis_stats).get("structural_miss") == 1
    plain = plan_batch(other, cutting=opts, **kwargs)
    assert len(fallback.selected.tasks) == len(plain.selected.tasks)
