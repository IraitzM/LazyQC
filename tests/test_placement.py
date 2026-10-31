"""Width-based placement of wide tasks on an alternative statevector backend."""

import pytest

from lazyqc import Program, plan_all

pytest.importorskip("qiskit_aer")

from lazyqc.backends.aer import AerBackend  # noqa: E402


def mixed_width_requests():
    """A saturated chain plus a local pair: one wide task and one narrow task."""
    chain = Program(6)
    for q in range(6):
        chain = chain.ry(q, 0.2 + 0.1 * q).rx(q, -0.3 * q)
    for q in range(5):
        chain = chain.cx(q, q + 1)
    pair = Program(6).ry(0, 0.7).cx(0, 1).rz(1, 0.4).h(4)
    return [
        chain.expectation({5: "Z"}),
        chain.expectation({5: "X"}),
        chain.expectation({2: "Y", 5: "Z"}),
        pair.expectation({1: "X"}),
        pair.expectation({4: "X"}),
        pair.expectation({}),
    ]


def test_wide_tasks_run_on_the_wide_backend_with_equal_values():
    plan = plan_all(mixed_width_requests(), strategy="grouped")
    widths = sorted(t.width for t in plan.selected.tasks)
    assert widths[0] < 4 <= widths[-1]
    reference = plan.collect()
    wide = AerBackend(device="CPU")
    placed = plan.collect(wide_backend=wide, wide_from=4)
    assert placed.values == pytest.approx(reference.values, abs=1e-12)
    routed = sum(1 for t in plan.selected.tasks if t.width >= 4)
    assert wide.counters.evolutions == routed
    assert placed.stats.evolutions == reference.stats.evolutions == len(plan.selected.tasks)


def test_placement_options_are_validated():
    plan = plan_all(mixed_width_requests())
    with pytest.raises(ValueError, match="together"):
        plan.collect(wide_backend=AerBackend(device="CPU"))
    with pytest.raises(ValueError, match="positive"):
        plan.collect(wide_backend=AerBackend(device="CPU"), wide_from=0)


def test_any_backend_implementing_the_protocol_runs_the_whole_plan():
    plan = plan_all(mixed_width_requests(), strategy="grouped")
    reference = plan.collect()
    backend = AerBackend(device="CPU")
    result = plan.collect(backend=backend)
    assert result.values == pytest.approx(reference.values, abs=1e-12)
    assert backend.counters.evolutions == len(plan.selected.tasks)
    with pytest.raises(ValueError, match="backend must provide"):
        plan.collect(backend=object())


def test_lightning_backend_matches_qulacs():
    pytest.importorskip("pennylane")
    from lazyqc.backends.lightning import LightningBackend

    plan = plan_all(mixed_width_requests(), strategy="grouped")
    backend = LightningBackend("lightning.qubit")
    result = plan.collect(backend=backend)
    assert result.values == pytest.approx(plan.collect().values, abs=1e-12)
    assert backend.counters.evolutions == len(plan.selected.tasks)


def test_quimb_backend_matches_qulacs_alone_and_as_wide_backend():
    pytest.importorskip("quimb")
    from lazyqc.backends.tensor import QuimbBackend

    plan = plan_all(mixed_width_requests(), strategy="grouped")
    reference = plan.collect().values
    backend = QuimbBackend()
    assert plan.collect(backend=backend).values == pytest.approx(reference, abs=1e-10)
    assert backend.counters.evolutions == len(plan.selected.tasks)
    wide = QuimbBackend()
    assert plan.collect(wide_backend=wide, wide_from=4).values == pytest.approx(
        reference, abs=1e-10
    )
    assert wide.counters.evolutions == sum(1 for t in plan.selected.tasks if t.width >= 4)
