"""Aer and Lightning lower a symbolic task once and rebind its angles (issue #33)."""

import numpy as np
import pytest
from test_rebind import ansatz

from lazyqc import Parameter, Pauli, Program, parameter_shift_requests, plan_all
from lazyqc.backends.qulacs import QulacsBackend


def aer():
    pytest.importorskip("qiskit_aer")
    from lazyqc.backends.aer import AerBackend

    return AerBackend(device="CPU")


def lightning():
    pytest.importorskip("pennylane")
    from lazyqc.backends.lightning import LightningBackend

    return LightningBackend("lightning.qubit")


BACKENDS = [pytest.param(aer, id="aer"), pytest.param(lightning, id="lightning")]


def count_lowerings(backend):
    """Wrap `lower` on one instance so tests see how often tasks are lowered."""
    calls = []
    real = backend.lower
    backend.lower = lambda *a, **k: calls.append(k.get("parametric", False)) or real(*a, **k)
    return calls


@pytest.mark.parametrize("make", BACKENDS)
def test_steps_match_qulacs_and_lower_each_task_once(make):
    backend = make()
    calls = count_lowerings(backend)
    symbolic, values = ansatz(5, 2, 3).parametrize()
    obs = [Pauli(((q, "Z"),)) for q in range(5)] + [Pauli(((0, "X"), (3, "Y")))]
    plan = plan_all([symbolic.expectation(o) for o in obs], symbolic=True, strategy="separate")
    rng = np.random.default_rng(0)
    for _ in range(4):
        step = {k: float(rng.uniform(-3, 3)) for k in values}
        got = plan.collect(backend=backend, bindings=step).values
        want = plan.collect(backend=QulacsBackend(), bindings=step).values
        np.testing.assert_allclose(got, want, atol=1e-10)
    assert len(calls) == len(plan.selected.tasks)
    assert all(calls)  # every lowering was parametric


@pytest.mark.parametrize("make", BACKENDS)
def test_shifted_parameters_and_shared_names(make):
    theta = Parameter("t")
    program = (
        Program(3).ry(0, theta).cx(0, 1).rzz(1, 2, theta.shifted(0.4)).rx(2, theta.shifted(-1.1))
    )
    plan = plan_all(
        [program.expectation({1: "Z"}), program.expectation({2: "Y"})],
        symbolic=True,
        strategy="full",
    )
    backend = make()
    for value in (0.3, -1.2, 2.5, 0.3):
        got = plan.collect(backend=backend, bindings={"t": value}).values
        want = plan.collect(backend=QulacsBackend(), bindings={"t": value}).values
        np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("make", BACKENDS)
def test_parameter_shift_batch_gives_the_same_jacobian(make):
    program = ansatz(4, 2, 7)
    indices = [i for i, g in enumerate(program.operations) if g.angle is not None]
    symbolic, values = program.parametrize(indices)
    obs = [Pauli(((q, "Z"),)) for q in range(4)]
    plan = plan_all(parameter_shift_requests(symbolic, obs, indices), symbolic=True)
    backend = make()
    for _ in range(2):
        values = {k: v + 0.1 for k, v in values.items()}
        got = plan.collect(backend=backend, bindings=values).values
        want = plan.collect(backend=QulacsBackend(), bindings=values).values
        np.testing.assert_allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("make", BACKENDS)
def test_numeric_tasks_lower_as_before_and_unbound_ones_fail(make):
    backend = make()
    program = Program(3).ry(0, 0.4).cx(0, 1).rz(1, 0.9)
    plan = plan_all([program.expectation({1: "Z"})])
    np.testing.assert_allclose(
        plan.collect(backend=backend).values, plan.collect().values, atol=1e-10
    )
    symbolic = plan_all([Program(1).ry(0, Parameter("a")).expectation({0: "Z"})], symbolic=True)
    task = symbolic.selected.tasks[0]
    with pytest.raises(ValueError, match="unbound"):
        backend.lower(task)
    assert backend.lower(task, parametric=True).slots


@pytest.mark.parametrize("make", BACKENDS)
def test_tasks_without_symbolic_gates_carry_no_slots(make):
    backend = make()
    plan = plan_all([Program(2).h(0).cx(0, 1).expectation({1: "Z"})])
    lowered = backend.lower(plan.selected.tasks[0], parametric=True)
    assert not lowered.slots


def test_qsim_rebinds_rzz_and_reuses_the_lowered_circuit():
    pytest.importorskip("qsimcirq")
    from lazyqc.backends.qsim import QSimBackend

    theta = Parameter("theta")
    program = (
        Program(3).ry(0, theta).cx(0, 1).rzz(1, 2, theta.shifted(0.4)).rx(2, theta.shifted(-1.1))
    )
    plan = plan_all(
        [program.expectation({1: "Z"}), program.expectation({2: "Y"})],
        symbolic=True,
        strategy="full",
    )
    backend = QSimBackend(threads=1)
    calls = count_lowerings(backend)

    for value in (0.3, -1.2, 2.5):
        got = plan.collect(backend=backend, bindings={"theta": value}).values
        want = plan.collect(backend=QulacsBackend(), bindings={"theta": value}).values
        np.testing.assert_allclose(got, want, atol=1e-4)

    assert len(calls) == len(plan.selected.tasks) == 1
    assert calls == [True]


def test_qsim_rejects_projector_tasks():
    pytest.importorskip("qsimcirq")
    from lazyqc.backends.qsim import QSimBackend
    from lazyqc.instruments import Projector
    from lazyqc.plan import Task

    task = Task((0,), (Projector((0,), 1),), ((0, Pauli()),))
    with pytest.raises(ValueError, match="unitary tasks only"):
        QSimBackend().lower(task)


@pytest.mark.parametrize("threads", [0, -1, 1.5, True])
def test_qsim_rejects_invalid_thread_counts(threads):
    from lazyqc.backends.qsim import QSimBackend

    with pytest.raises(ValueError, match="threads"):
        QSimBackend(threads=threads)
