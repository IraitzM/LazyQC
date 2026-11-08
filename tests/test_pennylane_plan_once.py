"""The PennyLane device plans a repeated tape structure once (issue #32)."""

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pnp = qml.numpy

from lazyqc.frontends import pennylane as frontend  # noqa: E402


def qnode(dev, diff="best", n=4):
    def f(w):
        for q in range(n):
            qml.RX(w[q], wires=q)
            qml.RY(w[n + q], wires=q)
        for q in range(n - 1):
            qml.CNOT(wires=[q, q + 1])
        qml.IsingZZ(w[2 * n], wires=[0, n - 1])
        return qml.math.stack([qml.expval(qml.Z(0)), qml.expval(qml.X(1) @ qml.Z(n - 1))])

    return qml.QNode(f, dev, diff_method=diff)


def weights(seed, n=4):
    return pnp.array(np.random.default_rng(seed).uniform(-3, 3, 2 * n + 1), requires_grad=True)


def test_loop_matches_default_qubit_and_reuses_the_plan():
    dev = qml.device("lazyqc.qubit", wires=4)
    ref = qml.device("default.qubit", wires=4)
    a, b = qnode(dev), qnode(ref, "backprop")
    flags = []
    for step in range(5):
        w = weights(step)
        assert np.allclose(a(w), b(w), atol=1e-10)
        flags.append(dev.last_stats["plan_reused"])
        assert np.allclose(qml.jacobian(a)(w), qml.jacobian(b)(w), atol=1e-10)
    # The Jacobian call repeats the forward pass, so the second sight comes within step 0.
    assert flags == [False, True, True, True, True]


def test_values_equal_the_fresh_plan_path_exactly():
    kept, fresh = qml.device("lazyqc.qubit", wires=4), qml.device("lazyqc.qubit", wires=4)
    a, b = qnode(kept), qnode(fresh)
    for step in range(4):
        w = weights(10 + step)
        fresh._plans.clear()
        fresh._once.clear()
        assert np.array_equal(a(w), b(w))
        assert np.array_equal(qml.jacobian(a)(w), qml.jacobian(b)(w))


def test_derivatives_reuse_the_plan_and_keep_structural_zeros():
    dev = qml.device("lazyqc.qubit", wires=3)
    ref = qml.device("default.qubit", wires=3)

    def f(w):
        qml.Rot(w[0], w[1], w[2], wires=0)
        qml.Rot(w[3], w[4], w[5], wires=2)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))

    a, b = qml.QNode(f, dev, diff_method="best"), qml.QNode(f, ref, diff_method="backprop")
    for step in range(4):
        w = pnp.array(np.random.default_rng(step).uniform(-3, 3, 6), requires_grad=True)
        assert np.allclose(qml.grad(a)(w), qml.grad(b)(w), atol=1e-10)
        assert dev.last_stats["skipped_structural_zeros"] == 4
    assert dev.last_stats["plan_reused"] is True


def test_a_different_structure_gets_its_own_plan(monkeypatch):
    calls = []
    real = frontend.plan_all
    monkeypatch.setattr(
        frontend, "plan_all", lambda *a, **k: calls.append(bool(k.get("symbolic"))) or real(*a, **k)
    )
    dev = qml.device("lazyqc.qubit", wires=4)
    for n in (3, 3, 4, 4, 3, 4):
        w = weights(n, n)
        qnode(dev, "parameter-shift", n)(w)
    assert calls.count(True) == 2  # one symbolic plan per structure, built on the second sight


def test_batched_tapes_share_one_plan():
    dev = qml.device("lazyqc.qubit", wires=2)

    @qml.qnode(dev)
    def f(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))

    for step in range(4):
        x = pnp.array([0.1 * step, 0.5, 1.2])  # broadcast over three tapes
        assert np.allclose(f(x), np.cos(x), atol=1e-12)
    assert dev.last_stats["plan_reused"] is True


@pytest.mark.parametrize("option", [{"optimization": "local"}, {"workers": 2}])
def test_numeric_only_options_keep_the_fresh_plan_path(option):
    dev = qml.device("lazyqc.qubit", wires=4, **option)
    ref = qml.device("default.qubit", wires=4)
    a, b = qnode(dev, "parameter-shift"), qnode(ref, "backprop")
    for step in range(3):
        w = weights(step)
        assert np.allclose(a(w), b(w), atol=1e-10)
        assert dev.last_stats["plan_reused"] is False
    assert not dev._plans


def test_the_plan_cache_is_bounded():
    dev = qml.device("lazyqc.qubit", wires=6)
    for n in range(1, 7):
        for _ in range(2):
            qnode(dev, "parameter-shift", n)(weights(n, n)) if n > 1 else None
    assert len(dev._plans) <= frontend._PLAN_CACHE
    assert len(dev._once) <= 4 * frontend._PLAN_CACHE
