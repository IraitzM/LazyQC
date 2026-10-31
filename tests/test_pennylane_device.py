import numpy as np
import pytest

qml = pytest.importorskip("pennylane")
pnp = qml.numpy

from lazyqc.frontends.pennylane import LazyQubit  # noqa: E402


def _circuit(n, layers, obs):
    def circuit(w):
        k = 0
        for _ in range(layers):
            for q in range(n):
                qml.RX(w[k], wires=q)
                qml.RY(w[k + 1], wires=q)
                k += 2
            for q in range(n - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.math.stack([qml.expval(o) for o in obs])

    return circuit


@pytest.mark.parametrize("n", [2, 5, 8])
def test_values_and_jacobian_match_default_qubit(n):
    rng = np.random.default_rng(n)
    obs = [
        qml.Z(0),
        qml.X(0) @ qml.Y(n - 1),
        qml.Hamiltonian([0.5, -1.2], [qml.Z(1), qml.X(0) @ qml.X(1)]),
    ]
    w = pnp.array(rng.uniform(0, 2 * np.pi, 4 * n), requires_grad=True)
    f = _circuit(n, 2, obs)
    dev = qml.device("lazyqc.qubit", wires=n)
    ref = qml.device("default.qubit", wires=n)
    for diff in ("parameter-shift", "best"):
        a = qml.QNode(f, dev, diff_method=diff)
        b = qml.QNode(f, ref, diff_method="backprop")
        assert np.allclose(a(w), b(w), atol=1e-10)
        ja = qml.jacobian(a)(w)
        jb = qml.jacobian(b)(w)
        assert np.allclose(ja, jb, atol=1e-10)


def test_decomposition_chain_rule_and_structural_zeros():
    dev = qml.device("lazyqc.qubit", wires=3)
    ref = qml.device("default.qubit", wires=3)

    def f(w):
        qml.Rot(w[0], w[1], w[2], wires=0)
        qml.Rot(w[3], w[4], w[5], wires=2)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))

    w = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], requires_grad=True)
    a = qml.QNode(f, dev, diff_method="best")
    b = qml.QNode(f, ref, diff_method="backprop")
    assert np.allclose(qml.grad(a)(w), qml.grad(b)(w), atol=1e-10)
    assert dev.last_stats["skipped_structural_zeros"] == 3
    assert np.allclose(qml.grad(a)(w)[3:], 0.0)


def test_rejects_shots_and_unsupported_observables():
    with pytest.raises(ValueError):
        qml.device("lazyqc.qubit", wires=1, shots=10)
    dev = LazyQubit(wires=1)

    @qml.qnode(dev)
    def f():
        return qml.expval(qml.Hadamard(0))

    with pytest.raises(Exception):
        f()
