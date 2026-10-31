"""Qiskit Estimator primitive backed by LazyQC (issues #5 and #13)."""

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402

from lazyqc.frontends import qiskit as frontend  # noqa: E402
from lazyqc.frontends import qiskit_estimator  # noqa: E402
from lazyqc.frontends.qiskit import Estimator  # noqa: E402


def ansatz(n, layers=2):
    """Parametric circuit with expression angles and gates outside the IR set."""
    a, b = Parameter("a"), Parameter("b")
    circuit = QuantumCircuit(n)
    for layer in range(layers):
        for q in range(n):
            circuit.ry(a if q % 2 else b, q)
            circuit.rz(2 * a + 0.3 * (q + layer), q)
        for q in range(n - 1):
            circuit.cx(q, q + 1)
        circuit.rzz(a * 0.5 + b, 0, n - 1)
        circuit.cz(0, n // 2)
    circuit.u(a, b, 0.2, n - 1)
    return circuit


def random_observables(n, count, seed):
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(count):
        labels = ["".join(rng.choice(list("IXYZ"), n)) for _ in range(5)]
        ops.append(SparsePauliOp(labels, rng.uniform(-2, 2, len(labels))))
    return ops


def compare(pubs, **estimator_options):
    ours = Estimator(**estimator_options).run(pubs).result()
    reference = StatevectorEstimator(**estimator_options).run(pubs).result()
    assert len(ours) == len(reference)
    for got, want in zip(ours, reference, strict=True):
        assert got.data.evs.shape == want.data.evs.shape
        np.testing.assert_allclose(got.data.evs, want.data.evs, atol=1e-10)
    return ours, reference


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_broadcast_pubs_agree_with_statevector_estimator(n):
    circuit = ansatz(n)
    rng = np.random.default_rng(n)
    observables = [random_observables(n, 3, n)]  # shape (1, 3)
    values = rng.uniform(-3, 3, (4, 1, 2))  # parameter shape (4, 1) broadcasts to (4, 3)
    compare([(circuit, observables, values)])


def test_several_pubs_with_different_shapes_in_one_job():
    first, second = ansatz(4), ansatz(5, layers=1)
    rng = np.random.default_rng(0)
    pubs = [
        (first, random_observables(4, 2, 1), rng.uniform(-3, 3, (2, 2))),  # shape (2,)
        (second, random_observables(5, 1, 2)[0], rng.uniform(-3, 3, (3, 2, 2))),  # shape (3, 2)
        (QuantumCircuit(2), SparsePauliOp("ZZ")),  # scalar, no parameters
    ]
    ours, _ = compare(pubs)
    assert [r.data.evs.shape for r in ours] == [(2,), (3, 2), ()]


def test_a_job_is_planned_once(monkeypatch):
    calls = []
    real = qiskit_estimator.plan_all
    monkeypatch.setattr(
        qiskit_estimator, "plan_all", lambda *a, **k: calls.append(len(a[0])) or real(*a, **k)
    )
    circuit = ansatz(4)
    rng = np.random.default_rng(1)
    pubs = [(circuit, random_observables(4, 3, i), rng.uniform(-3, 3, (3, 2))) for i in range(4)]
    result = Estimator().run(pubs).result()
    assert calls == [4 * 3]
    assert result.metadata["lazyqc"]["requests"] == 12


def test_rows_share_the_translated_circuit_and_cones(monkeypatch):
    transpiles = []
    real = frontend._to_basis
    monkeypatch.setattr(frontend, "_to_basis", lambda c: transpiles.append(1) or real(c))
    circuit = ansatz(6)
    observables = [random_observables(6, 2, 3)]  # shape (1, 2)
    values = np.random.default_rng(2).uniform(-3, 3, (20, 1, 2))
    result = Estimator().run([(circuit, observables, values)]).result()
    assert len(transpiles) == 1  # not one transpile per parameter row
    assert result[0].data.evs.shape == (20, 2)


def test_is_a_base_estimator_v2_and_exported_lazily():
    assert isinstance(Estimator(), BaseEstimatorV2)
    assert frontend.Estimator is Estimator
    with pytest.raises(AttributeError):
        frontend.NoSuchName


def test_zero_or_missing_precision_is_exact():
    circuit = ansatz(3)
    pub = (circuit, SparsePauliOp("XIZ"), [0.4, 1.1])
    exact = Estimator().run([pub]).result()[0]
    zero = Estimator(seed=1).run([pub], precision=0.0).result()[0]
    assert np.array_equal(exact.data.evs, zero.data.evs)
    assert not exact.data.stds.any()


def test_seeded_noise_matches_statevector_estimator_and_reports_precision():
    circuit = ansatz(4)
    observables = random_observables(4, 3, 7)
    values = np.random.default_rng(3).uniform(-3, 3, (5, 2))
    pubs = [(circuit, [observables], values[:, None, :])]
    ours, reference = compare(pubs, default_precision=0.05, seed=1234)
    assert ours[0].metadata["target_precision"] == 0.05
    # The reference leaves stds at zero; we report the precision that was applied.
    assert not reference[0].data.stds.any()
    assert np.all(ours[0].data.stds == 0.05)


def test_noise_has_the_requested_standard_deviation():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    pub = (circuit, [SparsePauliOp("Z")] * 4000)
    evs = Estimator(seed=5).run([pub], precision=0.1).result()[0].data.evs
    assert abs(evs.mean()) < 0.01  # exact value is 0
    assert evs.std() == pytest.approx(0.1, rel=0.05)


def test_per_pub_precision_overrides_the_default():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    noisy = (circuit, SparsePauliOp("Z"), None, 0.2)
    exact = (circuit, SparsePauliOp("Z"))
    result = Estimator(seed=3).run([noisy, exact]).result()
    assert result[0].metadata["target_precision"] == 0.2 and result[0].data.evs != 0.0
    assert result[1].data.evs == 0.0 and result[1].metadata["target_precision"] == 0.0


def test_unsupported_inputs_are_rejected():
    measured = QuantumCircuit(1, 1)
    measured.h(0)
    measured.measure(0, 0)
    with pytest.raises(ValueError, match="measurements"):
        Estimator().run([(measured, SparsePauliOp("Z"))]).result()
    anti = SparsePauliOp(["Z"], [1j])
    with pytest.raises(ValueError, match="Hermitian"):
        Estimator().run([(QuantumCircuit(1), anti)]).result()
    with pytest.raises(ValueError, match="qubits"):
        Estimator().run([(QuantumCircuit(2), SparsePauliOp("Z"))]).result()


def test_plan_and_collect_options_are_forwarded():
    circuit = ansatz(4)
    pub = (circuit, SparsePauliOp("ZZII"), [0.2, 0.7])
    result = Estimator(plan_options={"strategy": "full"}).run([pub]).result()
    assert result.metadata["lazyqc"]["selected_strategy"] == "full"


def test_u_gate_with_three_free_parameters_matches_statevector_estimator():
    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.u(theta, phi, lam, 0)
    circuit.cx(0, 1)
    circuit.u(lam * 0.5, theta + phi, 0.4, 1)
    observables = [SparsePauliOp(["XY", "ZZ", "IX"], [0.7, -1.2, 0.3])]
    values = np.random.default_rng(4).uniform(-3, 3, (6, 3))
    compare([(circuit, observables, values[:, None, :])])


def test_translation_is_reused_across_runs_and_invalidated_by_mutation(monkeypatch):
    transpiles = []
    real = frontend._to_basis
    monkeypatch.setattr(frontend, "_to_basis", lambda c: transpiles.append(1) or real(c))
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    estimator = Estimator()
    observable = SparsePauliOp("ZZ")
    for angle in (0.3, 0.9, 1.4):
        got = estimator.run([(circuit, observable, [angle])]).result()[0].data.evs
        want = StatevectorEstimator().run([(circuit, observable, [angle])]).result()[0].data.evs
        np.testing.assert_allclose(got, want, atol=1e-10)
    assert len(transpiles) == 1
    circuit.rx(0.5, 1)  # same object, different circuit
    got = estimator.run([(circuit, observable, [0.3])]).result()[0].data.evs
    want = StatevectorEstimator().run([(circuit, observable, [0.3])]).result()[0].data.evs
    np.testing.assert_allclose(got, want, atol=1e-10)
    assert len(transpiles) == 2
