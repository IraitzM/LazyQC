"""The Qiskit Estimator plans a repeated job structure once (issue #32)."""

import numpy as np
import pytest

pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402
from qiskit.primitives import StatevectorEstimator  # noqa: E402
from qiskit.quantum_info import SparsePauliOp  # noqa: E402
from test_qiskit_estimator import ansatz, random_observables  # noqa: E402

from lazyqc.frontends import qiskit_estimator  # noqa: E402
from lazyqc.frontends.qiskit import Estimator  # noqa: E402


class Counter:
    """Wrap `plan_all` to count calls and the `symbolic` flag they carry."""

    def __init__(self, monkeypatch):
        self.calls = []
        real = qiskit_estimator.plan_all
        monkeypatch.setattr(
            qiskit_estimator,
            "plan_all",
            lambda *a, **k: self.calls.append(bool(k.get("symbolic"))) or real(*a, **k),
        )


def run(estimator, pubs):
    return estimator.run(pubs).result()


def reference(pubs):
    return StatevectorEstimator().run(pubs).result()


def test_repeated_jobs_plan_once_and_match_the_reference(monkeypatch):
    counter = Counter(monkeypatch)
    circuit = ansatz(6)
    observables = [random_observables(6, 3, 5)]
    estimator = Estimator()
    rng = np.random.default_rng(0)
    reused = []
    for _ in range(5):
        pubs = [(circuit, observables, rng.uniform(-3, 3, (4, 1, 2)))]
        got = run(estimator, pubs)
        np.testing.assert_allclose(got[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10)
        reused.append(got.metadata["lazyqc"]["plan_reused"])
    assert reused == [False, False, True, True, True]
    assert counter.calls == [False, True]  # fresh on first sight, one symbolic plan after


def test_values_equal_the_fresh_plan_path_exactly():
    circuit = ansatz(5)
    observables = [random_observables(5, 2, 1)]
    rng = np.random.default_rng(3)
    kept, fresh = Estimator(), Estimator()
    for _ in range(4):
        pubs = [(circuit, observables, rng.uniform(-3, 3, (3, 1, 2)))]
        fresh._plans.clear()  # forces the fresh-plan path
        fresh._once.clear()
        a = run(kept, pubs)[0].data.evs
        b = run(fresh, pubs)[0].data.evs
        assert np.array_equal(a, b)


def test_several_pubs_with_shared_and_distinct_circuits():
    first, second = ansatz(4), ansatz(5, layers=1)
    rng = np.random.default_rng(1)
    estimator = Estimator()
    for _ in range(4):
        pubs = [
            (first, random_observables(4, 2, 1), rng.uniform(-3, 3, (2, 2))),
            (second, random_observables(5, 1, 2)[0], rng.uniform(-3, 3, (3, 2, 2))),
            (QuantumCircuit(2), SparsePauliOp("ZZ")),
        ]
        got, want = run(estimator, pubs), reference(pubs)
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g.data.evs, w.data.evs, atol=1e-10)


def test_three_parameter_u_gate_and_derived_angles():
    theta, phi, lam = Parameter("theta"), Parameter("phi"), Parameter("lam")
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.u(theta, phi, lam, 0)
    circuit.cx(0, 1)
    circuit.u(lam * 0.5, theta + phi, 0.4, 1)
    observables = [SparsePauliOp(["XY", "ZZ", "IX"], [0.7, -1.2, 0.3])]
    rng = np.random.default_rng(4)
    estimator = Estimator()
    for _ in range(4):
        pubs = [(circuit, observables, rng.uniform(-3, 3, (6, 1, 3)))]
        np.testing.assert_allclose(
            run(estimator, pubs)[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10
        )


def test_new_observables_or_row_count_get_their_own_plan(monkeypatch):
    counter = Counter(monkeypatch)
    circuit = ansatz(4)
    estimator = Estimator()
    rng = np.random.default_rng(2)
    one, other = [random_observables(4, 2, 1)], [random_observables(4, 2, 9)]
    for observables, rows in [(one, 3), (one, 3), (other, 3), (other, 3), (one, 5), (one, 5)]:
        pubs = [(circuit, observables, rng.uniform(-3, 3, (rows, 1, 2)))]
        np.testing.assert_allclose(
            run(estimator, pubs)[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10
        )
    assert counter.calls.count(True) == 3  # one symbolic plan per distinct structure


def test_mutating_the_circuit_invalidates_the_plan():
    theta = Parameter("theta")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    observable = SparsePauliOp("ZZ")
    estimator = Estimator()
    for angle in (0.3, 0.9, 1.4):
        pubs = [(circuit, observable, [angle])]
        np.testing.assert_allclose(
            run(estimator, pubs)[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10
        )
    circuit.rx(0.5, 1)
    pubs = [(circuit, observable, [0.3])]
    got = run(estimator, pubs)
    np.testing.assert_allclose(got[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10)
    assert got.metadata["lazyqc"]["plan_reused"] is False


@pytest.mark.parametrize("option", [{"optimization": "local"}, {"workers": 2}])
def test_numeric_only_options_keep_the_fresh_plan_path(monkeypatch, option):
    counter = Counter(monkeypatch)
    circuit = ansatz(4)
    observables = [random_observables(4, 2, 1)]
    estimator = Estimator(plan_options=option)
    rng = np.random.default_rng(5)
    for _ in range(3):
        pubs = [(circuit, observables, rng.uniform(-3, 3, (2, 1, 2)))]
        got = run(estimator, pubs)
        np.testing.assert_allclose(got[0].data.evs, reference(pubs)[0].data.evs, atol=1e-10)
        assert got.metadata["lazyqc"]["plan_reused"] is False
    assert not any(counter.calls)


def test_the_plan_cache_is_bounded():
    circuit = ansatz(3)
    estimator = Estimator()
    rng = np.random.default_rng(6)
    for seed in range(12):
        observables = [random_observables(3, 1, seed)]
        for _ in range(2):  # second sight builds the symbolic plan
            run(estimator, [(circuit, observables, rng.uniform(-3, 3, (2, 1, 2)))])
    assert len(estimator._plans) <= qiskit_estimator._PLAN_CACHE
    assert len(estimator._once) <= 4 * qiskit_estimator._PLAN_CACHE


def test_noise_is_unchanged_by_plan_reuse():
    circuit = ansatz(4)
    observables = [random_observables(4, 2, 1)]
    values = np.random.default_rng(7).uniform(-3, 3, (3, 1, 2))
    pubs = [(circuit, observables, values)]
    estimator = Estimator(default_precision=0.05, seed=11)
    runs = [run(estimator, pubs)[0].data.evs for _ in range(3)]
    for evs in runs:
        want = StatevectorEstimator(default_precision=0.05, seed=11).run(pubs).result()
        np.testing.assert_allclose(evs, want[0].data.evs, atol=1e-10)
