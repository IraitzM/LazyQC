"""Logical basis meaning, padded subspaces, and strict declarative frontend scope."""

import copy

import numpy as np
import pytest
from reference_numpy import statevector

from lazyqc import CutOptions, plan_all
from lazyqc.frontends import from_rhyme_ir

pytest.importorskip("qiskit")


def fixture():
    return {
        "schema_version": 1,
        "subsystems": [{"name": "level", "basis": [10, 20, 30]}],
        "operations": [
            {"kind": "interfere", "targets": ["level"], "pairs": [[0, 1]]},
            {"kind": "phase", "targets": ["level"], "angles": [0, 0.7, 0]},
            {"kind": "interfere", "targets": ["level"], "pairs": [[1, 2]]},
        ],
    }


def test_qutrit_decoding_padding_and_global_phase():
    data = fixture()
    original = copy.deepcopy(data)
    imported = from_rhyme_ir(data)
    assert data == original
    assert imported.program.n_qubits == 2
    queries = [imported.probability("level", value) for value in (10, 20, 30)]
    queries.extend(imported.invalid_probabilities("level"))
    for query, probability in zip(queries, (0.5, 0.25, 0.25, 0), strict=True):
        values = plan_all(query.requests, cutting=CutOptions(max_cuts=0)).collect().values
        assert query.evaluate(values) == pytest.approx(probability, abs=1e-10)
    p = imported.program
    state = statevector(p.n_qubits, [(g.name, g.qubits, g.angle) for g in p.operations])
    expected = np.array([1 / np.sqrt(2), np.exp(0.7j) / 2, np.exp(0.7j) / 2, 0])
    assert np.exp(1j * imported.global_phase) * state == pytest.approx(expected, abs=1e-10)


def test_target_order_and_joint_permutation():
    data = {
        "schema_version": 1,
        "subsystems": [
            {"name": "bit", "basis": ["off", "on"]},
            {"name": "level", "basis": [10, 20, 30]},
        ],
        "operations": [
            {"kind": "permutation", "targets": ["level", "bit"], "mapping": [5, 1, 2, 3, 4, 0]}
        ],
    }
    imported = from_rhyme_ir(data)
    for subsystem, value in (("bit", "on"), ("level", 30)):
        q = imported.probability(subsystem, value)
        assert q.evaluate(plan_all(q.requests).collect().values) == pytest.approx(1, abs=1e-10)
    p = imported.program
    s = statevector(p.n_qubits, [(g.name, g.qubits, g.angle) for g in p.operations])
    # bit = wire 0; level ordinal 2 occupies wires 1,2 => encoded index 5.
    assert abs(s[5]) ** 2 == pytest.approx(1, abs=1e-10)
    assert abs(s[6]) + abs(s[7]) < 1e-10


def test_entire_qutrit_unitary_preserves_padded_subspace():
    imported = from_rhyme_ir(fixture())
    h01 = np.eye(4, dtype=complex)
    h01[:2, :2] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    h12 = np.eye(4, dtype=complex)
    h12[1:3, 1:3] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    expected = h12 @ np.diag([1, np.exp(0.7j), 1, 1]) @ h01
    ops = [(g.name, g.qubits, g.angle) for g in imported.program.operations]
    for code in range(4):
        preparation = [("X", (q,), None) for q in range(2) if (code >> q) & 1]
        actual = np.exp(1j * imported.global_phase) * statevector(2, preparation + ops)
        assert actual == pytest.approx(expected[:, code], abs=1e-10)


def test_frontend_does_not_execute_native_backend(monkeypatch):
    import qulacs

    def forbidden(*a, **kw):
        pytest.fail("Native state allocation during import")

    monkeypatch.setattr(qulacs, "QuantumState", forbidden)
    from_rhyme_ir(fixture())


@pytest.mark.parametrize(
    "operation",
    [
        {"kind": "callback", "targets": ["level"], "function": lambda x: x},
        {"kind": "phase", "targets": ["level"], "angles": [0, float("nan"), 0]},
        {"kind": "permutation", "targets": ["level"], "mapping": [0, 0, 2]},
        {"kind": "interfere", "targets": ["level"], "pairs": [[0, 1], [1, 2]]},
        {"kind": "measure", "targets": ["level"]},
    ],
)
def test_unsupported_or_invalid_semantics_rejected(operation):
    data = fixture()
    data["operations"] = [operation]
    with pytest.raises(ValueError):
        from_rhyme_ir(data)
