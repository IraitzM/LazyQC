"""Qiskit frontend: plain Qiskit code in, IR requests out."""

import pytest

from lazyqc import plan_all
from lazyqc.frontends import from_mqt_core, from_qiskit

qiskit = pytest.importorskip("qiskit")


def test_bell_translates_and_collects():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    imported = from_qiskit(circuit, [{0: "Z", 1: "Z"}, {0: "X", 1: "X"}])
    assert imported.translated_ops == ("H", "CX")
    assert imported.dropped == ()
    values = plan_all(imported.requests).collect().values
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(1.0)


def test_rzz_unrolls_to_basis():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rzz(0.5, 0, 1)
    imported = from_qiskit(circuit, [{0: "Z", 1: "Z"}])
    assert set(imported.translated_ops) <= {"H", "CX", "RZ"}
    assert "rzz" in imported.source_ops
    values = plan_all(imported.requests).collect().values
    # RZZ(0.5) on |+0> balances the ZZ parity subspaces exactly.
    assert values[0] == pytest.approx(0.0)


def test_parameterized_circuit_needs_bindings():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    circuit = QuantumCircuit(1)
    circuit.ry(theta, 0)
    with pytest.raises(ValueError, match="bindings"):
        from_qiskit(circuit, [{0: "Z"}])
    imported = from_qiskit(circuit, [{0: "Z"}], bindings={"theta": 0.0})
    values = plan_all(imported.requests).collect().values
    assert values[0] == pytest.approx(1.0)


def test_measurements_rejected_unless_stripped():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    with pytest.raises(ValueError, match="measurements"):
        from_qiskit(circuit, [{0: "Z"}])
    imported = from_qiskit(circuit, [{0: "Z"}], strip_measurements=True)
    assert imported.translated_ops == ("H",)


def test_nonunitary_is_rejected():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(1)
    circuit.reset(0)
    with pytest.raises(ValueError, match="without IR translation"):
        from_qiskit(circuit, [{0: "Z"}])


def test_barriers_dropped_and_recorded():
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.barrier()
    circuit.cx(0, 1)
    imported = from_qiskit(circuit, [{0: "Z", 1: "Z"}])
    assert imported.dropped == ("barrier",)
    assert imported.translated_ops == ("H", "CX")


def test_mqt_core_path():
    mqt_core = pytest.importorskip("mqt.core")
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    computation = mqt_core.load(circuit)
    imported = from_mqt_core(computation, [{0: "Z", 1: "Z"}])
    values = plan_all(imported.requests).collect().values
    assert values[0] == pytest.approx(1.0)
