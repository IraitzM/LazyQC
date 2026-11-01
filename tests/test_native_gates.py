"""Native CZ and RZZ gates in the IR (issue #14)."""

import math
import random

import numpy as np
import pytest
from reference_numpy import expectation, statevector

from lazyqc import Gate, Parameter, Program, adjoint_gradient, plan_all
from lazyqc.backends.qulacs import QulacsBackend
from lazyqc.ir import DIAGONAL_GATES, GATES, ROTATIONS, TWO_QUBIT_GATES
from lazyqc.passes import cone_indices
from lazyqc.rewrites import local_simplify


def random_program(n, gates, seed, native=True):
    """Random circuit with CZ and RZZ; `native=False` writes the translated forms instead."""
    rng = random.Random(seed)
    program = Program(n)
    for _ in range(gates):
        kind = rng.choice(["h", "ry", "rx", "rz", "cx", "cz", "rzz"])
        a, b = rng.sample(range(n), 2)
        angle = rng.uniform(-3.1, 3.1)
        if kind == "h":
            program = program.h(a)
        elif kind in ("ry", "rx", "rz"):
            program = getattr(program, kind)(a, angle)
        elif kind == "cx":
            program = program.cx(a, b)
        elif kind == "cz":
            program = program.cz(a, b) if native else program.h(b).cx(a, b).h(b)
        else:
            program = program.rzz(a, b, angle) if native else program.cx(a, b).rz(b, angle).cx(a, b)
    return program


def oracle_values(program, observables):
    operations = [(g.name, g.qubits, g.angle) for g in program.resolved_operations()]
    state = statevector(program.n_qubits, operations)
    return [expectation(state, o.items()) for o in observables]


OBSERVABLES = [{0: "Z"}, {1: "X", 3: "Y"}, {0: "Z", 2: "Z", 4: "X"}, {4: "Y"}]


def test_gate_rules():
    assert {"CZ", "RZZ"} <= GATES and "RZZ" in ROTATIONS and "CZ" not in ROTATIONS
    assert TWO_QUBIT_GATES == {"CX", "CZ", "RZZ"} and {"CZ", "RZZ", "RZ", "Z"} == DIAGONAL_GATES
    assert Gate("CZ", (0, 1)).qubits == (0, 1) and Gate("RZZ", (2, 0), 0.5).angle == 0.5
    for bad in (Gate,):
        with pytest.raises(ValueError, match="2 distinct"):
            bad("CZ", (1, 1))
        with pytest.raises(ValueError, match="2 distinct"):
            bad("RZZ", (0,), 0.1)
        with pytest.raises(ValueError, match="angle"):
            bad("CZ", (0, 1), 0.3)
    assert Gate("RZZ", (0, 1), 0.4).inverse() == Gate("RZZ", (0, 1), -0.4)
    assert Gate("CZ", (0, 1)).inverse() == Gate("CZ", (0, 1))
    assert Gate("RZZ", (0, 1), 0.4).effects.basis_preserving
    theta = Parameter("theta")
    assert Program(2).rzz(0, 1, theta).operations[0].angle == theta
    assert Program(["a", "b"]).cz("a", "b").operations == (Gate("CZ", (0, 1)),)


@pytest.mark.parametrize("seed", range(6))
def test_qulacs_agrees_with_the_dense_oracle(seed):
    program = random_program(5, 25, seed)
    values = plan_all([program.expectation(o) for o in OBSERVABLES], strategy="full").collect()
    np.testing.assert_allclose(values.values, oracle_values(program, OBSERVABLES), atol=1e-12)


def backend_factories():
    factories = {"qulacs": QulacsBackend}

    def aer():
        pytest.importorskip("qiskit_aer")
        from lazyqc.backends.aer import AerBackend

        return AerBackend(device="CPU")

    def lightning():
        pytest.importorskip("pennylane_lightning")
        from lazyqc.backends.lightning import LightningBackend

        return LightningBackend("lightning.qubit")

    def quimb():
        pytest.importorskip("quimb")
        from lazyqc.backends.tensor import QuimbBackend

        return QuimbBackend()

    def mps():
        pytest.importorskip("quimb")
        from lazyqc.backends.tensor import QuimbMPSBackend

        return QuimbMPSBackend()

    return factories | {"aer": aer, "lightning": lightning, "quimb": quimb, "mps": mps}


@pytest.mark.parametrize("name", list(backend_factories()))
@pytest.mark.parametrize("seed", range(3))
def test_every_backend_agrees_with_the_oracle_and_with_translated_forms(name, seed):
    backend = backend_factories()[name]()
    assert {"CZ", "RZZ"} <= set(backend.capabilities()["gates"])
    native = random_program(5, 25, seed)
    translated = random_program(5, 25, seed, native=False)
    expected = oracle_values(native, OBSERVABLES)
    for program in (native, translated):
        requests = [program.expectation(o) for o in OBSERVABLES]
        plan = plan_all(requests, strategy="full")
        kwargs = {"mps_backend": backend} if name == "mps" else {"backend": backend}
        if name == "mps":
            plan = plan_all(requests, strategy="mps")
        values = plan.collect(**kwargs).values
        np.testing.assert_allclose(values, expected, atol=1e-12)


def test_cone_follows_both_wires_of_a_native_two_qubit_gate():
    qubits = ((0,), (1,), (0, 1), (2,))
    wires, retained = cone_indices(qubits, [0])
    assert wires == (0, 1) and retained == (0, 1, 2)
    program = Program(3).rzz(1, 2, 0.4).h(1).cz(0, 1).rzz(0, 2, 0.9)
    plan = plan_all([program.expectation({0: "Z"})], strategy="grouped")
    # The last RZZ touches wire 0, so the whole circuit is in the cone of Z0.
    assert plan.selected.tasks[0].wires == (0, 1, 2)
    assert [g.name for g in plan.selected.tasks[0].operations] == ["RZZ", "H", "CZ", "RZZ"]
    outside = Program(3).h(1).cz(0, 1).rzz(1, 2, 0.4)  # RZZ(1, 2) runs after the last use of 0
    task = plan_all([outside.expectation({0: "Z"})], strategy="grouped").selected.tasks[0]
    assert task.wires == (0, 1) and [g.name for g in task.operations] == ["H", "CZ"]


def test_local_simplify_cancels_native_gates():
    cz, rzz = Gate("CZ", (0, 1)), Gate("RZZ", (0, 1), 0.7)
    assert local_simplify((cz, cz)) == ()
    assert local_simplify((rzz, Gate("RZZ", (0, 1), -0.7))) == ()
    assert local_simplify((Gate("RZZ", (0, 1), 0.0),)) == ()
    other = Gate("RZZ", (1, 2), -0.7)
    assert local_simplify((rzz, other)) == (rzz, other)  # different wires never cancel


def test_qiskit_frontend_keeps_the_native_gates():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp, Statevector

    from lazyqc.frontends import from_qiskit

    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.rzz(0.7, 1, 2)
    circuit.ry(0.3, 2)
    imported = from_qiskit(circuit, [{0: "X", 2: "Z"}, {1: "Y"}])
    assert imported.translated_ops == ("H", "CZ", "RZZ", "RY")
    values = plan_all(imported.requests).collect().values
    state = Statevector(circuit)
    for value, label in zip(values, ("ZIX", "IYI"), strict=True):
        assert value == pytest.approx(state.expectation_value(SparsePauliOp(label)).real, abs=1e-12)


def test_estimator_with_parametric_rzz_matches_statevector_estimator():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter as QiskitParameter
    from qiskit.primitives import StatevectorEstimator
    from qiskit.quantum_info import SparsePauliOp

    from lazyqc.frontends.qiskit import Estimator

    a, b = QiskitParameter("a"), QiskitParameter("b")
    circuit = QuantumCircuit(4)
    circuit.h(range(4))
    circuit.rzz(2 * a + 0.1, 0, 1)
    circuit.rzz(b, 1, 2)
    circuit.cz(2, 3)
    circuit.rx(a * b, 3)
    observables = [SparsePauliOp(["ZZII", "IXXI"], [0.4, -1.1]), SparsePauliOp("IIIZ")]
    values = np.random.default_rng(1).uniform(-3, 3, (7, 2))
    pub = (circuit, [observables], values[:, None, :])
    ours = Estimator().run([pub]).result()[0].data.evs
    reference = StatevectorEstimator().run([pub]).result()[0].data.evs
    np.testing.assert_allclose(ours, reference, atol=1e-10)


def test_rzz_gradient_matches_shift_rule_and_finite_differences():
    theta = Parameter("theta")
    program = Program(3).ry(0, 0.4).ry(1, 0.9).rzz(0, 1, theta).ry(2, 0.3).cz(1, 2).rx(0, 0.6)
    observables = [{0: "Y"}, {1: "X", 2: "Z"}]
    weights = [0.7, -1.3]

    def loss(value):
        bound = program.bind(theta=value)
        values = plan_all([bound.expectation(o) for o in observables]).collect().values
        return sum(w * v for w, v in zip(weights, values, strict=True))

    point, step = 0.35, 1e-5
    finite = (loss(point + step) - loss(point - step)) / (2 * step)
    shift = 0.5 * (loss(point + math.pi / 2) - loss(point - math.pi / 2))
    gradient = adjoint_gradient(program.bind(theta=point), observables, weights)["theta"]
    assert shift == pytest.approx(finite, abs=1e-8)
    assert gradient == pytest.approx(shift, abs=1e-10)


def test_pennylane_device_runs_native_gates_and_gradients():
    qml = pytest.importorskip("pennylane")
    pnp = qml.numpy

    def circuit(w):
        for q in range(3):
            qml.Hadamard(wires=q)
        qml.IsingZZ(w[0], wires=[0, 1])
        qml.CZ(wires=[1, 2])
        qml.IsingZZ(w[1], wires=[0, 2])
        qml.RX(w[2], wires=1)
        return qml.math.stack([qml.expval(qml.Y(0)), qml.expval(qml.X(1) @ qml.Z(2))])

    w = pnp.array([0.4, -0.8, 1.1], requires_grad=True)
    ours = qml.QNode(circuit, qml.device("lazyqc.qubit", wires=3), diff_method="parameter-shift")
    ref = qml.QNode(circuit, qml.device("default.qubit", wires=3), diff_method="backprop")
    assert np.allclose(ours(w), ref(w), atol=1e-10)
    assert np.allclose(qml.jacobian(ours)(w), qml.jacobian(ref)(w), atol=1e-8)


def test_zx_conversion_preserves_native_gates():
    pytest.importorskip("pyzx")
    from test_zx import matrix

    from lazyqc.rewrites import from_pyzx, to_pyzx

    operations = (
        Gate("H", (0,)),
        Gate("CZ", (0, 1)),
        Gate("RZZ", (1, 2), 0.7),
        Gate("RY", (2,), 0.3),
        Gate("RZZ", (0, 2), -1.2),
    )
    converted = from_pyzx(to_pyzx(3, operations), 3)
    before, after = matrix(3, operations), matrix(3, converted)
    phase = np.vdot(before.ravel(), after.ravel())
    np.testing.assert_allclose(after, before * (phase / abs(phase)), atol=1e-10)


def test_mps_bond_model_counts_native_gates_like_cx():
    pytest.importorskip("quimb")
    import quimb.tensor as qtn

    from lazyqc.tensor_cost import mps_bond_profile

    program = random_program(8, 40, 3)
    mps = qtn.CircuitMPS(8, cutoff=0.0)
    for g in program.resolved_operations():
        mps.apply_gate(g.name, *([] if g.angle is None else [g.angle]), *g.qubits)
    measured = tuple(mps.psi.bond_size(i, i + 1) for i in range(7))
    assert mps_bond_profile(8, program.resolved_operations()) == measured
