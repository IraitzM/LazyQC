"""Weighted Pauli sums as observables (issue #12)."""

import random

import pytest

from lazyqc import CutOptions, Pauli, PauliSum, Program, SumRequest, plan_all

qiskit_quantum_info = pytest.importorskip("qiskit.quantum_info")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402


def random_program(n, seed, layers=3):
    rng = random.Random(seed)
    program = Program(n)
    for _ in range(layers):
        for q in range(n):
            program = program.ry(q, rng.uniform(-3, 3)).rz(q, rng.uniform(-3, 3))
        for q in range(n - 1):
            program = program.cx(q, q + 1)
    return program


def qiskit_circuit(program):
    circuit = QuantumCircuit(program.n_qubits)
    for gate in program.resolved_operations():
        if gate.name == "CX":
            circuit.cx(*gate.qubits)
        else:
            getattr(circuit, gate.name.lower())(gate.angle, *gate.qubits)
    return circuit


def to_sum(operator):
    """Map a SparsePauliOp to LazyQC terms; Qiskit labels put qubit 0 rightmost."""
    terms = []
    for label, coefficient in operator.to_list():
        assert abs(coefficient.imag) < 1e-12
        factors = tuple((q, p) for q, p in enumerate(reversed(label)) if p != "I")
        terms.append((coefficient.real, Pauli(factors)))
    return PauliSum(tuple(terms))


@pytest.mark.parametrize("n", [3, 6, 9, 12])
def test_sums_agree_with_qiskit_sparse_pauli_op(n):
    rng = random.Random(n)
    program = random_program(n, seed=n)
    state = Statevector(qiskit_circuit(program))
    operators = []
    for _ in range(3):
        labels = ["".join(rng.choice("IXYZ") for _ in range(n)) for _ in range(12)]
        operators.append(SparsePauliOp(labels, [rng.uniform(-2, 2) for _ in labels]))
    requests = [program.expectation(to_sum(op)) for op in operators]
    values = plan_all(requests).collect().values
    expected = [state.expectation_value(op).real for op in operators]
    assert values == pytest.approx(expected, abs=1e-10)


def test_sum_matches_term_by_term_and_keeps_request_order():
    program = random_program(5, seed=1)
    z = Pauli(((0, "Z"),))
    xx = Pauli(((1, "X"), (2, "X")))
    plain = [program.expectation(z), program.expectation(xx)]
    sums = program.expectation([(0.5, z), (-2.0, xx), (3.0, Pauli())])
    values = plan_all([plain[0], sums, plain[1]]).collect().values
    terms = plan_all(plain).collect().values
    assert len(values) == 3
    assert values[0] == pytest.approx(terms[0], abs=1e-12)
    assert values[2] == pytest.approx(terms[1], abs=1e-12)
    assert values[1] == pytest.approx(0.5 * terms[0] - 2.0 * terms[1] + 3.0, abs=1e-12)


def test_terms_with_equal_cones_share_tasks():
    program = random_program(8, seed=2)
    terms = [
        (1.0, Pauli(((0, "Z"), (1, "Z")))),
        (0.5, Pauli(((0, "X"), (1, "X")))),
        (0.25, Pauli(((0, "Y"), (1, "Y")))),
    ]
    plan = plan_all([program.expectation(terms)])
    assert len(plan.selected.tasks) < len(terms)
    assert plan.explain()["request_count"] == len(terms)


def test_explain_reports_sum_mapping():
    program = random_program(4, seed=3)
    request = program.expectation({0: "Z"})
    total = program.expectation([(2.0, {0: "Z"}), (-1.0, {1: "X"})])
    explanation = plan_all([request, total]).explain()
    assert explanation["original_request_count"] == 2
    assert explanation["sums"] == [
        {"request_index": 1, "term_request_indices": [1, 2], "coefficients": [2.0, -1.0]}
    ]


def test_repeated_terms_merge_and_identity_is_an_offset():
    z = Pauli(((0, "Z"),))
    merged = PauliSum(((1.0, z), (2.5, z), (4.0, Pauli())))
    assert merged.terms == ((3.5, z), (4.0, Pauli()))
    program = Program(1).x(0)
    assert plan_all([program.expectation(merged)]).collect().values == pytest.approx((-3.5 + 4.0,))
    assert plan_all([program.expectation([(5.0, Pauli())])]).collect().values == (5.0,)


def test_empty_sum_is_zero():
    program = Program(2).h(0)
    values = plan_all([program.expectation({0: "X"}), program.expectation([])]).collect().values
    assert values == pytest.approx((1.0, 0.0))


@pytest.mark.parametrize("bad", [1 + 2j, True, float("nan"), float("inf"), "1.0", None])
def test_non_real_or_non_finite_coefficients_are_rejected(bad):
    with pytest.raises(ValueError, match="coefficients"):
        PauliSum(((bad, Pauli(((0, "Z"),))),))


def test_bad_terms_are_rejected():
    with pytest.raises(ValueError):
        PauliSum(((1.0, "Z"),))
    with pytest.raises(ValueError, match="outside the program"):
        SumRequest(Program(1), PauliSum(((1.0, Pauli(((3, "Z"),))),)))


def test_sums_work_on_the_batch_path_and_with_a_cache():
    program = random_program(6, seed=4)
    request = program.expectation([(0.3, {0: "Z", 5: "Z"}), (-0.7, {2: "X"})])
    direct = plan_all([request]).collect().values
    batch = plan_all([request], cutting=CutOptions(max_fragment_qubits=6, max_cuts=0)).collect()
    assert batch.values == pytest.approx(direct, abs=1e-10)


def test_recombination_uses_fsum():
    from lazyqc.sums import recombine

    big = 1e16
    layout = [("sum", ((0, 1.0), (1, 1.0), (2, 1.0)))]
    # A naive left-to-right sum loses the 1.0; fsum keeps it.
    assert (big + 1.0) + -big == 0.0
    assert recombine([big, 1.0, -big], layout) == (1.0,)
