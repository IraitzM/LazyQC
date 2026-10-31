"""Exact-MPS fallback when no state-vector plan is feasible (issue #4)."""

import random

import pytest

from lazyqc import NoFeasiblePlan, Pauli, Program, plan_all
from lazyqc.tensor_cost import mps_bond_profile, mps_max_bond

qtn = pytest.importorskip("quimb.tensor")


def random_program(n, gates, seed, long_range=True):
    rng = random.Random(seed)
    program = Program(n)
    for _ in range(gates):
        kind = rng.choice(["RX", "RY", "RZ", "H", "CX"])
        if kind == "CX":
            a = rng.randrange(n)
            b = rng.randrange(n) if long_range else min(a + 1, n - 1)
            if a != b:
                program = program.cx(a, b)
        elif kind == "H":
            program = program.h(rng.randrange(n))
        else:
            program = getattr(program, kind.lower())(rng.randrange(n), rng.uniform(-3, 3))
    return program


def quimb_bonds(program):
    mps = qtn.CircuitMPS(program.n_qubits, cutoff=0.0)
    for g in program.resolved_operations():
        args = ([] if g.angle is None else [g.angle]) + list(g.qubits)
        mps.apply_gate(g.name, *args)
    return tuple(mps.psi.bond_size(i, i + 1) for i in range(program.n_qubits - 1))


@pytest.mark.parametrize("long_range", [False, True])
@pytest.mark.parametrize("seed", range(5))
def test_bond_model_matches_quimb(seed, long_range):
    program = random_program(9, 40, seed, long_range)
    operations = program.resolved_operations()
    assert len(operations) > 20
    assert mps_bond_profile(9, operations) == quimb_bonds(program)
    assert mps_max_bond(9, operations) == max(quimb_bonds(program))


def ghz_ladder(n, layers=3):
    program = Program(n)
    for layer in range(layers):
        for q in range(n):
            program = program.ry(q, 0.3 + 0.1 * q + layer)
        for q in range(n - 1):
            program = program.cx(q, q + 1)
    return program


def requests_for(program):
    n = program.n_qubits
    return [
        program.expectation(Pauli(((0, "Z"), (n - 1, "Z")))),
        program.expectation(Pauli(((n // 2, "X"),))),
    ]


def test_saturated_batch_falls_back_to_mps_and_matches_reference():
    program = ghz_ladder(12)
    requests = requests_for(program)
    reference = plan_all(requests, max_state_qubits=14).collect().values
    with pytest.raises(NoFeasiblePlan):
        plan_all(requests, max_state_qubits=2, analysis="always", strategy="grouped")
    plan = plan_all(requests, max_state_qubits=2)
    assert plan.selected.strategy == "mps"
    assert plan.selected.representation == "mps"
    result = plan.collect()
    assert result.values == pytest.approx(reference, abs=1e-9)
    assert result.stats.allocations == 1 and result.stats.max_state_qubits == 0


def test_feasible_state_vector_plan_keeps_priority():
    program = ghz_ladder(8)
    plan = plan_all(requests_for(program), max_state_qubits=12)
    assert plan.selected.representation == "statevector"
    assert all(a.strategy != "mps" for a in plan.alternatives)


def test_explain_reports_representation_and_reason():
    plan = plan_all(requests_for(ghz_ladder(12)), max_state_qubits=2)
    explanation = plan.explain()
    assert explanation["representation"] == "mps"
    assert explanation["max_mps_bond"] == mps_max_bond(12, ghz_ladder(12).resolved_operations())
    assert "exact MPS fallback" in explanation["selection_reason"]
    assert explanation["estimated_state_bytes"] == plan.selected.state_bytes


def test_explicit_mps_strategy_and_payload_limit():
    requests = requests_for(ghz_ladder(10))
    plan = plan_all(requests, strategy="mps")
    assert plan.selected.strategy == "mps"
    with pytest.raises(NoFeasiblePlan, match="MPS payload"):
        plan_all(requests, strategy="mps", max_state_bytes=1024)
    with pytest.raises(NoFeasiblePlan, match="MPS payload"):
        plan_all(requests, max_state_qubits=2, max_state_bytes=1024)


def test_constant_outputs_survive_mps_plan():
    program = ghz_ladder(8)
    requests = [*requests_for(program), program.expectation(Pauli(()))]
    values = plan_all(requests, max_state_qubits=2).collect().values
    assert values[-1] == 1.0
