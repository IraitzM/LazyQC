"""Independent dense validation, adversarial curvature and API safeguards."""

from dataclasses import replace
from math import cos, pi, sin

import numpy as np
import pytest
from reference_numpy import expectation, statevector

from lazyqc import Gate, Parameter, Program, adjoint_gradient, plan_all, profile_sensitivity
from lazyqc.sensitivity import plan_approximate


def oracle(program, observable):
    ops = [(g.name, g.qubits, g.angle) for g in program.resolved_operations()]
    return expectation(statevector(program.n_qubits, ops), observable)


def test_shared_parameter_requires_occurrence_shifts():
    p = Program(1).ry(0, Parameter("t")).ry(0, Parameter("t")).bind(t=0.3)
    profile = profile_sensitivity(p, [{0: "Z"}], [0, 1], evaluation_error=1e-12)
    assert profile.parameter_gradient("t")[0] == pytest.approx(-2 * sin(0.6), abs=1e-12)
    wrong_global_shift = (
        oracle(p.bind(t=0.3 + pi / 2), {0: "Z"}) - oracle(p.bind(t=0.3 - pi / 2), {0: "Z"})
    ) / 2
    assert abs(wrong_global_shift) < 1e-12
    partial = profile_sensitivity(p, [{0: "Z"}], [0], evaluation_error=1e-12)
    with pytest.raises(ValueError, match="every occurrence"):
        partial.parameter_gradient("t")


def test_cone_local_adjoint_and_weighted_parameter_gradient_match_shifts():
    p = (
        Program(3)
        .ry(0, Parameter("theta"))
        .cx(0, 1)
        .rz(1, Parameter("theta"))
        .ry(2, Parameter("phi"))
        .cx(1, 2)
        .bind(theta=0.31, phi=-0.27)
    )
    observables = ({0: "Z"}, {1: "X"}, {0: "Z"})
    weights = (0.4, -1.2, 0.3)
    shifts = profile_sensitivity(p, observables, [0, 2, 3], evaluation_error=1e-12)
    adjoint = profile_sensitivity(
        p, observables, [0, 2, 3], evaluation_error=1e-12, method="adjoint"
    )
    assert adjoint.shifted_requests == 0
    assert adjoint.methods == ("adjoint",) * len(observables)
    np.testing.assert_allclose(adjoint.jacobian, shifts.jacobian, atol=1e-12)
    expected = tuple(
        sum(weight * row[j] for weight, row in zip(weights, shifts.jacobian, strict=True))
        for j in range(len(shifts.gate_indices))
    )
    actual = adjoint_gradient(p, observables, weights)
    assert actual["theta"] == pytest.approx(expected[0] + expected[1], abs=1e-12)
    assert actual["phi"] == pytest.approx(expected[2], abs=1e-12)


def test_auto_selects_adjoint_per_cone_when_it_reduces_state_sweeps():
    p = Program(2).ry(0, 0.2).rz(0, -0.3).ry(1, 0.4)
    profile = profile_sensitivity(
        p, [{0: "Z"}, {1: "Z"}], [0, 1, 2], evaluation_error=1e-12, method="auto"
    )
    assert profile.methods == ("adjoint", "shift")
    reference = profile_sensitivity(p, [{0: "Z"}, {1: "Z"}], [0, 1, 2], evaluation_error=1e-12)
    np.testing.assert_allclose(profile.jacobian, reference.jacobian, atol=1e-12)
    with pytest.raises(ValueError, match="method"):
        profile_sensitivity(
            p, [{0: "Z"}], [0], evaluation_error=1e-12, method="unknown"
        )


def test_stationary_point_and_simultaneous_curvature():
    p = Program(1).ry(0, pi)
    profile = profile_sensitivity(p, [{0: "Z"}], [0], evaluation_error=1e-12)
    assert abs(profile.jacobian[0][0]) < 1e-12
    assert profile.bounds([-pi])[0] >= 2
    result = plan_approximate(profile, error_budget=0.01)
    assert result.zeroed_indices == ()
    assert result.plan.collect().values[0] == pytest.approx(-1)
    p = Program(1).ry(0, 0.0).ry(0, 0.0)
    profile = profile_sensitivity(p, [{0: "Z"}], [0, 1], evaluation_error=1e-12)
    assert 1 - cos(0.2) <= profile.bounds([0.1, 0.1])[0]


@pytest.mark.parametrize("seed", range(12))
def test_random_jacobian_and_composed_bounds_against_dense_oracle(seed):
    rng = np.random.default_rng(seed)
    gates = []
    for _ in range(3):
        for q in range(3):
            gates.append(Gate("R" + rng.choice(list("XYZ")), (q,), rng.uniform(-pi, pi)))
        gates.extend((Gate("CX", (0, 1)), Gate("CX", (1, 2))))
    p = Program(3, tuple(gates))
    indices = [i for i, g in enumerate(gates) if g.name.startswith("R")]
    obs = ({0: "Y", 2: "X"}, {1: "Z"}, {})
    profile = profile_sensitivity(p, obs, indices, evaluation_error=1e-11)
    adjoint = profile_sensitivity(
        p, obs, indices, evaluation_error=1e-11, method="adjoint"
    )
    np.testing.assert_allclose(adjoint.jacobian, profile.jacobian, atol=1e-12)
    reference = np.array([oracle(p, o) for o in obs])
    for j, i in enumerate(indices):
        shifted = []
        for d in (1e-5, -1e-5):
            variant = list(gates)
            variant[i] = replace(variant[i], angle=variant[i].angle + d)
            shifted.append([oracle(Program(3, tuple(variant)), o) for o in obs])
        fd = (np.array(shifted[0]) - shifted[1]) / 2e-5
        np.testing.assert_allclose(fd, np.array(profile.jacobian)[:, j], atol=1e-9)
    for scale in (1e-6, 0.001, 0.1, 1.0):
        deltas = rng.uniform(-scale, scale, len(indices))
        variant = list(gates)
        for i, d in zip(indices, deltas, strict=True):
            variant[i] = replace(variant[i], angle=variant[i].angle + d)
        actual = np.array([oracle(Program(3, tuple(variant)), o) for o in obs])
        assert np.all(np.abs(actual - reference) <= np.array(profile.bounds(deltas)) + 1e-13)


def test_zeroing_bridge_reduces_cone_and_preserves_exact_default():
    p = Program(2).h(0).h(1).cx(0, 1).rz(1, 0.001).cx(0, 1)
    profile = profile_sensitivity(p, [{0: "X"}], [3], evaluation_error=1e-12)
    result = plan_approximate(profile, error_budget=1e-5)
    assert result.zeroed_indices == (3,)
    assert result.exact_plan.selected.max_width == 2
    assert result.plan.selected.max_width == 1
    assert abs(result.plan.collect().values[0] - oracle(p, {0: "X"})) <= result.error_bounds[0]
    assert plan_all([p.expectation({0: "X"})]).selected.max_width == 2
    assert p.operations[3].angle == 0.001
    assert not result.search_truncated
    assert plan_approximate(profile, error_budget=0).zeroed_indices == ()


def test_identity_and_irrelevant_gates_need_no_evaluation():
    p = Program(2).ry(1, 0.4)
    profile = profile_sensitivity(p, [{0: "Z"}, {}], [0], evaluation_error=0)
    assert profile.shifted_requests == 0
    assert profile.bounds([1]) == (0, 0)


def test_reject_invalid_inputs_and_obey_search_budget():
    p = Program(1).ry(0, 0.01).rz(0, 0.02)
    for indices in ([0, 0], [-1], [True], [2]):
        with pytest.raises(ValueError):
            profile_sensitivity(p, [{0: "Z"}], indices, evaluation_error=0)
    for error in (-1, float("nan"), True):
        with pytest.raises(ValueError):
            profile_sensitivity(p, [{0: "Z"}], [0], evaluation_error=error)
    profile = profile_sensitivity(p, [{0: "Z"}], [0, 1], evaluation_error=1e-12)
    result = plan_approximate(profile, error_budget=0.01, max_candidates=1)
    assert result.candidates_checked == 1 and result.search_truncated
    assert result.zeroed_indices == ()
    with pytest.raises(ValueError):
        plan_approximate(profile, error_budget=[0.1, 0.2])
    with pytest.raises(ValueError):
        profile.bounds([0.1])
