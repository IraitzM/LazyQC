"""Deterministic planner policy tests; wall-clock speed is never a test assertion."""

import json
from dataclasses import replace

import pytest

from lazyqc import CalibratedCostModel, Program, plan_all
from lazyqc.calibration import calibrate
from lazyqc.cost import FEATURES, environment, features, fit_cost_model


def profile(**weights):
    return CalibratedCostModel(
        tuple(weights.get(name, 0) for name in FEATURES), tuple(environment().items()), 1, 17, 0
    )


def requests():
    p = Program(4).ry(0, 0.3).cx(0, 1).ry(2, 0.7).cx(2, 3)
    return [p.expectation({q: axis}) for q in range(4) for axis in ("X", "Z")]


def test_model_changes_choice_without_backend(monkeypatch):
    from lazyqc.backends.qulacs import QulacsBackend

    def forbidden(*args):
        raise AssertionError("Planning invoked simulator")

    monkeypatch.setattr(QulacsBackend, "lower", forbidden)
    monkeypatch.setattr(QulacsBackend, "evolve", forbidden)
    assert plan_all(requests()).selected.strategy == "grouped"
    plan = plan_all(requests(), cost_model=profile(tasks=1000))
    assert plan.selected.strategy == "full"
    assert plan.explain()["cost_model_status"] == "calibrated"
    assert plan.selected.predicted_execution_ns == 1000
    assert (
        plan_all(requests(), cost_model=profile(state_amplitudes=1)).selected.strategy == "grouped"
    )


def test_profile_roundtrip_and_validation(tmp_path, monkeypatch):
    model = profile(tasks=12, mixed_expectation_work=0.1)
    path = tmp_path / "model.json"
    model.save(path)
    assert CalibratedCostModel.load(path) == model
    assert CalibratedCostModel.load(path).profile_id == model.profile_id
    plan = plan_all(requests(), cost_model=model)
    monkeypatch.setenv("QULACS_NUM_THREADS", "99")
    with pytest.raises(ValueError, match="environment mismatch"):
        CalibratedCostModel.load(path)
    with pytest.raises(ValueError, match="environment mismatch"):
        plan_all(requests(), cost_model=model)
    with pytest.raises(ValueError, match="environment mismatch"):
        plan.collect()


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_invalid_coefficients(value):
    with pytest.raises(ValueError, match="coefficients"):
        profile(tasks=value)


def test_schema_and_domain(tmp_path):
    path = tmp_path / "model.json"
    profile().save(path)
    data = json.loads(path.read_text())
    data["features"][0] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="feature schema"):
        CalibratedCostModel.load(path)
    model = replace(profile(tasks=1), max_width=2)
    plan = plan_all(requests(), cost_model=model)
    assert "fallback" in plan.cost_model_status
    assert all(c.predicted_execution_ns is None for c in plan.alternatives)
    # Rejected candidates cannot trigger fallback or win on predicted speed.
    limited = plan_all(requests(), max_state_qubits=2, cost_model=model)
    assert limited.cost_model_status == "calibrated"
    assert limited.selected.strategy == "grouped"
    assert limited.alternatives[0].rejection is not None
    assert limited.alternatives[0].predicted_execution_ns is None


def test_empty_and_forced_plans():
    model = profile(collection=5, tasks=10)
    assert plan_all([], cost_model=model).collect().values == ()
    p = Program(1)
    assert plan_all([p.expectation({})], cost_model=model).collect().values == (1.0,)
    plan = plan_all(requests(), strategy="separate", cost_model=model)
    assert len(plan.alternatives) == 1
    assert plan.selected.strategy == "separate"


def test_fitted_predictions_on_synthetic_targets():
    candidates = plan_all(requests()).alternatives
    model = profile(collection=1000, tasks=7000, one_qubit_work=3)
    targets = [model.predict(c) for c in candidates]
    fitted = fit_cost_model(candidates, targets)
    assert [fitted.predict(c) for c in candidates] == pytest.approx(targets, rel=1e-6)
    with pytest.raises(ValueError, match="positive"):
        fit_cost_model(candidates, [0, 1, 2])
    assert features(candidates[0])[FEATURES.index("tasks")] == 1


def test_small_calibration():
    model, report = calibrate(widths=(1, 2), repeats=1)
    assert model.calibration_ns > 0
    assert len(report["runs"]) == len(report["candidates"])
    for row, candidate in zip(report["first_calls"], report["candidates"], strict=True):
        assert row["allocations"] == len(candidate["tasks"])
    result = plan_all(requests(), cost_model=model, max_state_qubits=2).collect()
    assert result.values == pytest.approx(plan_all(requests(), strategy="full").collect().values)
