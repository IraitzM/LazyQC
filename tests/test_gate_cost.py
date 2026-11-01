"""Per-gate lowering cost in the work score (issue #2)."""

import pytest

import lazyqc.plan as plan_module
from lazyqc import Program, plan_all
from lazyqc.plan import GATE_COST, Task


def brick(n, layers):
    """Single-qubit rotations and a staggered CX layer: cones widen with the layer count."""
    program = Program(n)
    for layer in range(layers):
        for q in range(n):
            program = program.ry(q, 0.1 * (q + 1) + layer)
        for q in range(layer % 2, n - 1, 2):
            program = program.cx(q, q + 1)
    return program


def requests(n, layers):
    program = brick(n, layers)
    return [program.expectation({q: "Z"}) for q in range(n)]


def test_score_adds_the_gate_cost_to_amplitude_work():
    program = brick(4, 2)
    task = plan_all(requests(4, 2), strategy="full").selected.tasks[0]
    amplitude_work = (1 << task.width) * (1 + len(task.operations) + len(task.outputs))
    assert isinstance(task, Task)
    assert task.score == pytest.approx(amplitude_work + GATE_COST * len(task.operations))
    assert GATE_COST > 0 and program.operations


def test_tiny_circuit_runs_dense_instead_of_many_cone_tasks(monkeypatch):
    reqs = requests(6, 2)
    monkeypatch.setattr(plan_module, "GATE_COST", 0.0)
    assert plan_all(reqs, analysis="always").selected.strategy == "grouped"
    monkeypatch.setattr(plan_module, "GATE_COST", GATE_COST)
    chosen = plan_all(reqs, analysis="always")
    assert chosen.selected.strategy == "full"
    assert len(chosen.selected.tasks) == 1


def test_narrow_cones_still_reduce():
    chosen = plan_all(requests(16, 2), analysis="always")
    assert chosen.selected.strategy == "grouped"
    assert chosen.selected.max_width < 16


@pytest.mark.parametrize("n,layers", [(6, 2), (8, 3), (16, 2)])
def test_either_choice_gives_the_same_values(n, layers, monkeypatch):
    reqs = requests(n, layers)
    values = {}
    for cost in (0.0, GATE_COST):
        monkeypatch.setattr(plan_module, "GATE_COST", cost)
        values[cost] = plan_all(reqs, analysis="always").collect().values
    assert values[0.0] == pytest.approx(values[GATE_COST], abs=1e-12)
