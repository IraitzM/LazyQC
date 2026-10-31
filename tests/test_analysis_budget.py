"""Structure-keyed cone analysis and the auto analysis budget."""

import random

import pytest

from lazyqc import Gate, Pauli, PlanningCache, Program, plan_all
from lazyqc.passes import causal_cone
from lazyqc.plan import Candidate, Task
from lazyqc.planner import _analyze


def reference_analyze(requests, strategy):
    """The original per-request cone analysis, kept as an executable specification."""
    full_groups, cone_groups, separate, constants, resolved = {}, {}, [], [], {}
    for index, request in enumerate(requests):
        program = request.program
        if program not in resolved:
            resolved[program] = program.resolved_operations()
        operations = resolved[program]
        if not request.observable.factors:
            constants.append((index, 1.0))
            continue
        output = (index, request.observable)
        if strategy in ("full", "auto"):
            full_groups.setdefault((tuple(range(program.n_qubits)), operations), []).append(output)
        if strategy != "full":
            cone_key = causal_cone(operations, request.observable)
            if strategy in ("grouped", "auto"):
                cone_groups.setdefault(cone_key, []).append(output)
            if strategy in ("separate", "auto"):
                separate.append(Task(*cone_key, (output,)))
    candidates = []
    if strategy in ("full", "auto"):
        candidates.append(
            Candidate("full", tuple(Task(*k, tuple(v)) for k, v in full_groups.items()))
        )
    if strategy in ("separate", "auto"):
        candidates.append(Candidate("separate", tuple(separate)))
    if strategy in ("grouped", "auto"):
        candidates.append(
            Candidate("grouped", tuple(Task(*k, tuple(v)) for k, v in cone_groups.items()))
        )
    return candidates, constants


def random_batch(seed):
    """Mix shared topologies, equal programs as distinct objects, and identity outputs."""
    rng = random.Random(seed)
    n = rng.randint(2, 6)
    base = []
    for _ in range(rng.randint(1, 14)):
        if rng.random() < 0.35:
            base.append(Gate("CX", tuple(rng.sample(range(n), 2))))
        else:
            base.append(
                Gate(
                    rng.choice(("RX", "RY", "RZ")), (rng.randrange(n),), rng.choice((0.0, 0.3, 1.1))
                )
            )
    programs = [Program(n, tuple(base))]
    for _ in range(rng.randint(1, 4)):
        operations = list(base)
        i = rng.randrange(len(operations))
        if operations[i].name != "CX" and rng.random() < 0.7:
            operations[i] = Gate(
                operations[i].name, operations[i].qubits, operations[i].angle + 0.5
            )
        elif rng.random() < 0.5:
            operations.append(Gate("RY", (rng.randrange(n),), 0.3))
        programs.append(Program(n, tuple(operations)))
    programs.append(Program(n, tuple(base)))  # equal content, distinct object
    requests = []
    for program in programs:
        for _ in range(rng.randint(1, 5)):
            weight = rng.choice((0, 1, 1, 2))
            wires = rng.sample(range(n), weight)
            requests.append(
                program.expectation(Pauli(tuple((q, rng.choice("XYZ")) for q in wires)))
            )
    rng.shuffle(requests)
    return tuple(requests)


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("strategy", ["full", "separate", "grouped", "auto"])
def test_structure_keyed_analysis_matches_specification(seed, strategy):
    requests = random_batch(seed)
    expected, constants = reference_analyze(requests, strategy)
    actual = _analyze(requests, strategy)
    assert [(c.strategy, c.tasks) for c in actual.candidates] == [
        (c.strategy, c.tasks) for c in expected
    ]
    assert list(actual.constants) == constants


def dense_heavy_requests():
    """Many outputs on a narrow, gate-heavy circuit of two independent pairs."""
    rng = random.Random(7)
    operations = []
    for _ in range(200):
        pair = rng.choice(((0, 1), (2, 3)))
        if rng.random() < 0.3:
            operations.append(Gate("CX", pair))
        else:
            operations.append(Gate("RY", (rng.choice(pair),), rng.uniform(-1, 1)))
    program = Program(4, tuple(operations))
    return [program.expectation({q % 4: "XYZ"[q % 3]}) for q in range(60)]


def test_budget_keeps_dense_plan_when_analysis_cannot_repay():
    requests = dense_heavy_requests()
    plan = plan_all(requests)
    budget = plan.explain()["analysis_budget"]
    assert budget["skipped"] is True
    assert budget["predicted_analysis_ns"] >= budget["predicted_dense_ns"]
    assert plan.selected.strategy == "full"
    assert [c.strategy for c in plan.alternatives] == ["full"]
    assert plan.explain()["selection_reason"].startswith("analysis skipped")
    exhaustive = plan_all(requests, analysis="always")
    assert exhaustive.explain()["analysis_budget"] is None
    assert len(exhaustive.alternatives) == 3
    assert plan.collect().values == pytest.approx(exhaustive.collect().values, abs=1e-12)


def test_budget_never_blocks_required_or_cached_analysis():
    requests = dense_heavy_requests()
    # A dense plan outside the limits must not be kept; cone plans are required.
    limited = plan_all(requests, max_state_qubits=2)
    assert limited.explain()["analysis_budget"] is None
    assert limited.selected.strategy in ("separate", "grouped")
    # A cache signals reuse, so the analysis is performed and stored.
    assert plan_all(requests, cache=PlanningCache()).explain()["analysis_budget"] is None
    # Wide circuits keep the budget visible but analyse.
    wide = Program(16).ry(0, 0.2).cx(0, 1).ry(5, 0.4)
    plan = plan_all([wide.expectation({0: "Z"}), wide.expectation({5: "Z"})])
    assert plan.explain()["analysis_budget"]["skipped"] is False
    assert plan.selected.strategy in ("separate", "grouped")


def test_unknown_analysis_mode_is_rejected():
    with pytest.raises(ValueError, match="analysis"):
        plan_all(dense_heavy_requests(), analysis="never")
