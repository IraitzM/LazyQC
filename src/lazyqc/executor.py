"""Serial execution. No numerical results survive a collect call."""

from dataclasses import dataclass, replace
from time import perf_counter_ns

from .backends.qulacs import QulacsBackend


@dataclass(frozen=True)
class ExecutionStats:
    allocations: int
    evolutions: int
    expectations: int
    max_state_qubits: int
    estimated_state_bytes: int
    lowering_ns: int
    evolution_ns: int
    expectation_ns: int
    assembly_ns: int
    execution_ns: int


@dataclass(frozen=True)
class Result:
    values: tuple[float, ...]
    stats: ExecutionStats
    selected_strategy: str
    selected_optimization: str = "none"


def execute(plan, *, backend=None) -> Result:
    """Execute a serial plan and return values together with timing statistics."""
    if plan.cost_model is not None:
        plan.cost_model.validate_environment()
    backend = backend if backend is not None else QulacsBackend()
    if not isinstance(backend, QulacsBackend):
        raise ValueError("This release supports QulacsBackend only")
    start = perf_counter_ns()
    before = replace(backend.counters)
    lowering_ns = evolution_ns = expectation_ns = assembly_ns = 0
    mark = perf_counter_ns()
    values = [None] * plan.request_count
    for index, value in plan.constant_outputs:
        values[index] = value
    assembly_ns += perf_counter_ns() - mark
    for task in plan.selected.tasks:
        # Recheck before native allocation even if a caller manually assembled a Plan.
        if task.width > plan.max_state_qubits or task.state_bytes > plan.max_state_bytes:
            raise ValueError("Task exceeds the plan state limits")
        mark = perf_counter_ns()
        lowered = backend.lower(task)
        lowering_ns += perf_counter_ns() - mark
        mark = perf_counter_ns()
        state = backend.evolve(lowered)
        evolution_ns += perf_counter_ns() - mark
        mark = perf_counter_ns()
        outputs = backend.expectations(state, lowered.observables)
        expectation_ns += perf_counter_ns() - mark
        mark = perf_counter_ns()
        for (index, _), value in zip(task.outputs, outputs, strict=True):
            values[index] = value
        # Release before allocating the next state; one state payload live at a time.
        del state, lowered
        assembly_ns += perf_counter_ns() - mark
    mark = perf_counter_ns()
    if any(value is None for value in values):
        raise RuntimeError("Plan did not produce every requested result")
    values = tuple(values)
    assembly_ns += perf_counter_ns() - mark
    after = backend.counters
    stats = ExecutionStats(
        after.allocations - before.allocations,
        after.evolutions - before.evolutions,
        after.expectations - before.expectations,
        plan.selected.max_width,
        plan.selected.state_bytes,
        lowering_ns,
        evolution_ns,
        expectation_ns,
        assembly_ns,
        perf_counter_ns() - start,
    )
    return Result(values, stats, plan.selected.strategy, plan.selected.optimization)
