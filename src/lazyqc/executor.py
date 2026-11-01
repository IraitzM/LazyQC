"""Serial execution. No numerical results survive a collect call."""

from dataclasses import dataclass, replace
from time import perf_counter_ns

from .backends.qulacs import QulacsBackend
from .instruments import Projector
from .ir import finite_angle
from .threads import native_thread_limit


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
    native_threads: str = "default"


@dataclass(frozen=True)
class Result:
    values: tuple[float, ...]
    stats: ExecutionStats
    selected_strategy: str
    selected_optimization: str = "none"


def _check_backend(backend, role):
    """Accept any object implementing the backend protocol used below."""
    if not hasattr(backend, "counters") or not all(
        callable(getattr(backend, method, None)) for method in ("lower", "evolve", "expectations")
    ):
        raise ValueError(f"{role} must provide lower, evolve, expectations and counters")


def execute(
    plan,
    *,
    backend=None,
    wide_backend=None,
    wide_from=None,
    mps_backend=None,
    native_threads="auto",
    bindings=None,
) -> Result:
    """Execute a serial plan and return values together with timing statistics.

    Any backend implementing lower/evolve/expectations can run the plan; Qulacs
    is the default and the only one supporting projector (cut) tasks. With
    `wide_backend`, unitary tasks of at least `wide_from` qubits run there (for
    example a GPU statevector) and narrower tasks stay on `backend`. Tensor-network
    tasks (exact MPS fallback) run on `mps_backend`, quimb by default.

    `native_threads` sets Qulacs' OpenMP team for the call: ``"auto"`` (default) uses one thread
    when every task it runs is narrower than `lazyqc.threads.SMALL_TASK_QUBITS`, ``"default"``
    or ``None`` leaves the team alone, and a positive integer fixes its size.

    `bindings` maps parameter names to angles for plans built from unbound `Parameter`s.
    """
    if plan.cost_model is not None:
        plan.cost_model.validate_environment()
    values_by_name = {k: finite_angle(v) for k, v in (bindings or {}).items()}
    if missing := [n for n in plan.parameters if n not in values_by_name]:
        raise ValueError(f"Unbound parameters: {', '.join(missing)}")
    backend = backend if backend is not None else QulacsBackend()
    _check_backend(backend, "backend")
    if wide_backend is not None:
        _check_backend(wide_backend, "wide_backend")
    if (wide_backend is None) != (wide_from is None):
        raise ValueError("wide_backend and wide_from must be given together")
    if wide_from is not None and (type(wide_from) is not int or wide_from < 1):
        raise ValueError("wide_from must be a positive integer")
    if mps_backend is None and any(t.bond for t in plan.selected.tasks):
        from .backends.tensor import QuimbMPSBackend

        mps_backend = QuimbMPSBackend()
    if mps_backend is not None:
        _check_backend(mps_backend, "mps_backend")
    backends = (backend,) if wide_backend is None else (backend, wide_backend)
    if mps_backend is not None:
        backends += (mps_backend,)
    start = perf_counter_ns()
    before = [replace(b.counters) for b in backends]
    lowering_ns = evolution_ns = expectation_ns = assembly_ns = 0
    mark = perf_counter_ns()
    values = [None] * plan.request_count
    for index, value in plan.constant_outputs:
        values[index] = value
    assembly_ns += perf_counter_ns() - mark

    def route(task):
        """Return the backend that runs `task`."""
        if task.bond:
            return mps_backend
        if wide_from is not None and task.width >= wide_from:
            if not any(isinstance(g, Projector) for g in task.operations):
                return wide_backend
        return backend

    if isinstance(backend, QulacsBackend):
        widths = [t.width for t in plan.selected.tasks if route(t) is backend]
        limit, threads_label = native_thread_limit(native_threads, widths)
    else:
        limit, threads_label = native_thread_limit(None, ())
    with limit:
        for task in plan.selected.tasks:
            # Recheck before native allocation even if a caller manually assembled a Plan.
            if task.state_bytes > plan.max_state_bytes or (
                not task.bond and task.width > plan.max_state_qubits
            ):
                raise ValueError("Task exceeds the plan state limits")
            target = route(task)
            mark = perf_counter_ns()
            if plan.parameters and hasattr(target, "rebind"):
                # Symbolic plans lower once; later collections only set the new angles.
                key = (type(target), id(task))
                entry = plan.lowered.get(key)
                if entry is None or entry[0] is not task:
                    entry = plan.lowered[key] = (
                        task,
                        target.lower(task, parametric=bool(task.symbolic)),
                    )
                lowered = entry[1]
                if lowered.slots:
                    target.rebind(lowered, values_by_name)
            else:
                lowered = target.lower(task.bound(values_by_name))
            lowering_ns += perf_counter_ns() - mark
            mark = perf_counter_ns()
            state = target.evolve(lowered)
            evolution_ns += perf_counter_ns() - mark
            mark = perf_counter_ns()
            outputs = target.expectations(state, lowered.observables)
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
    counts = [
        sum(
            getattr(b.counters, field) - getattr(prior, field) for b, prior in zip(backends, before)
        )
        for field in ("allocations", "evolutions", "expectations")
    ]
    stats = ExecutionStats(
        *counts,
        plan.selected.max_width,
        plan.selected.state_bytes,
        lowering_ns,
        evolution_ns,
        expectation_ns,
        assembly_ns,
        perf_counter_ns() - start,
        threads_label,
    )
    return Result(values, stats, plan.selected.strategy, plan.selected.optimization)
