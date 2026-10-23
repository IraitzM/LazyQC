"""Bounded exact experiment execution and signed reconstruction; no cross-collect values."""

import math
import multiprocessing
import os
import resource
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from time import perf_counter_ns

from .backends.qulacs import QulacsBackend
from .batch import task_durations
from .cost import THREAD_KEYS


@dataclass(frozen=True)
class BatchStats:
    allocations: int
    evolutions: int
    expectations: int
    max_state_qubits: int
    estimated_state_bytes: int
    peak_active_state_bytes: int
    peak_in_flight: int
    worker_processes: int
    worker_peak_rss_bytes: int
    lowering_ns: int
    evolution_ns: int
    expectation_ns: int
    reconstruction_ns: int
    scheduling_ns: int
    execution_ns: int
    execution_mode: str = "serial"
    dispatch_reason: str = "explicit workers"
    pool_startup_ns: int = 0
    pool_reused: bool = False
    worker_pids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BatchResult:
    values: tuple[float, ...]
    stats: BatchStats
    selected_strategy: str
    selected_optimization: str


def _one(task, backend=None):
    """Execute one task and return values, timings, process ID, and peak RSS."""
    backend = backend or QulacsBackend()
    start = perf_counter_ns()
    lowered = backend.lower(task)
    built = perf_counter_ns()
    state = backend.evolve(lowered)
    evolved = perf_counter_ns()
    values = backend.expectations(state, lowered.observables)
    finished = perf_counter_ns()
    del state, lowered
    return (
        values,
        built - start,
        evolved - built,
        finished - evolved,
        os.getpid(),
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
    )


def _initialize_worker(barrier=None, max_address_space_bytes=None):
    """Configure a spawned worker for one native thread and optional memory limit."""
    # Spawn avoids inheriting initialized native OpenMP pools. Explicitly one thread
    # per worker; a calibrated batch profile must use the same configuration.
    for key in THREAD_KEYS:
        os.environ[key] = "1"
    if max_address_space_bytes is not None:
        from .isolation import set_address_space_limit

        set_address_space_limit(max_address_space_bytes)
    if barrier is not None:
        import qulacs  # noqa: F401

        barrier.wait(timeout=30)


def execute_batch(plan, *, backend=None, session=None):
    """Execute a batch plan serially or with its bounded worker pool."""
    if session is not None:
        from .session import ExecutionSession

        if not isinstance(session, ExecutionSession):
            raise ValueError("session must be ExecutionSession")
        with session._claim():
            return _execute_batch(plan, backend=backend, session=session)
    return _execute_batch(plan, backend=backend)


def _execute_batch(plan, *, backend=None, session=None):
    """Run experiments, schedule state payloads, and reconstruct request results."""
    if plan.cost_model is not None:
        plan.cost_model.validate_environment()
    if backend is not None and (not isinstance(backend, QulacsBackend) or plan.workers != 1):
        raise ValueError("A supplied Qulacs backend is supported only with one worker")
    start = perf_counter_ns()
    tasks = plan.selected.tasks
    workers, reason = session._choose(plan) if session else (plan.workers, "explicit workers")
    startup_ns = 0
    pool_reused = bool(session and session._pool is not None)
    for task in tasks:
        if task.width > plan.max_state_qubits or task.state_bytes > min(
            plan.max_state_bytes, plan.total_state_bytes
        ):
            raise ValueError("Experiment exceeds executable state limits")
    values = [None] * len(tasks)
    lower = evolve = expectations = 0
    peak_live = peak_in_flight = peak_rss = 0
    processes = set()

    def record(i, row):
        """Accumulate one completed experiment's values and worker statistics."""
        nonlocal lower, evolve, expectations, peak_rss
        values[i], lo, ev, ex, pid, rss = row
        lower += lo
        evolve += ev
        expectations += ex
        processes.add(pid)
        peak_rss = max(peak_rss, rss)

    if workers == 1:
        for i, task in enumerate(tasks):
            record(i, _one(task, backend))
            peak_live = max(peak_live, task.state_bytes)
        peak_in_flight = int(bool(tasks))
    elif tasks:
        durations = task_durations(tasks, plan.cost_model)
        pending = sorted(range(len(tasks)), key=lambda i: (-durations[i], i))
        # Only pure dataclass descriptions cross the process boundary, never states.
        if session:
            mark = perf_counter_ns()
            session._ensure_pool()
            startup_ns = 0 if pool_reused else perf_counter_ns() - mark
        context = (
            nullcontext(session._pool)
            if session
            else ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
            )
        )
        with context as pool:
            active, live = {}, 0
            while pending or active:
                for i in list(pending):
                    if len(active) >= workers:
                        break
                    if live + tasks[i].state_bytes <= plan.total_state_bytes:
                        pending.remove(i)
                        active[pool.submit(_one, tasks[i])] = i
                        live += tasks[i].state_bytes
                        peak_live = max(peak_live, live)
                        peak_in_flight = max(peak_in_flight, len(active))
                if not active:
                    raise ValueError("Scheduler cannot fit a pending task")
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in completed:
                    i = active.pop(future)
                    live -= tasks[i].state_bytes
                    record(i, future.result())  # Fail loudly; never return partial results.
    experiments_done = perf_counter_ns()
    result = [None] * plan.request_count
    for i, value in plan.constant_outputs:
        result[i] = value
    for r in plan.selected.reconstruction:
        result[r.output] = math.fsum(
            term.coefficient * math.prod(values[i][slot] for i, slot in term.factors)
            for term in r.terms
        )
    if any(v is None or not math.isfinite(v) or abs(v) > 1 + 1e-8 for v in result):
        raise ArithmeticError(
            "Reconstruction produced missing, nonfinite or out-of-range expectations"
        )
    finish = perf_counter_ns()
    # Parallel phase sums are worker CPU/wall sums, not a decomposition of parent wall time.
    overhead = (
        max(0, experiments_done - start - (lower + evolve + expectations))
        if workers == 1
        else experiments_done - start
    )
    stats = BatchStats(
        len(tasks),
        len(tasks),
        sum(len(t.outputs) for t in tasks),
        plan.selected.max_width,
        plan.selected.state_bytes,
        peak_live,
        peak_in_flight,
        len(processes),
        peak_rss,
        lower,
        evolve,
        expectations,
        finish - experiments_done,
        overhead,
        finish - start,
        "parallel" if workers > 1 and tasks else "serial",
        reason,
        startup_ns,
        pool_reused,
        tuple(sorted(processes)),
    )
    return BatchResult(tuple(result), stats, plan.selected.strategy, plan.selected.optimization)
