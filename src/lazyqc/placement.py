"""Find the task width from which a wide backend (a GPU) beats the default backend on this host.

``plan.collect(wide_backend=..., wide_from=w)`` sends every task of at least `w` qubits to the
wide backend. The right `w` depends on the device, the driver and the CPU, so it is measured
here instead of being assumed.
"""

import random
from time import perf_counter_ns

from .backends.qulacs import QulacsBackend
from .ir import Gate, Pauli
from .plan import Task
from .threads import native_thread_limit


def _representative_task(width, gates, rng):
    """Return one task of `width` qubits and `gates` gates with a single Z output."""
    operations = []
    for _ in range(gates):
        if width > 1 and rng.random() < 0.35:
            operations.append(Gate("CX", tuple(rng.sample(range(width), 2))))
        else:
            operations.append(
                Gate(rng.choice(("RX", "RY", "RZ")), (rng.randrange(width),), rng.uniform(-3, 3))
            )
    return Task(tuple(range(width)), tuple(operations), ((0, Pauli(((0, "Z"),))),))


def _best_time(backend, task, repeats, clock, limit):
    """Return the fastest of `repeats` timed runs after one warm-up run, in clock units."""
    best = None
    with limit:
        for attempt in range(repeats + 1):
            start = clock()
            lowered = backend.lower(task)
            state = backend.evolve(lowered)
            backend.expectations(state, lowered.observables)
            elapsed = clock() - start
            if attempt and (best is None or elapsed < best):
                best = elapsed
    return best


def calibrate_crossover(
    wide_backend,
    *,
    backend=None,
    widths=range(6, 23),
    gates=40,
    repeats=3,
    confirm=2,
    seed=1729,
    native_threads="auto",
    clock=perf_counter_ns,
):
    """Measure the smallest width from which `wide_backend` is faster than `backend`.

    Runs a representative task (random rotations and CX, one ``Z`` output) of each width on
    both backends and stops once the wide backend has been at least as fast for `confirm`
    consecutive widths, so wide CPU runs are avoided. The result depends on how many gates a
    task has, because a GPU pays a launch cost per gate: set `gates` near your tasks.

    Args:
        wide_backend: Backend for wide tasks, for example an Aer or Lightning GPU backend.
        backend: The narrow-task backend; `QulacsBackend()` by default.
        widths: Candidate widths in increasing order.
        gates: Gates per representative task.
        repeats: Timed runs per width and backend, after one warm-up run.
        confirm: Consecutive wins that settle the crossover.
        seed: Seed of the representative circuits.
        native_threads: Qulacs thread policy for the narrow backend, as in ``collect``.
        clock: Nanosecond clock; replaceable for deterministic tests.

    Returns:
        ``(width, report)``. `width` is the first width of the winning run, ready to pass as
        ``wide_from``, or ``None`` when the wide backend never won in `widths`. `report` lists
        ``(width, narrow_ns, wide_ns)`` for every width measured.
    """
    widths = tuple(widths)
    if (
        not widths
        or any(type(w) is not int or w < 1 for w in widths)
        or list(widths) != sorted(set(widths))
    ):
        raise ValueError("widths must be distinct positive integers in increasing order")
    for name, value in (("gates", gates), ("repeats", repeats), ("confirm", confirm)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    backend = backend if backend is not None else QulacsBackend()
    for role, b in (("backend", backend), ("wide_backend", wide_backend)):
        if not all(callable(getattr(b, m, None)) for m in ("lower", "evolve", "expectations")):
            raise ValueError(f"{role} must provide lower, evolve and expectations")
    rng = random.Random(seed)
    report, run = [], []
    for width in widths:
        task = _representative_task(width, gates, rng)
        if isinstance(backend, QulacsBackend):
            limit, _ = native_thread_limit(native_threads, (width,))
        else:
            limit, _ = native_thread_limit(None, ())
        narrow = _best_time(backend, task, repeats, clock, limit)
        wide = _best_time(wide_backend, task, repeats, clock, native_thread_limit(None, ())[0])
        report.append((width, narrow, wide))
        run = [*run, width] if wide <= narrow else []
        if len(run) >= confirm:
            return run[0], tuple(report)
    return None, tuple(report)
