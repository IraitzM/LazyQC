"""Native thread policy for Qulacs' OpenMP team.

Cone reduction produces many small states. For those, Qulacs' default team of one thread per
core is slower than a single thread (measured 2 to 3 times at 10 to 12 qubits), and the idle
team keeps competing with other OpenMP runtimes in the same process, such as Aer's. Wide
states still gain from threads, though not from a full team, so the policy keys on task width.

The limit has to be set before the first parallel region runs: workers that already exist keep
interfering, and lowering the count afterwards recovers only part of the speed. The executor
therefore decides once per ``collect`` call, from the widths of the tasks it is about to run.
"""

from contextlib import contextmanager, nullcontext

# Tasks narrower than this run on one thread (measured on the benchmark host: below 14 qubits a
# single thread is up to 3 times faster than the default team).
SMALL_TASK_QUBITS = 14
# Wide tasks gain from threads, but not from one per logical core: on a 20-thread host a team of
# 8 was up to 5.7 times faster than 20 (at 20 qubits). The cap only lowers a larger team.
WIDE_TEAM_CAP = 8

POLICIES = ("auto", "default")

_controller = None


def _qulacs_libraries():
    """Return the OpenMP runtimes Qulacs uses, or an empty list when they cannot be controlled."""
    global _controller
    try:
        import qulacs  # noqa: F401  (loads its bundled OpenMP runtime before the scan)
        from threadpoolctl import ThreadpoolController
    except ImportError:
        return []
    if _controller is None:
        _controller = ThreadpoolController()
    libraries = [lib for lib in _controller.lib_controllers if lib.user_api == "openmp"]
    # The wheels bundle their own runtime; a source build shares the system one.
    own = [lib for lib in libraries if "qulacs" in lib.filepath.lower()]
    return own or libraries


@contextmanager
def _limited(libraries, threads, *, only_lower=False):
    """Set the calling thread's OpenMP team size, then restore it.

    With `only_lower`, a team already at or below `threads` is left as it is.
    """
    previous = [lib.num_threads for lib in libraries]
    for lib, count in zip(libraries, previous, strict=True):
        lib.set_num_threads(min(threads, count) if only_lower else threads)
    try:
        yield
    finally:
        for lib, count in zip(libraries, previous, strict=True):
            lib.set_num_threads(count)


def native_thread_limit(policy, widths):
    """Return ``(context manager, label)`` for running tasks of the given widths on Qulacs.

    Args:
        policy: ``"auto"`` runs on one thread when every task is narrower than
            `SMALL_TASK_QUBITS`, and otherwise caps the team at `WIDE_TEAM_CAP` threads.
            ``"default"`` or ``None`` never changes the team. A positive integer fixes the
            team size.
        widths: Widths of the tasks that will run on the Qulacs backend.
    """
    if policy is None or policy == "default":
        return nullcontext(), "default"
    capped = False
    if policy == "auto":
        widths = tuple(widths)
        if not widths:
            return nullcontext(), "default"
        if max(widths) >= SMALL_TASK_QUBITS:
            threads, capped = WIDE_TEAM_CAP, True
            label = (
                f"capped at {WIDE_TEAM_CAP} (auto: a task has at least {SMALL_TASK_QUBITS} qubits)"
            )
        else:
            threads = 1
            label = f"1 (auto: every task is narrower than {SMALL_TASK_QUBITS} qubits)"
    elif type(policy) is int and policy >= 1:
        threads, label = policy, f"{policy} (requested)"
    else:
        raise ValueError(f"native_threads must be one of {POLICIES}, None or a positive integer")
    libraries = _qulacs_libraries()
    if not libraries:
        return nullcontext(), "default (OpenMP runtime not controllable)"
    if capped and all(lib.num_threads <= threads for lib in libraries):
        return nullcontext(), f"default (team already at most {WIDE_TEAM_CAP})"
    return _limited(libraries, threads, only_lower=capped), label
