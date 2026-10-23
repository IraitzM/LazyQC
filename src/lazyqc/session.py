"""Explicit reusable worker ownership and measured serial/parallel dispatch decisions."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from threading import Lock
from time import perf_counter_ns

from .batch import schedule_estimate, task_durations
from .cost import environment


def _ping(payload):
    """Return a worker heartbeat payload for pool warmup."""
    return len(payload)


@dataclass(frozen=True)
class TransportProfile:
    startup_ns: int
    roundtrip_ns: int
    probe_bytes: int
    workers: int
    environment_items: tuple


class ExecutionSession:
    """Own a local process pool; never cache states, circuits or numerical results.

    `auto` requires explicit warmup and a BatchCostModel before choosing parallel
    work. Without both, it executes serially instead of paying unmeasured startup.
    `parallel` forces worker use. One collection can use a session at a time.
    """

    def __init__(self, workers=2, *, policy="auto", max_address_space_bytes=None):
        """Create a reusable execution session with worker and memory policies."""
        from .isolation import validate_address_space_limit

        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError("workers must be between 1 and 32")
        if policy not in ("auto", "parallel"):
            raise ValueError("policy must be auto or parallel")
        validate_address_space_limit(max_address_space_bytes)
        self.workers = workers
        self.policy = policy
        self.max_address_space_bytes = max_address_space_bytes
        self._pool = None
        self._closed = False
        self._lock = Lock()
        self.profile = None

    @contextmanager
    def _claim(self):
        """Return the session lock context used to serialize session ownership."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("ExecutionSession already has an active operation")
        try:
            if self._closed:
                raise RuntimeError("ExecutionSession is closed")
            yield
        except BaseException:
            self._close_unlocked()
            raise
        finally:
            self._lock.release()

    def _ensure_pool(self):
        """Start the configured worker pool if it has not already been started."""
        from .batch_executor import _initialize_worker

        if self._pool is not None:
            return
        start = perf_counter_ns()
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(self.workers)
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=ctx,
            initializer=_initialize_worker,
            initargs=(barrier, self.max_address_space_bytes),
        )
        futures = [self._pool.submit(_ping, b"") for _ in range(self.workers)]
        for future in futures:
            future.result(timeout=40)
        startup = perf_counter_ns() - start
        payload = bytes(16384)
        samples = []
        for _ in range(7):
            mark = perf_counter_ns()
            self._pool.submit(_ping, payload).result(timeout=10)
            samples.append(perf_counter_ns() - mark)
        self.profile = TransportProfile(
            startup,
            int(median(samples)),
            len(payload),
            self.workers,
            tuple(sorted(environment().items())),
        )

    def warmup(self):
        """Start workers and import native dependencies before the first batch.

        Explicitly pay startup and measure IPC, without allocating quantum states.
        """
        with self._claim():
            self._ensure_pool()
            return self.profile

    def _choose(self, plan):
        """Choose the effective worker count and explain the policy decision."""
        if plan.workers != self.workers:
            raise ValueError("Plan and ExecutionSession worker counts must match")
        if self.profile and self.profile.environment_items != tuple(sorted(environment().items())):
            raise ValueError("Worker transport environment mismatch; create a new session")
        if self.policy == "parallel":
            return self.workers, "explicit persistent workers"
        if plan.cost_model is None or self.profile is None:
            return 1, "serial: task/transport calibration unavailable; startup avoided"
        durations = task_durations(plan.selected.tasks, plan.cost_model)
        parallel, _ = schedule_estimate(
            plan.selected.tasks, self.workers, plan.total_state_bytes, durations
        )
        # Conservative parent-side dispatch allowance. Scale large task messages
        # by their serialized size relative to the calibration probe.
        import pickle

        transfers = sum(
            max(1, len(pickle.dumps(t, protocol=5)) / self.profile.probe_bytes)
            for t in plan.selected.tasks
        )
        parallel += transfers * self.profile.roundtrip_ns
        if self._pool is None:
            parallel += self.profile.startup_ns
        if parallel < sum(durations):
            return self.workers, "parallel: estimated native work repays measured dispatch"
        return 1, "serial: measured dispatch exceeds predicted parallel saving"

    def _close_unlocked(self):
        """Close the worker pool while the caller holds the session lock."""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
        self._closed = True

    def close(self):
        """Close workers and release all session resources."""
        with self._lock:
            self._close_unlocked()

    def __enter__(self):
        """Warm up and return this session for use as a context manager."""
        if self._closed:
            raise RuntimeError("ExecutionSession is closed")
        return self

    def __exit__(self, *exc):
        """Close the session when leaving a context-manager block."""
        self.close()
