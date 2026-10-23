"""Optional process isolation for hard planning deadlines and address-space caps."""

import math
import multiprocessing
import pickle
import resource
import tempfile
from pathlib import Path
from time import perf_counter


class PlanningResourceError(RuntimeError):
    pass


def validate_address_space_limit(limit):
    """Validate an optional positive address-space limit in bytes."""
    if limit is not None:
        if type(limit) is not int or limit < 1:
            raise ValueError("max_address_space_bytes must be a positive integer or None")
        if not hasattr(resource, "RLIMIT_AS"):
            raise ValueError("Address-space limits are unsupported on this platform")


def set_address_space_limit(limit):
    """Apply an address-space limit to the current worker when supported."""
    validate_address_space_limit(limit)
    if limit is not None:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _plan_worker(requests, options, limit, output):
    """Plan requests in an isolated worker and send either the result or exception."""
    from .planner import plan_all

    try:
        set_address_space_limit(limit)
        result = (True, plan_all(requests, **options))
    except Exception as exc:
        result = (False, exc)
    with open(output, "wb") as stream:
        pickle.dump(result, stream, protocol=5)


def plan_isolated(requests, *, timeout_s=10.0, max_address_space_bytes=None, **options):
    """Run planning in a bounded subprocess and return its plan.

    Plan in a disposable process. Deadline includes spawn and plan serialization.

    No shared PlanningCache; no backend states are allocated. The memory cap is
    virtual address space, not RSS. The caller's limits remain unchanged.
    """
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout_s must be finite and positive")
    validate_address_space_limit(max_address_space_bytes)
    if options.get("cache") is not None:
        raise ValueError("Isolated planning cannot share a PlanningCache")
    start = perf_counter()
    requests = tuple(requests)
    with tempfile.TemporaryDirectory(prefix="lazyqc-plan-") as directory:
        output = str(Path(directory) / "result.pickle")
        process = multiprocessing.get_context("spawn").Process(
            target=_plan_worker, args=(requests, options, max_address_space_bytes, output)
        )
        process.start()
        try:
            process.join(max(0, timeout_s - (perf_counter() - start)))
            if process.is_alive():
                raise TimeoutError("Planning exceeded the isolated wall-time limit")
            if process.exitcode != 0 or not Path(output).exists():
                raise PlanningResourceError(
                    f"Planning process exited without a result (exitcode={process.exitcode}); "
                    "the address-space limit may have been exceeded"
                )
            try:
                with open(output, "rb") as stream:
                    success, result = pickle.load(stream)
            except (EOFError, pickle.UnpicklingError) as exc:
                raise PlanningResourceError(
                    "Planning process could not serialize its result"
                ) from exc
            if not success:
                raise result
            return result
        finally:
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
