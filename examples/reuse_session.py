"""Explicit calibration and persistent workers, with fresh states on every collect."""

import os


def main():
    for key in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    from lazyqc import (
        CutOptions,
        ExecutionSession,
        PlanningCache,
        Program,
        calibrate_batch,
        plan_all,
    )

    model, _ = calibrate_batch(widths=(1, 2, 4), repeats=3)
    p = Program(2).h(0).cx(0, 1)
    requests = [p.expectation({0: a, 1: a}) for a in "XYZ"]
    cache = PlanningCache(8, max_bytes=2**20)
    plan = plan_all(
        requests,
        workers=2,
        cost_model=model,
        cache=cache,
        cutting=CutOptions(max_fragment_qubits=1, expected_collections=16),
    )
    with ExecutionSession(2) as session:
        print("Explicit pool setup:", session.warmup())
        for _ in range(2):
            result = plan.collect(session=session)
            print(result.values, result.stats.execution_mode, result.stats.dispatch_reason)
    print("Classical cache:", cache.info())


if __name__ == "__main__":
    main()
