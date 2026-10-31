"""Run an exact Bell correlation with one-qubit fragments and two local workers."""

import json
import os


def main():
    for name in (
        "QULACS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[name] = "1"
    from lazyqc import CutOptions, PlanningCache, Program, plan_all

    p = Program(("left", "right")).h("left").cx("left", "right")
    requests = [p.expectation({"left": axis, "right": axis}) for axis in "XYZ"]
    plan = plan_all(
        requests,
        optimization="zx",
        cutting=CutOptions(max_fragment_qubits=1, max_cuts=1),
        workers=2,
        max_state_qubits=1,
        max_state_bytes=32,
        total_state_bytes=64,
        cache=PlanningCache(8),
    )
    print(json.dumps(plan.explain(), indent=2))
    result = plan.collect()
    print("XX, YY, ZZ:", result.values)  # (1, -1, 1)
    print(result.stats)


if __name__ == "__main__":
    main()
