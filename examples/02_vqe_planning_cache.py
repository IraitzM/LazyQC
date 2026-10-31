"""Tutorial 2: Variational Loops and Classical Planning Cache in LazyQC.

This tutorial demonstrates:
1. Simulating an iterative variational quantum algorithm (VQE-style loop).
2. Using PlanningCache to eliminate redundant classical AST and graph planning.
3. Observing the 'miss' on cold plan compilation vs. 'hit' on subsequent iterations.
4. Verifying that cache hits still trigger fresh quantum state evolution.
5. Monitoring object-reachability memory usage within the cache.
"""

import os
import time

for env_var in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env_var] = "1"

from lazyqc import Parameter, PlanningCache, Program, plan_all


def build_hamiltonian_queries(theta_val: float):
    """Constructs a parameterized 4-qubit circuit and requests Hamiltonian term expectations."""
    theta = Parameter("theta")
    # A simple 4-qubit ansatz with parameterized rotations and entangling CX gates
    p = (
        Program(4)
        .ry(0, theta)
        .ry(1, theta)
        .cx(0, 1)
        .ry(2, 0.5)
        .cx(2, 3)
        .bind(theta=theta_val)
    )

    # Hamiltonian H = 0.5 * Z0 Z1 + 0.3 * X0 X1 + 0.8 * Z2 Z3
    queries = [
        p.expectation({0: "Z", 1: "Z"}),
        p.expectation({0: "X", 1: "X"}),
        p.expectation({2: "Z", 3: "Z"}),
    ]
    return queries


def main():
    print("=" * 70)
    print("Tutorial 2: Iterative Variational Execution & PlanningCache")
    print("=" * 70)

    # Instantiate a classical planning cache bounded by entry count and byte capacity
    cache = PlanningCache(max_entries=32, max_bytes=10 * 1024 * 1024)
    print(f"Initialized PlanningCache: {cache.info()}")

    # Simulated optimization loop over repeated parameters
    # E.g., re-evaluating batches across parameter trials
    test_angles = [0.2, 0.2, 0.2, 0.5, 0.5]

    for step, angle in enumerate(test_angles):
        t0 = time.perf_counter()
        queries = build_hamiltonian_queries(angle)

        # Plan with the cache
        plan = plan_all(queries, strategy="auto", cache=cache)
        explanation = plan.explain()
        planning_time_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        result = plan.collect()
        collection_time_ms = (time.perf_counter() - t1) * 1000

        # Calculate energy E = 0.5 * <Z0 Z1> + 0.3 * <X0 X1> + 0.8 * <Z2 Z3>
        z0z1, x0x1, z2z3 = result.values
        energy = 0.5 * z0z1 + 0.3 * x0x1 + 0.8 * z2z3

        cache_status = explanation.get("cache_status", "miss")
        print(
            f"Step {step+1:02d} | Angle: {angle:.2f} | "
            f"Cache: {cache_status:<4} | "
            f"Planning: {planning_time_ms:6.3f} ms | "
            f"Exec: {collection_time_ms:6.3f} ms | "
            f"Energy: {energy:.6f}"
        )

    print("\n" + "-" * 50)
    print("Final Cache Status:")
    info = cache.info()
    from dataclasses import asdict
    for k, v in asdict(info).items():
        print(f"  {k}: {v}")
    print("-" * 50)
    print("Note: Even on cache hits, collect() allocates fresh quantum states")
    print("and evaluates numerical expectations without stale cache pollution.")


if __name__ == "__main__":
    main()
