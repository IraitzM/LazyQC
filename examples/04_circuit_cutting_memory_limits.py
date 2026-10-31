"""Tutorial 4: Circuit Cutting under Memory Limits in LazyQC.

This tutorial demonstrates:
1. Simulating an entangled 4-qubit circuit under a strict memory ceiling:
   each simulator worker is restricted to at most 2 qubits (max_state_qubits=2, max_state_bytes=64).
2. Configuring CutOptions to partition the circuit into sub-registers (e.g., qubits 0-1 and qubits 2-3).
3. Using Quasi-Probability Decompositions (QPD) to sever entangling CX gates across the boundary.
4. Classical reconstruction of exact expectation values from cut branch experiments.
5. Inspecting the plan explanation to see cut gates, fragment components, and QPD recipes.
"""

import os

for env_var in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env_var] = "1"

from lazyqc import CutOptions, Program, plan_all


def main():
    print("=" * 70)
    print("Tutorial 4: Circuit Cutting with Strict Memory Constraints")
    print("=" * 70)

    # 1. Define an entangled 4-qubit GHZ-like chain
    # State: |0000> + |1111> / sqrt(2)
    p = (
        Program(4)
        .h(0)
        .cx(0, 1)
        .cx(1, 2)  # Entangling gate connecting partition (0,1) and (2,3)
        .cx(2, 3)
    )
    print("Constructed 4-qubit entangled chain: H(0) -> CX(0,1) -> CX(1,2) -> CX(2,3)")

    # Request correlations across the boundary
    queries = [
        p.expectation({0: "Z", 3: "Z"}),      # Boundary-spanning correlator: should be +1
        p.expectation({1: "Z", 2: "Z"}),      # Cut-adjacent correlator: should be +1
        p.expectation({0: "X", 1: "X", 2: "X", 3: "X"}),  # 4-body parity correlator: should be +1
    ]
    print(f"Registered {len(queries)} cross-partition expectation queries.")

    # 2. Plan with strict memory bounds
    # Suppose our target execution environment cannot allocate more than 2 qubits (64 bytes).
    # Without cutting, this would raise NoFeasiblePlan!
    print("\nPlanning with CutOptions(max_fragment_qubits=2, partition=((0, 1), (2, 3)))...")
    plan = plan_all(
        queries,
        cutting=CutOptions(
            max_fragment_qubits=2,
            max_cuts=1,
            partition=((0, 1), (2, 3)),
            search="joint",
        ),
        max_state_qubits=2,
        max_state_bytes=64,
    )

    # 3. Inspect the classical plan
    explanation = plan.explain()
    print("\n" + "-" * 50)
    print("Plan Explanation Summary:")
    print(f"  Max fragment qubits: {explanation.get('max_state_qubits', 2)}")
    print(f"  Candidate provenance: {explanation.get('strategy', 'cut')}")
    if "cutting" in explanation:
        cut_info = explanation["cutting"]
        print(f"  Cut locations: {cut_info.get('cut_gates', [])}")
        print(f"  Partitions: {cut_info.get('partitions', [])}")
    print("-" * 50)

    # 4. Collect exact reconstructed results
    print("\nExecuting cut branches and reconstructing expectation values...")
    result = plan.collect()

    print("\nExact Reconstructed Expectations:")
    for idx, (q, val) in enumerate(zip(queries, result.values)):
        print(f"  Query {idx} ({q.observable.factors}): {val:.6f}")

    print("\nExecution Statistics:")
    print(f"  Allocations: {result.stats.allocations}")
    print(f"  Evolutions:  {result.stats.evolutions}")
    print(f"  Peak active state payload: {result.stats.peak_active_state_bytes} bytes (<= 64 B limit)")
    print(f"  Execution mode: {result.stats.execution_mode}")


if __name__ == "__main__":
    main()
