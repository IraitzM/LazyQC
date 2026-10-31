"""Tutorial 1: Fundamentals of Lazy Quantum Result Queries in LazyQC.

This tutorial demonstrates:
1. Building immutable quantum programs with symbolic parameters.
2. Registering deferred result queries (Pauli expectations) without simulating.
3. Comparing classical execution strategies: 'full', 'separate', 'grouped', and 'auto'.
4. Inspecting the classical plan with plan.explain() before simulation starts.
5. Executing the plan with plan.collect().
"""

import os

# Ensure deterministic single-threaded native execution
for env_var in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env_var] = "1"

from lazyqc import Parameter, Program, plan_all


def main():
    print("=" * 70)
    print("Tutorial 1: Basic Lazy Evaluation and Strategy Comparison")
    print("=" * 70)

    # 1. Define an immutable quantum program
    # Logical wires can be named strings (e.g., 'a', 'b') or integer indices.
    theta = Parameter("theta")
    program = (
        Program(("a", "b", "c", "d"))
        .ry("a", theta)
        .cx("a", "b")
        .ry("c", 0.8)
        .cx("c", "d")
        .bind(theta=0.4)
    )
    print(f"Program constructed with {len(program.qubits)} qubits and {len(program.operations)} operations.")

    # 2. Declare deferred result queries
    # Notice: NO quantum state is allocated or evolved here!
    queries = [
        program.expectation({"a": "Z"}),
        program.expectation({"b": "Z"}),
        program.expectation({"a": "X", "b": "X"}),
        program.expectation({"c": "Z", "d": "Z"}),
    ]
    print(f"Registered {len(queries)} deferred expectation queries.")

    # 3. Strategy Comparison
    strategies = ["full", "separate", "grouped", "auto"]
    for strat in strategies:
        print("\n" + "-" * 50)
        print(f"Planning with strategy: '{strat}'")
        print("-" * 50)

        # Classical planning: constructs the task DAG without running the simulator
        plan = plan_all(queries, strategy=strat)
        explanation = plan.explain()

        # Explanation details
        selected = explanation.get("selected_strategy", strat)
        evolutions = explanation.get("evolutions", len(explanation.get("groups", [])))
        max_qubits = explanation.get("max_state_qubits")
        print(f"Strategy selected: {selected}")
        print(f"Planned evolutions: {evolutions}")
        print(f"Max statevector width: {max_qubits} qubits (vs 4 in full register)")

        # 4. Materialization / Collection
        # Calling collect() triggers native Qulacs simulation for the planned tasks
        result = plan.collect()
        print("Expectation values:")
        for q_idx, val in enumerate(result.values):
            print(f"  Query {q_idx}: {val:.6f}")
        print(f"Simulation statistics: {result.stats}")


if __name__ == "__main__":
    main()
