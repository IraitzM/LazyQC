"""Tutorial 5: Declarative Logical Frontend (Rhyme-Style) in LazyQC.

This tutorial demonstrates:
1. Declaring quantum subsystems using logical, discrete basis values
   (e.g., a qutrit with energy levels [10, 20, 30]) without manual wire indexing.
2. Applying declarative operations (interference pairs, phase tables) over logical values.
3. Automatically compiling logical operations into an encoded quantum circuit.
4. Expressing discrete probability queries (e.g. P(level == 20)) as decomposed Pauli expectations.
5. Evaluating probabilities via lazy batch planning and collection.
"""

import json
import os

for env_var in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[env_var] = "1"

from lazyqc import plan_all
from lazyqc.frontends.rhyme import from_rhyme_ir


def main():
    print("=" * 70)
    print("Tutorial 5: Declarative Logical Subsystems & Probability Queries")
    print("=" * 70)

    # 1. Define a declarative schema for a qutrit system
    # The subsystem 'level' has discrete logical states [10, 20, 30].
    # Encoded into ceil(log2(3)) = 2 qubits.
    qutrit_spec = {
        "schema_version": 1,
        "subsystems": [{"name": "level", "basis": [10, 20, 30]}],
        "operations": [
            {"kind": "interfere", "targets": ["level"], "pairs": [[0, 1]]},
            {"kind": "phase", "targets": ["level"], "angles": [0.0, 0.7, 0.0]},
            {"kind": "interfere", "targets": ["level"], "pairs": [[1, 2]]},
        ],
    }
    print("Logical Specification:")
    print(json.dumps(qutrit_spec, indent=2))

    # 2. Compile declarative IR into an encoded LazyQC program
    imported = from_rhyme_ir(qutrit_spec)
    print(f"\nCompiled into Program with {imported.program.n_qubits} qubits.")

    # 3. Create deferred probability queries for each logical basis value
    query_10 = imported.probability("level", 10)
    query_20 = imported.probability("level", 20)
    query_30 = imported.probability("level", 30)

    # Combine all underlying Pauli expectation requests for joint batch planning
    all_requests = query_10.requests + query_20.requests + query_30.requests
    print(f"Total underlying Pauli requests to evaluate: {len(all_requests)}")

    # 4. Plan and collect results lazily
    print("\nPlanning and collecting expectation batch...")
    plan = plan_all(all_requests, strategy="grouped")
    result = plan.collect()

    # 5. Evaluate discrete basis probabilities from collected Pauli expectations
    offset = 0
    p10 = query_10.evaluate(result.values[offset : offset + len(query_10.requests)])
    offset += len(query_10.requests)

    p20 = query_20.evaluate(result.values[offset : offset + len(query_20.requests)])
    offset += len(query_20.requests)

    p30 = query_30.evaluate(result.values[offset : offset + len(query_30.requests)])

    print("\n" + "-" * 50)
    print("Evaluated Discrete State Probabilities:")
    print(f"  P(level = 10): {p10:.4f}")
    print(f"  P(level = 20): {p20:.4f}")
    print(f"  P(level = 30): {p30:.4f}")
    print(f"  Total Valid Probability: {p10 + p20 + p30:.6f}")
    print("-" * 50)

    # 6. Verify that padding space (state 3) has zero leakage probability
    invalid_queries = imported.invalid_probabilities("level")
    if invalid_queries:
        leakage_requests = [r for q in invalid_queries for r in q.requests]
        leakage_results = plan_all(leakage_requests).collect().values
        leakage_prob = invalid_queries[0].evaluate(leakage_results)
        print(f"  Padded Space Leakage (code 3): {leakage_prob:.6f} (Strictly 0.0)")


if __name__ == "__main__":
    main()
