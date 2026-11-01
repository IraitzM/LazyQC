## v0.4.0 (2026-11-01)

### Feat

- **plan**: charge a lowering cost per gate in the work score
- **planner**: plan once and rebind angles across training steps

## v0.3.0 (2026-11-01)

### Feat

- **qiskit**: `Estimator` implementing `BaseEstimatorV2`, so `EstimatorQNN` and other primitive-based code can swap to LazyQC; every PUB of a job is planned together and parametric circuits are translated once (#5, #13, #23)
- **ir**: native `CZ` and `RZZ` gates across the IR, all backends, the adjoint kernel, the Qiskit and PennyLane frontends and the ZX conversion; QAOA circuits keep about half the gates (#14, #26)
- **executor**: `collect(native_threads="auto")` sizes Qulacs' OpenMP team by task width, one thread below 14 qubits and at most 8 above; adds the `threadpoolctl` dependency and `stats.native_threads` (#24, #25)

## v0.2.0 (2026-10-31)

### Feat

- **ir**: weighted Pauli sums as observables; `Program.expectation` accepts a `PauliSum` or `(coefficient, term)` pairs, terms with equal cones share a simulation (#12, #21)
- **planner**: hand off to exact matrix-product states when no state-vector plan fits, with a static bond-dimension model and `explain()` support (#4, #20)
- **backends**: Aer, Lightning and quimb backends, plus wide-task placement on a GPU (#19)
- **pennylane**: `lazyqc.qubit` device with cone-restricted device derivatives (#6, #16)
- **gradients**: cone-local adjoint differentiation (#9, #17)

### Fix

- **python**: Version pinned

## v0.1.0 (2026-10-23)

### Fix

- **tests**: Removed benchmark dependencies
