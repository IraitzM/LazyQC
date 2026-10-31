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
