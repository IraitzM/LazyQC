# Observable sensitivity and approximate planning

This extension is opt-in: `lazyqc.sensitivity` profiles a fixed bound
program and proposes zero-angle substitutions under per-observable absolute
error budgets. Ordinary `plan_all` remains exact and does not execute a simulator
during planning. This is parameter approximation, not a reduced-precision backend.

The completed [results report](../results/sensitivity/SUMMARY.md) covers three frozen
sessions, 96 independent random circuits and nine timing fixtures. There were no
observed bound violations in 1,152 simultaneous-perturbation output comparisons.
The constructed 16-qubit two-block cases reduced largest-state payload by 256x
and achieved 2.89–3.57x serial total-time speedup over eight collections, with
maximum observable bias 1.14e-5. Small and connected controls regressed. These
timings charge profiling and search; they do not establish a universal benefit.
The [trade-off figure](../paper/figures/sensitivity_tradeoff.png) and
[finite-difference diagnostic](../paper/figures/sensitivity_precision.png) show both
the accuracy and overhead findings.

The paper now formalizes this bound as a theorem, proves that its quadratic
coefficient is asymptotically sharp, and derives sufficient step-size and
resource-reduction corollaries. The [theoretical contribution note](THEORETICAL_CONTRIBUTION.md)
states the assumptions and prior-art boundaries. These are proved sufficient
conditions; measured runtime speedups remain workload-specific.

```python
from lazyqc import Program
from lazyqc.sensitivity import profile_sensitivity, plan_approximate

p = Program(2).h(0).h(1).cx(0, 1).rz(1, 0.001).cx(0, 1)
profile = profile_sensitivity(
    p, [{0: "X"}], [3], evaluation_error=1e-10,
)
proposal = plan_approximate(profile, error_budget=1e-5)
print(proposal.zeroed_indices, proposal.error_bounds)
values = proposal.plan.collect().values
```

Gate indices refer to occurrences in the original program. Profiling performs two
parameter-shift evaluations per relevant occurrence, grouping requested outputs.
`profile.parameter_gradient(name)` sums all occurrences of a shared parameter;
it refuses incomplete coverage. Changing an angle or the requested observables
requires a new profile. The API never reuses a local Jacobian across optimiser steps.

For local multiprocessing, pass the same `workers` value to both functions and an
explicit `ExecutionSession(workers, policy="parallel")` to profiling and the
selected batch plan's `collect(session=session)`. Own the session with a context
manager. Its warmup and shutdown are additional costs. Small reduced fragments
can run faster serially even when shifted profiling circuits benefit from workers.

## Bound and assumptions

For each norm-one terminal Pauli observable, let C be its backward structural cone,
delta_g the change in an eligible rotation occurrence, and L = sum over C of
abs(delta_g). The exact single-occurrence shift rule is

    J_kg = [f_k(theta_g + pi/2) - f_k(theta_g - pi/2)] / 2.

All other occurrences, including those sharing the same symbolic parameter, stay
fixed during this evaluation. A Pauli rotation has generator P/2. Its channel's
first derivative has induced trace norm at most one, by the commutator bound
||[P/2, A]||_1 <= ||A||_1. Products of unitary channels preserve this norm; applying
two derivative maps also has norm at most one. Therefore every first and mixed
second partial derivative of a norm-one expectation has magnitude at most one.
Taylor's integral remainder along the simultaneous perturbation is at most L²/2.
Telescoping over occurrences also gives |Delta f| <= L, and bounded expectations
give |Delta f| <= 2. These arguments hold along the entire segment, not just at its
starting point. Angle changes cannot introduce gates outside the original cone.

Given an assumed absolute error eta for each numerical expectation evaluation,
the shift derivative error is at most eta per occurrence. The implementation uses

    B_k = min(2, L, abs(J_hat_k delta) + eta L + L²/2) + 2 eta.

The last allowance covers comparison of two numerically evaluated endpoints. An
unchanged relevant circuit or identity output has zero approximation change; the
API returns zero in this special case, which should not be read as a certificate
for independent numerical evaluations of that unchanged quantity. For
large L the implementation saturates the quadratic computation at L=2, which
does not change the minimum because both the original and saturated terms are
already at least two. These are conditional analytic bounds: the implementation
does not establish eta through directed rounding or interval arithmetic. The
study assumes eta=1e-10 and checks against an independent NumPy implementation;
agreement between double-precision implementations is empirical evidence, not a
proof of the floating-point allowance. Finite sums used for bounds are not rounded
outward. Do not describe this mode as formally certified arithmetic.

The search tries the full zeroed subset, then subsets ordered by cardinality and
local sensitivity, subject to a candidate limit (64 by default). Only substitutions
meeting every output budget are planned. Exact local cancellation and causal-cone
grouping then expose savings. The exact plan remains a candidate and wins cost ties.
Selection uses existing execution cost estimates; without calibration these are
work scores, not seconds. It is a bounded search, not a globally optimal solution.
The current implementation requires the original plan to be feasible, because it
profiles that program; it does not solve an otherwise unsimulable circuit for free.

## Validation protocol

1. **Derivative accuracy and composed perturbations.** Generate 96 independent
   seeds of two- to five-qubit noncommuting circuits with X/Y/Z rotations and CX
   gates. Compare occurrence derivatives against independently implemented central
   differences at h=1e-5. Test six simultaneous perturbations at scales 1e-6, 1e-3,
   0.1, and 1 radian on three Pauli outputs. Record actual errors and bounds.
2. **Adversarial cases.** Test a stationary point f(theta)=cos(theta) at pi, where
   deleting the rotation changes the output by two despite zero gradient; shared
   parameters, where shifting all occurrences together gives a wrong derivative;
   and a step-size sweep from 1e-2 to 1e-16 showing finite-difference cancellation.
3. **Structural and negative controls.** Prepare random local blocks, connect them
   with CX-RZ(delta)-CX bridges, then apply further noncommuting local layers.
   Use two blocks at widths 8, 12 and 16, and three blocks at width 15. Use three
   circuit seeds for widths 15/16 and one for the smaller controls. An additional
   16-qubit control retains a CX bridge after zeroing, preventing separation.
4. **Accuracy policy ablations.** On the same eligible bridge occurrences, compare
   exact execution, uniform rounding to a 0.02-radian grid, gradient-only acceptance
   and bounded subset selection at budgets 1e-6, 1e-4 and 1e-2. The current eligible
   bridge angles are 0.003 and 0.006, so uniform rounding sets them to zero. This is
   a controlled candidate ablation, not whole-circuit quantisation.
5. **Charged execution and resources.** Run 1/2/4 local workers with one native
   thread each, one first collection and seven interleaved warm repetitions per
   policy. Repeat the entire suite in three sequential interpreter sessions.
   Charge shifted-circuit profiling, approximation search, worker startup and
   shutdown against exact planning plus execution over eight collections. Include
   an independently checked NumPy reference for each original/modified program.
   Resource probes are outside the timing series. Report largest state payload,
   task counts and sampled parent-plus-pool RSS; summed RSS double-counts shared
   pages and is not a memory guarantee. Allocator retention can obscure payload
   savings in RSS. CPU tick resolution is too coarse for short reduced tasks.

The timing harness forces a worker count to isolate its consequences. It does not
automatically select the best worker count or reuse horizon. A future policy should
minimise profiling + search + R times execution under the error and payload budgets;
the measured lifetime comparisons test whether that policy would be useful. The
reference oracle and accuracy-policy sweeps are validation costs and excluded from
deployment timing. Program construction is common to both policies and excluded.
The measured session warmup/shutdown is charged equally to both counterfactual
workflows; each pair uses one actual persistent pool and no state reuse.

## Reproduction

```bash
.venv/bin/python benchmarks/run_sensitivity.py --smoke --repeats 2 --output /tmp/sensitivity.json
.venv/bin/python benchmarks/run_sensitivity.py --output results/sensitivity/session0/evaluation.json
.venv/bin/python benchmarks/run_sensitivity.py --output results/sensitivity/session1/evaluation.json
.venv/bin/python benchmarks/run_sensitivity.py --output results/sensitivity/session2/evaluation.json
.venv/bin/python benchmarks/summarize_sensitivity.py
MPLCONFIGDIR=/tmp/lazyqc-mpl /usr/bin/python3 benchmarks/plot_sensitivity.py
.venv/bin/pytest tests/test_sensitivity.py
```

Run timing sessions sequentially on an otherwise idle host. JSON artifacts record
source SHA-256 hashes and raw observations. The summary verifies identical measured
sources before aggregation. Interpreter sessions are repeated measurements, not
additional independent circuit seeds. The deliberately weak bridges demonstrate
a mechanism, not representative performance for arbitrary variational circuits,
tensor networks, near-Clifford simulators, or changing parameter trajectories.

Final measurements used the frozen snapshot because workspace files were changing
during pilot runs. `pilot0` and `pilot1` are retained but excluded from all reported
aggregates. To execute the measured snapshot directly:

```bash
PYTHONPATH="$PWD/results/sensitivity/frozen_source/src" .venv/bin/python \
  results/sensitivity/frozen_source/benchmarks/run_sensitivity.py \
  --output /tmp/sensitivity-frozen.json
```

Final validation: all 18 sensitivity tests pass, including real multiprocessing,
independent dense matrices and provenance rejection. The available full suite has
106 passes and eight module-level skips because optional Qiskit/cutting/PyZX
dependencies are not installed. None of these optional packages is required by
the sensitivity study. The IEEEtran manuscript builds with resolved
citations and references and no overfull boxes; the new figure and appendix table
were visually inspected.

The parameter-shift identity follows Schuld et al., *Physical Review A* 99, 032331
(2019), https://doi.org/10.1103/PhysRevA.99.032331. Observable-dependent error control
is related to Eddins, Tran and Rall's lightcone shading (arXiv:2409.04401), whose
application is error mitigation. This framework contribution combines established
derivative/bound ideas with explicit approximation provenance, bounded candidate
planning and charged local execution; it does not claim a new derivative identity.
