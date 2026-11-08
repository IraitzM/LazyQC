<p align="center">
  <img src="https://raw.githubusercontent.com/IraitzM/LazyQC/main/docs/assets/logo.svg" alt="LazyQC logo" width="220">
</p>

<h1 align="center">LazyQC</h1>

<p align="center">
  <a href="https://github.com/IraitzM/LazyQC/actions/workflows/ci.yml"><img src="https://github.com/IraitzM/LazyQC/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/IraitzM/LazyQC/blob/main/LICENSE"><img src="https://img.shields.io/github/license/IraitzM/LazyQC" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-blue" alt="Python 3.12+">
  <a href="https://pypi.org/project/lazyqc/"><img src="https://img.shields.io/pypi/v/lazyqc" alt="PyPI"></a>
  <a href="https://github.com/IraitzM/LazyQC/stargazers"><img src="https://img.shields.io/github/stars/IraitzM/LazyQC?style=social" alt="GitHub stars"></a>
</p>

LazyQC is a Python framework for lazy, observable-driven quantum simulation. You describe programs and request expectation values first; LazyQC plans the batch (observable cones, shared preparations, ZX rewrites, and exact CX cuts under memory limits) and only runs the native [Qulacs](https://github.com/qulacs/qulacs) simulation when you call `collect()`.

## Installation

Requires Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```sh
uv add lazyqc
```

Optional extras: `zx` (PyZX rewrites), `cutting` (exact CX cuts, Qiskit frontend), `tensor` and `mqt` (alternative backends for benchmarking).

### PennyLane device

With `pip install "lazyqc[pennylane]"`, any QNode can run on LazyQC:

```python
import pennylane as qml

dev = qml.device("lazyqc.qubit", wires=4)

@qml.qnode(dev, diff_method="best")  # device derivatives skip parameters outside every cone
def f(w):
    qml.RX(w[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(1))
```

`dev.last_stats["skipped_structural_zeros"]` reports shifted terms skipped after a gradient call.

## Gradients with PennyLane

For bound symbolic parameters, `adjoint_gradient` computes a weighted scalar-loss
gradient with Qulacs backpropagation inside each observable's causal cone. Observables
with the same cone share a backpropagation:

```python
import pennylane as qml
from lazyqc import Parameter, Program, adjoint_gradient

dev = qml.device("lazyqc.qubit", wires=2)

@qml.qnode(dev, diff_method="best")  # device derivatives skip parameters outside every cone
def circuit(w):
    qml.RX(w[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.Z(0))
```

Alternatively, using the native LazyQC gradient function:

```python
from lazyqc import Parameter, Program, adjoint_gradient

theta = Parameter("theta")
program = Program(2).ry(0, theta).cx(0, 1).rz(1, theta).bind(theta=0.4)
gradient = adjoint_gradient(program, [{0: "Z"}, {1: "X"}], [0.7, -0.2])
print(gradient["theta"])
```

`profile_sensitivity(..., method="adjoint")` computes occurrence-wise derivatives;
`method="auto"` selects shifts or adjoint independently for each observable cone.

## Quick start

```python
from lazyqc import Program, plan_all

program = Program(4).ry(0, 0.4).cx(0, 1).ry(2, 0.8).cx(2, 3)
requests = [
    program.expectation({0: "Z"}),
    program.expectation({0: "X", 1: "X"}),
    program.expectation({2: "Z", 3: "Z"}),
]

plan = plan_all(requests, strategy="auto")  # classical planning only
print(plan.explain())
print(plan.collect().values)  # native simulation happens here
```

### Gate set

`H`, `X`, `Y`, `Z`, `RX`, `RY`, `RZ`, `CX`, `CZ` and `RZZ` (`program.cz(a, b)`,
`program.rzz(a, b, angle)` with `RZZ(theta) = exp(-i theta Z Z / 2)`). The Qiskit and PennyLane
frontends keep `CZ` and `RZZ` (`IsingZZ`) as single gates instead of expanding them into
`H, CX, H` and `CX, RZ, CX`, which halves the gate count of QAOA-style circuits. `RZZ` uses the
same two-term shift rule as the single-qubit rotations. Circuit cutting still cuts `CX` only.
See `examples/07_qaoa_native_gates.py` for a native against translated comparison.

### Weighted sums of Pauli terms

A Hamiltonian is a list of `(coefficient, term)` pairs. LazyQC plans the terms
together, so terms whose cones coincide share one simulation, and returns the
weighted sum as a single value. Coefficients are real; `Pauli()` is the identity.

```python
hamiltonian = program.expectation([
    (0.5, {0: "Z"}),
    (-1.0, {0: "X", 1: "X"}),
    (2.0, {}),  # identity offset
])
print(plan_all([hamiltonian]).collect().values)  # one value for the whole sum
```

### Training loops: plan once, rebind at every step

Training repeats one circuit structure with new angles. Free the angles with
`Program.parametrize`, build the parameter-shift batch once, and pass the values of each
step to `collect`:

```python
from lazyqc import Pauli, parameter_shift_requests, plan_all

symbolic, values = program.parametrize(indices)          # angles become p0, p1, ...
requests = parameter_shift_requests(symbolic, [Pauli(((0, "Z"),))], indices)
plan = plan_all(requests, symbolic=True)                 # structure only, no angles yet

for step in range(steps):
    result = plan.collect(bindings=values)               # lowered circuits are reused
    ...                                                  # update `values` from the Jacobian
```

Shifts are offsets of the same parameter (`Parameter("p0").shifted(pi / 2)`), so one plan
serves every step and values equal planning from scratch. A symbolic plan lowers each task
once and only sets the new angles afterwards on the Qulacs, Aer and Lightning backends (a
parametric circuit, Qiskit parameter binds and a rebound tape); backends without `rebind`,
such as quimb, bind and lower each task instead. Symbolic planning does not combine with
optimization, cutting or batch options. See `examples/08_plan_once_training.py`.

The Qiskit `Estimator` and the PennyLane device do this for you. When the same circuit
structure arrives a second time (an `EstimatorQNN` training step, a QNode called in a loop),
they keep a symbolic plan and pass the new values as bindings. The first job of a structure
takes the usual path, so one-off jobs pay nothing, and at most four plans are kept per
estimator or device. Values equal the fresh-plan path exactly. The Estimator reports
`result.metadata["lazyqc"]["plan_reused"]` and the device `dev.last_stats["plan_reused"]`;
`optimization`, `cutting`, `workers`, `total_state_bytes` and `cost_model` keep the usual path.

### Qiskit Estimator

`lazyqc.frontends.qiskit.Estimator` implements Qiskit's `BaseEstimatorV2`, so code that goes
through the Estimator primitive, including `EstimatorQNN`, can use LazyQC by swapping the
estimator. Every PUB of a job is planned together, and the circuit is translated once.

```python
from qiskit.quantum_info import SparsePauliOp
from lazyqc.frontends.qiskit import Estimator

estimator = Estimator()  # Estimator(default_precision=0.01, seed=7) adds emulated noise
result = estimator.run([(circuit, SparsePauliOp(["ZZI", "IXX"], [0.5, -1.0]), values)]).result()
print(result[0].data.evs)
```

Values are exact; a `precision` above zero adds Gaussian noise to them as
`StatevectorEstimator` does. See `examples/06_qiskit_estimator_qnn.py` for a training loop timed against Aer.

### Placing wide tasks on a GPU

`plan.collect(wide_backend=..., wide_from=w)` runs every task of at least `w` qubits on the wide
backend (for example `AerBackend` or `LightningBackend` on a GPU) and the narrower ones on
Qulacs. The best `w` depends on the device, the driver and the CPU, so measure it on your host
instead of reusing a constant:

```python
from lazyqc import calibrate_crossover
from lazyqc.backends.aer import AerBackend

gpu = AerBackend(device="GPU")
wide_from, report = calibrate_crossover(gpu, gates=40)   # takes about a second
result = plan.collect(wide_backend=gpu, wide_from=wide_from)
```

`gates` should be close to the size of your tasks, because a GPU pays a launch cost per gate:
the crossover was 15 or 16 qubits for tasks of 10 gates and 14 or 15 for 160 gates on a laptop
RTX 4060. The device matters more than the threshold: on that host Lightning GPU was 1.8 to 2.8
times slower than Aer GPU on 22 to 24 qubit tasks, so compare both before settling.

### Native threads

Qulacs uses one OpenMP thread per core by default, which is slower than a single thread on the
small states cone reduction produces (2 to 3 times at 10 to 12 qubits) and competes with other
OpenMP libraries in the same process, such as Aer. `plan.collect()` therefore runs on one thread
when every task is narrower than 14 qubits, and caps the team at 8 threads for wider tasks. The
setting used is reported in `result.stats.native_threads`. Pass
`collect(native_threads="default")` to keep Qulacs' own team, or an integer to fix its size.

## Gradients

For bound symbolic parameters, `adjoint_gradient` computes a weighted scalar-loss
gradient with Qulacs backpropagation inside each observable's causal cone. Observables
with the same cone share a backpropagation:

```python
from lazyqc import Parameter, Program, adjoint_gradient

theta = Parameter("theta")
program = Program(2).ry(0, theta).cx(0, 1).rz(1, theta).bind(theta=0.4)
gradient = adjoint_gradient(program, [{0: "Z"}, {1: "X"}], [0.7, -0.2])
print(gradient["theta"])
```

`profile_sensitivity(..., method="adjoint")` computes occurrence-wise derivatives;
`method="auto"` selects shifts or adjoint independently for each observable cone.

More examples live in [`examples/`](https://github.com/IraitzM/LazyQC/blob/main/examples) and [`notebooks/`](https://github.com/IraitzM/LazyQC/blob/main/notebooks), and design notes and results in [`docs/`](https://github.com/IraitzM/LazyQC/blob/main/docs).

## Development

```sh
uv sync --locked --all-extras
uv run pytest -q
```

## Contributing

Contributions are welcome. Please read our [Code of Conduct](https://github.com/IraitzM/LazyQC/blob/main/CODE_OF_CONDUCT.md) before opening an issue or pull request.

## Citation

If you use LazyQC in your research, please cite the accompanying paper (see also [`CITATION.cff`](https://github.com/IraitzM/LazyQC/blob/main/CITATION.cff)):

```bibtex
@article{montalban2026lazyqc,
  title   = {LazyQC: Avoiding Simulation Work in Batched Quantum Expectation Workloads},
  author  = {Montalb{\'a}n, Iraitz and Combarro, El{\'i}as F. and Ranilla, Jos{\'e}},
  year    = {2026},
  note    = {arXiv preprint; journal and DOI to be added upon publication}
}
```
