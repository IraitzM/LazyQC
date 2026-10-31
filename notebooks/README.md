# Notebooks

Four short, executable tours of LazyQC. Each one builds its circuits with the
`Program` API, asks its questions as lazy requests and shows what the planner
decides before anything is simulated.

- `01_three_engines.ipynb`: one 4-qubit program evaluated by the LazyQC planner,
  by Qulacs and by quimb, with the candidates the planner weighed and what each
  engine costs.
- `02_batch_sharing.ipynb`: full, separate, grouped and auto plans on independent
  pairs, a sweep from 2 to 20 qubits that shows where sharing starts to pay, the
  regret of `auto`, and the analysis budget that skips planning when it cannot pay.
- `03_gradients_depth_wide.ipynb`: the workloads the planner is built for. A
  parameter-shift Jacobian and its structural zeros, analysis keyed on circuit shape,
  how depth saturates the cones, 128-qubit registers with narrow cones, and placing
  wide tasks on a GPU.
- `04_where_each_wins.ipynb`: three scenarios (a separable batch, circuit cutting
  evaluated once or many times, structural replay across parameter steps) and a
  table of rules of thumb.
- `05_cones_cuts_rewrites.ipynb`: drawings of causal cones (and what a conservative
  scan keeps compared with Qiskit's `LightCone`), cut positions with their
  fragments, and local rewrites.
- `06_pennylane_gradients.ipynb`: PennyLane integration with LazyQC. Demonstrates
  `qml.device("lazyqc.qubit")` with cone-restricted parameter-shift gradients,
  and the native `adjoint_gradient` function for weighted Pauli sums with cone-aware
  backpropagation. Shows `profile_sensitivity(..., method="auto")` selecting shift
  vs. adjoint per observable cone.

`nbtools.py` holds the small timing, table and drawing helpers the notebooks
share. It is a teaching aid, not part of the package. Jupyter starts each
notebook in this folder, so a plain `from nbtools import ...` finds it.

## Run

The notebooks need all extras (quimb for the tensor engine, Qiskit for exact
cuts) plus the `notebooks` group:

```sh
uv sync --all-extras --group notebooks
.venv/bin/python -m ipykernel install --user --name lazyqc --display-name "Python (lazyqc venv)"
```

To execute them headlessly and store the outputs:

```sh
uv run --locked --all-extras --group notebooks --with nbconvert \
  jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

Each notebook pins the native libraries to one thread before importing Qulacs,
so timings compare algorithms rather than core counts. Timings are medians of a
few repeats on one machine; expect them to move between runs, especially below a
millisecond.
