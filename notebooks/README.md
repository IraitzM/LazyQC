# Notebooks: one program, three engines

Each notebook defines its program once through `lazyqc` and runs it several
ways (lazy batch planner, direct Qulacs, quimb tensor contraction) to compare
correctness, timing, and reuse behavior across scenarios.

- `01_three_engines.ipynb` — same 4-qubit program, three executors.
- `02_batch_sharing.ipynb` — full / separate / grouped / auto vs quimb.
- `03_where_each_wins.ipynb` — scenario matrix: separable batches, cut
  fragments one-shot vs repeated, VQE binding replay.
- `04_cones_cuts_rewrites.ipynb` — circuit diagrams with cone/cut overlays
  and rewrite before/after, via `benchmarks/circuit_viz.py` (needs
  matplotlib, already in the notebooks group).

## Run

Requires the project environment with all extras plus the notebooks group
(already declared in `pyproject.toml`):

```sh
uv sync --all-extras --group notebooks
.venv/bin/python -m ipykernel install --user --name lazyqc --display-name "Python (lazyqc venv)"
```

Execute headlessly (fills outputs in place):

```sh
uv run --group notebooks --all-extras --with nbconvert \
  jupyter nbconvert --to notebook --execute notebooks/01_three_engines.ipynb \
  --output 01_three_engines.ipynb --ExecutePreprocessor.timeout=900
```

One native thread per process is set inside the first cell of each notebook
before importing Qulacs. Timings are medians over a few interleaved repeats
on the author's i7-13700H; expect session variance on sub-millisecond edges.
