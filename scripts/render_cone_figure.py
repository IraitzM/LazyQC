"""Render the causal-cone illustration for the research-update deck.

Same computation as the cone cells of notebooks/05_cones_cuts_rewrites.ipynb:
QAOA-path fixture, full 17-gate circuit plus one faded-gate cone panel per
observable. Writes slides/slide-assets/cone_qaoa.png.
"""

import os
import sys
from pathlib import Path

for key in (
    "QULACS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[key] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib  # noqa: E402  (Agg backend must be set before pyplot import)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from nbtools import cone_indices, draw_circuit  # noqa: E402

from lazyqc import Program  # noqa: E402
from lazyqc.fixtures import load_fixture  # noqa: E402
from lazyqc.ir import Gate, Pauli  # noqa: E402

fixture, cases = load_fixture(ROOT / "tests/fixtures/p4_p5.json")
case = next(c for c in cases if c["name"] == "qaoa_path")
ops = [
    Gate(op["gate"], tuple(op["qubits"]), op.get("angle_radians"))
    for op in case["operations"]
]
program = Program(case["qubits"], tuple(ops))
resolved = program.resolved_operations()
labels = [
    ", ".join(f"{f['qubit']}{f['pauli']}" for f in r["observable"])
    for r in case["requests"]
]

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
draw_circuit(program, ax=axes[0, 0], title=f"full circuit: {len(resolved)} gates, 3 questions")
for ax, k in zip((axes[0, 1], axes[1, 0], axes[1, 1]), range(3)):
    factors = tuple((f["qubit"], f["pauli"]) for f in case["requests"][k]["observable"])
    keep = cone_indices(resolved, Pauli(factors))
    draw_circuit(
        program,
        ax=ax,
        keep=keep,
        title=f"cone of {labels[k]}: {len(keep)}/{len(resolved)} gates",
    )
fig.tight_layout()
output = ROOT / "slides/slide-assets/cone_qaoa.png"
fig.savefig(output, dpi=150)
print(f"wrote {output}")
