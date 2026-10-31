"""Render the figures of the conference talk (slides/conference.qmd).

Redraws the paper's direct, gradient and memory-wall results at slide size from
the recorded results/direct/summary.json (never reruns a benchmark), and draws a
brickwork cone and the Jacobian heatmap of notebooks/03_gradients_depth_wide.ipynb.
Writes into slides/slide-assets/.

    uv run --locked --all-extras --with matplotlib python scripts/render_talk_figures.py
"""

import json
import math
import os
import random
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
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib  # noqa: E402  (Agg backend must be set before pyplot import)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from nbtools import cone_indices, draw_circuit  # noqa: E402
from summarize_direct import DIRECT, FAMILIES, FAMILY_LABELS, STYLE  # noqa: E402

from lazyqc import Parameter, Program, plan_all  # noqa: E402
from lazyqc.ir import Pauli  # noqa: E402

OUT = ROOT / "slides" / "slide-assets"
SUMMARY = ROOT / "results" / "direct" / "summary.json"
Z_ALL = "same_state_local_outputs"
MPS = ("Exact MPS (quimb)", "#7f56c2", "D", "-")
GRID, SPINE, MUTED, INK = "#e4e3df", "#c3c2b7", "#52514e", "#0b0b0b"

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13})


def style_axes(ax, xticks):
    ax.set_yscale("log")
    ax.set_xticks(xticks)
    ax.tick_params(labelsize=12, colors=MUTED)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(SPINE)


def line(ax, points, style, **extra):
    label, color, marker, dash = style
    xs, ys = zip(*sorted(points))
    ax.plot(xs, ys, dash, color=color, marker=marker, markersize=7, linewidth=2.2,
            label=label, markeredgecolor="white", markeredgewidth=0.8, **extra)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


def legend_below(fig, axes, ncol):
    handles, labels = [], []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, -0.02))


def direct_figure(scaling):
    """Planner against direct simulation with Z on every wire, one panel per family."""
    configs = [c for c in scaling["configs"] if c["pattern"] == Z_ALL]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        rows = [c for c in configs if c["family"] == family]
        for method in ("qulacs_1t", "aer_cpu", "lightning_gpu", "aer_gpu", "lazyqc", "lazyqc_gpu"):
            points = [(c["qubits"], c["ms"]) for c in rows if c["method"] == method]
            if points:
                line(ax, points, STYLE[method], zorder=3 if method.startswith("lazyqc") else 2)
        # Same ratio as the paper's tables: fastest direct method over the planner with placement.
        at28 = {c["method"]: c["ms"] for c in rows if c["qubits"] == 28}
        ratio = min(at28[m] for m in DIRECT if m in at28) / at28["lazyqc_gpu"]
        text = f"{ratio:,.0f}× faster\nat 28 qubits" if ratio > 1.5 else "matches the\nfastest GPU"
        ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", fontsize=13,
                color=STYLE["lazyqc_gpu"][1], fontweight="bold")
        ax.set_title(FAMILY_LABELS[family], color=INK)
        ax.set_xlabel("qubits", color=MUTED)
        style_axes(ax, [12, 16, 20, 24, 28])
    axes[0].set_ylabel("time for all outputs (ms)", color=MUTED)
    legend_below(fig, axes, ncol=6)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "talk_direct.png")


def gradient_figure(jacobian):
    """Full-Jacobian time per method, one panel per family."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, family in zip(axes, FAMILIES, strict=True):
        for method in ("lightning_cpu_adjoint", "aer_gpu_shift", "lightning_gpu_adjoint",
                       "lazyqc", "lazyqc_gpu"):
            points = [(c["qubits"], c["ms"]) for c in jacobian["configs"]
                      if c["family"] == family and c["method"] == method]
            if points:
                line(ax, points, STYLE[method])
        if family == "data_reuploading":
            ax.text(0.96, 0.05, "evaluated up to\n16 qubits", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=12, color=MUTED)
        ax.set_title(FAMILY_LABELS[family], color=INK)
        ax.set_xlabel("qubits", color=MUTED)
        style_axes(ax, [12, 16, 20, 24])
    axes[0].set_ylabel("full Jacobian time (ms)", color=MUTED)
    legend_below(fig, axes, ncol=5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "talk_gradients.png")


def memory_wall_figure(wall):
    """Narrow-cone planning against a shared exact MPS beyond any state vector."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, family in zip(axes, FAMILIES[:2], strict=True):
        lazy = [(p["qubits"], p["total_ms"]) for p in wall["planned"]
                if p["family"] == family and p["pattern"] == Z_ALL]
        mps = [(m["qubits"], m["total_ms"]) for m in wall["mps"]
               if m["family"] == family and m["pattern"] == Z_ALL and m["depth"] == 4]
        line(ax, mps, MPS)
        line(ax, lazy, STYLE["lazyqc"])
        widths = [p for p in wall["planned"] if p["family"] == family and p["pattern"] == Z_ALL]
        low = min(p["min_width"] for p in widths)
        high = max(p["max_width"] for p in widths)
        ratio = dict(mps)[128] / dict(lazy)[128]
        ax.text(0.96, 0.05, f"{ratio:.1f}× faster at 128 qubits\ncones of {low}–{high} qubits",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
                color=STYLE["lazyqc"][1], fontweight="bold")
        ax.set_title(FAMILY_LABELS[family] + ", depth 4", color=INK)
        ax.set_xlabel("logical qubits", color=MUTED)
        style_axes(ax, [32, 48, 64, 100, 128])
    axes[0].set_ylabel("time for all outputs (ms)", color=MUTED)
    legend_below(fig, axes, ncol=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "talk_memory_wall.png")


def brickwork(n, depth, angles=None):
    """`depth` layers of RY on every wire followed by CX on alternating neighbouring pairs."""
    program, k = Program(n), 0
    for layer in range(depth):
        for q in range(n):
            program = program.ry(q, angles[k] if angles else 0.1 + 0.01 * k)
            k += 1
        for q in range(layer % 2, n - 1, 2):
            program = program.cx(q, q + 1)
    return program


def cone_figure():
    """One output's backward cone in a small brickwork: the irrelevant gates fade."""
    program = brickwork(8, 3)
    operations = program.resolved_operations()
    keep = cone_indices(operations, Pauli(((3, "Z"),)))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    draw_circuit(program, ax=ax, keep=keep, legend=True,
                 title=rf"$\langle Z_3\rangle$ needs {len(keep)} of {len(operations)} gates")
    save(fig, "talk_cone.png")


def jacobian_figure():
    """Parameter-shift Jacobian of a 12-qubit brickwork; grey cells are exact zeros."""
    n, depth = 12, 3
    thetas = [Parameter(f"theta_{i}") for i in range(n * depth)]
    ansatz = brickwork(n, depth, thetas)
    rng = random.Random(3)
    point = {t.name: rng.uniform(-1, 1) for t in thetas}
    programs = [ansatz.bind(**point)]
    for t in thetas:
        for shift in (math.pi / 2, -math.pi / 2):
            programs.append(ansatz.bind(**{**point, t.name: point[t.name] + shift}))
    requests = [p.expectation({k: "Z"}) for p in programs for k in range(n)]
    plan = plan_all(requests)
    values = plan.collect().values
    tasks = len(plan.explain()["groups"])
    P = len(thetas)
    J = np.array([[(values[(1 + 2 * i) * n + k] - values[(2 + 2 * i) * n + k]) / 2
                   for i in range(P)] for k in range(n)])
    zeros = int((J == 0.0).sum())

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#EEEEEE")
    fig, ax = plt.subplots(figsize=(12, 4.4))
    image = ax.imshow(np.ma.masked_equal(np.abs(J), 0.0), cmap=cmap, aspect="auto")
    for layer in range(1, depth):
        ax.axvline(layer * n - 0.5, color="white", linewidth=4)
    ax.set_yticks(range(n), [f"Z{k}" for k in range(n)], fontsize=11)
    ax.set_xticks([layer * n + n / 2 - 0.5 for layer in range(depth)],
                  [f"layer {layer + 1}" for layer in range(depth)])
    ax.set_xlabel(r"parameter $\theta_i$ (wires 0 to 11 within each layer)", color=MUTED)
    ax.set_title(f"{len(requests)} requests → {tasks} tasks; "
                 f"{zeros} of {J.size} entries are exactly zero (grey)", color=INK)
    fig.colorbar(image, ax=ax, label=r"$|\partial\langle Z_k\rangle/\partial\theta_i|$")
    fig.tight_layout()
    save(fig, "talk_jacobian.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(SUMMARY.read_text())
    direct_figure(summary["scaling"])
    gradient_figure(summary["jacobian"])
    memory_wall_figure(summary["memory_wall"])
    cone_figure()
    jacobian_figure()


if __name__ == "__main__":
    main()
