"""Render the slide-deck figures from the recorded JSON artifacts.

Every figure the deck embeds can be reproduced from a plain script reading
``results/`` (plus ``paper/figures/`` for pre-rendered study plots).
Nothing writes under ``results/``; output lands in ``slide-assets/`` for the
Quarto and Marp decks to embed. Some extension plots summarize historical
tables below; the pair plots are calculated from the current JSON artifacts.

Run with::

    uv run --with matplotlib python scripts/render_slide_figures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "slides" / "slide-assets"
OUT.mkdir(exist_ok=True)
RESULTS = ROOT / "results"
PAPER_FIGS = ROOT / "paper" / "figures"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summary_index(summary: list[dict]) -> dict[tuple[str, str], dict]:
    return {(s["case"], s["strategy"]): s for s in summary}


# ── Benchmark A: 12-qubit pairs bar ──────────────────────────────────────────
initial = load_json(RESULTS / "initial.json")
idx = summary_index(initial["summary"])
strategies = ["full", "separate", "grouped", "auto"]
labels = ["full\n(dense)", "separate\n(24 cones)", "grouped\n(6 shared)", "auto"]
medians_ms = [idx[("independent_pairs_12q", s)]["median_ns"]["total_ns"] / 1e6 for s in strategies]
colours = ["#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"]

fig, ax = plt.subplots(figsize=(8.2, 3.8))
bars = ax.bar(labels, medians_ms, color=colours, width=0.62)
for bar, v in zip(bars, medians_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.03, f"{v:.3f} ms",
            ha="center", fontsize=9)
ax.set_ylabel("median total (ms, 20 interleaved trials)")
pair_speedup = medians_ms[0] / medians_ms[2]
ax.set_title(f"12 qubits, 24 observables: grouped {pair_speedup:.2f}x over dense", fontsize=11)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "pairs_bar.png")

# ── Timing breakdown (medians, ms) ───────────────────────────────────────────
phases = ["construction_ns", "planning_ns", "lowering_ns",
          "evolution_ns", "expectation_ns", "assembly_ns"]
phase_labels = ["construct", "plan", "lower", "evolve", "expect", "assemble"]
vals = [[idx[("independent_pairs_12q", s)]["median_ns"][p] / 1e6 for p in phases]
        for s in strategies]
fig, ax = plt.subplots(figsize=(8.6, 3.8))
bottom = [0.0] * len(strategies)
for j, lab in enumerate(phase_labels):
    v = [vals[i][j] for i in range(len(strategies))]
    ax.bar(labels, v, bottom=bottom, label=lab)
    bottom = [b + x for b, x in zip(bottom, v)]
ax.set_ylabel("median phase time (ms)")
ax.set_title("Where the 12-qubit milliseconds go (phase medians need not sum)",
             fontsize=11)
ax.legend(fontsize=8, ncol=3)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "timing_breakdown.png")

# ── Size sweep: full vs grouped vs auto ──────────────────────────────────────
sweep = load_json(RESULTS / "pairs_sweep.json")
sidx = summary_index(sweep["summary"])
widths = [2, 4, 8, 12, 16]
series = {}
for s in ["full", "grouped", "auto"]:
    series[s] = [sidx[(f"independent_pairs_{n}q", s)]["median_ns"]["total_ns"] / 1e6
                 for n in widths]
fig, ax = plt.subplots(figsize=(8.2, 3.8))
for s, marker in [("full", "o"), ("grouped", "s"), ("auto", "^")]:
    ax.plot(widths, series[s], marker=marker, lw=1.8, label=s)
ax.set_xscale("log", base=2)
ax.set_xticks(widths)
ax.set_xticklabels([str(w) for w in widths])
ax.set_yscale("log")
ax.set_xlabel("qubits (4 observables per pair)")
ax.set_ylabel("median total (ms, log scale)")
ax.set_title("Crossover between 8 and 12 qubits; auto misses at 4 and 8",
             fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
save(fig, "sweep_curve.png")

# ── Cutting sharing: B3 vs B5 unique experiments ─────────────────────────────
# Values are the paper Table 2 historical session (seven paired trials);
# benchmarks/run_p4_p5.py --repeats 7 reproduces the raw evaluation.json.
workloads = ["QAOA path", "QAOA star", "Overlapping", "Disjoint"]
b3 = [148, 76, 8, 8]
b5 = [67, 58, 1, 8]
x = list(range(len(workloads)))
w = 0.36
fig, ax = plt.subplots(figsize=(8.2, 3.4))
ax.bar([i - w / 2 for i in x], b3, width=w, label="B3 independent + dedup")
ax.bar([i + w / 2 for i in x], b5, width=w, label="B5 joint bounded")
for i, (a, b) in enumerate(zip(b3, b5)):
    ax.text(i - w / 2, a * 1.04, str(a), ha="center", fontsize=8)
    ax.text(i + w / 2, max(b, 1) * 1.12, str(b), ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(workloads)
ax.set_ylabel("unique fragment experiments")
ax.set_title("Joint selection removes 54.7% / 23.7% / 87.5% of experiments",
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "cutting_sharing.png")

# ── Amortization schematic over 512 calls ────────────────────────────────────
# Paper Table 3 ranges across three interpreter sessions; raw series live in
# results/priorities/{session0,session1,session2}.
am_labels = ["QAOA path", "Overlap", "QAOA star", "Pauli batch"]
am_speed = [1.595, 1.215, 0.965, 0.66]
am_colors = ["#2ca02c" if v >= 1 else "#d62728" for v in am_speed]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
bars = ax.bar(am_labels, am_speed, color=am_colors, width=0.58)
ax.axhline(1.0, color="black", ls=":", lw=1.4)
for bar, v in zip(bars, am_speed):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f"{v:.2f}x",
            ha="center", fontsize=9)
ax.set_ylabel("512-call speedup vs independent")
ax.set_title("Reuse repays on path/overlap, never on star/Pauli (3 sessions)",
             fontsize=11)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "amortization.png")

# ── Workers scaling (from SUMMARY.md ranges; raw in local_parallel/) ────────
worker_labels = ["tiny\n4q", "medium\n10q", "bal 14q\n2t", "bal 14q\n4t",
                 "bal 14q\n8t", "wide\n16q", "shallow\n14q", "deep\n14q",
                 "imbal.\n14q", "bridge\n8q"]
warm4 = [0.395, 1.22, 1.695, 2.835, 2.865, 3.23, 2.285, 2.88, 1.27, 1.015]
life = [0.835, 0.90, 1.095, 1.44, 1.76, 2.595, 1.01, 1.96, 1.015, 0.885]
x = list(range(len(worker_labels)))
w = 0.38
fig, ax = plt.subplots(figsize=(10.5, 3.6))
ax.bar([i - w / 2 for i in x], warm4, width=w, label="4-worker warm speedup")
ax.bar([i + w / 2 for i in x], life, width=w, label="4-worker lifetime speedup")
ax.axhline(1.0, color="black", ls=":", lw=1.4)
ax.set_xticks(x)
ax.set_xticklabels(worker_labels, fontsize=8)
ax.set_ylabel("speedup vs serial")
ax.set_title("Warm wins peak at wide_16q (3.23x); lifetime decides the rest",
             fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "workers_scaling.png")

# ── Scaling extremes schematic ───────────────────────────────────────────────
scaling = load_json(RESULTS / "scaling_extremes.json")
best_rows = scaling.get("best_case", [])
print("best_case rows:", len(best_rows))
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=False)
ns = [r["n_qubits"] for r in best_rows]
tot = [r["total_median_ms"] for r in best_rows]
axes[0].plot(ns, tot, marker="o", lw=1.8)
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ns)
axes[0].set_xticklabels([str(n) for n in ns], fontsize=8)
axes[0].set_xlabel("qubits (independent pairs, live rows)")
axes[0].set_ylabel("median plan/collect (ms)")
axes[0].set_title("150q pairs: 75 local evolutions", fontsize=10)
axes[1].bar(["support cone", "true work"], [24, 0], color=["#7f7f7f", "#2ca02c"])
axes[1].set_ylabel("qubits / expectation work")
axes[1].set_title("Complete graph: cone covers all, answer is 0", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.25, axis="y")
fig.suptitle("Best case scales in components; worst case exposes the analysis",
             fontsize=11)
fig.tight_layout()
save(fig, "scaling_extremes.png")

# ── Sensitivity payload bars (from sensitivity/SUMMARY.md ranges) ────────────
sens_labels = ["8q→4", "12q→6", "16q→8", "15q→5", "16q ctrl"]
payload_factor = [16, 64, 256, 1024, 1]
life_sens = [0.71, 1.35, 3.23, 1.43, 0.795]
fig, ax = plt.subplots(figsize=(8.4, 3.4))
bars = ax.bar(sens_labels, life_sens,
              color=["#2ca02c" if v >= 1 else "#d62728" for v in life_sens])
ax.axhline(1.0, color="black", ls=":", lw=1.4)
for bar, v, f in zip(bars, life_sens, payload_factor):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.06, f"{v:.2f}x\n(/{f})",
            ha="center", fontsize=8)
ax.set_ylabel("lifetime speedup over 8 collections")
ax.set_title("Weak bridges repay profiling; connected/small controls regress",
             fontsize=11)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "sensitivity_payload.png")

# ── Planning cost schematic ──────────────────────────────────────────────────
cost_labels = ["heuristic\nchoice", "calibrated\nchoice", "cold joint\nsearch"]
cost_ms = [0.16, 0.23, 4.68]
fig, ax = plt.subplots(figsize=(8.2, 3.4))
bars = ax.bar(cost_labels, cost_ms, color=["#1f77b4", "#9467bd", "#d62728"])
for bar, v in zip(bars, cost_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, v * 1.04, f"{v:.2f} ms",
            ha="center", fontsize=9)
ax.set_ylabel("planning-time scale (ms, illustrative)")
ax.set_title("Better selection costs planning; small jobs cannot absorb it",
             fontsize=11)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
save(fig, "planning_cost.png")

# ── Copy the two pre-rendered study figures for direct embedding ─────────────
for name in ["local_parallel_scaling.png", "local_parallel_phases.png",
             "sensitivity_tradeoff.png", "sensitivity_precision.png"]:
    src = PAPER_FIGS / name
    if src.exists():
        shutil.copy(src, OUT / name)
        print(f"copied {name}")

print("\nall figures written to slide-assets/")
