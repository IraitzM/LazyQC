"""Figures for the research-update deck (proposal storyline, paper structure).

Reads recorded workload-manifest artifacts; nothing writes under results/.
Output lands in slides/slide-assets/.

Run with::

    uv run --with matplotlib python scripts/render_update_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUT = ROOT / "slides" / "slide-assets"
OUT.mkdir(exist_ok=True)
V2 = ROOT / "results" / "workload_manifest" / "v2"
V1 = ROOT / "results" / "workload_manifest" / "v1"

R_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


policy = load(V2 / "policy_evaluation.json")
fixed = load(V2 / "evaluation_r7_fixed.json")
traj = load(V2 / "trajectory.json")
traj_zx = load(V2 / "trajectory_zx.json")
tensor = load(V2 / "tensor_compare/evaluation.json")
heldout = load(V2 / "evaluation_heldout_guard.json")
conn12 = load(V2 / "tensor_connected12q.json")

ACCENT = "#007D84"
GREY = "#7f7f7f"
BLUE = "#1f77b4"
GREEN = "#2ca02c"
PURPLE = "#9467bd"
ORANGE = "#ff7f0e"


def ht(case: str) -> dict[str, tuple[float, float]]:
    """Median (H cold planning, T warm cached total) per mode, milliseconds."""
    entry = next(c for c in policy["cases"] if c["case"] == case)
    h = entry["H_ns"]
    return {"B3": (h["B3"][0] / 1e6, h["B3"][1] / 1e6), "B5": (h["B5"][0] / 1e6, h["B5"][1] / 1e6)}


def tensor_ht(case: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """(H_plan, T_contract) medians for B4 plus (H, T) for B5, milliseconds."""
    rows = [r for r in tensor["runs"] if r["case"] == case and r["status"] == "ok"]
    b4 = [r for r in rows if r["mode"] == "B4_tensor" and r["repetition"] >= 0]
    b5 = [r for r in rows if r["mode"] == "B5_joint" and r["repetition"] >= 0]
    h4 = median(r["planning_ns"] for r in b4) / 1e6
    t4 = median(r["execution_ns"] for r in b4) / 1e6
    h5 = median(r["planning_ns"] for r in b5) / 1e6
    warm5 = [r for r in fixed["runs"] if r["case"] == case and r["mode"] == "B5_cached"]
    t5 = median(r["total_ns"] for r in warm5) / 1e6
    return (h4, t4), (h5, t5)


# ── Lifetime curves with tensor comparators ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, case in zip(axes, ["qaoa_path", "qaoa_star"]):
    life = ht(case)
    (h4, t4), _ = tensor_ht(case)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("collections R")
    ax.set_title(case.replace("qaoa_", "QAOA "))
    ax.plot(R_GRID, [life["B3"][0] + r * life["B3"][1] for r in R_GRID],
            color=GREY, marker="o", ms=3, label="B3 independent")
    ax.plot(R_GRID, [life["B5"][0] + r * life["B5"][1] for r in R_GRID],
            color=GREEN, marker="s", ms=3, label="B5 joint")
    ax.plot(R_GRID, [r * (h4 + t4) for r in R_GRID],
            color=ORANGE, linestyle="--", marker="^", ms=3, label="B4 tensor (per-call)")
    ax.plot(R_GRID, [h4 + r * t4 for r in R_GRID],
            color=ORANGE, linestyle=":", marker="v", ms=3, label="B4 tensor (path cached)")
    ax.grid(True, which="both", alpha=0.3)
axes[0].set_ylabel("lifetime total (ms)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Lifetime L(R) = H + R·T: joint planning repays only with reuse")
save(fig, "lifetime_curves.png")

# ── Unique-task reduction ────────────────────────────────────────────────────
cases = [
    ("QAOA path", "qaoa_path", fixed),
    ("QAOA star", "qaoa_star", fixed),
    ("Overlap 16", "overlapping_uncut_batch16", heldout),
    ("Disjoint 64", "disjoint_batch64", heldout),
    ("Local Pauli 16", "local_pauli_batch16", heldout),
]
labels, b3, b5 = [], [], []
for label, case, source in cases:
    entry = next(s for s in source["summary"] if s["case"] == case)
    labels.append(label)
    b3.append(entry["B3_independent"]["unique_experiments"][0])
    b5.append(entry["B5_joint"]["unique_experiments"][0])
x = range(len(labels))
fig, ax = plt.subplots(figsize=(9.5, 3.8))
ax.bar([i - 0.2 for i in x], b3, width=0.4, color=GREY, label="B3 independent")
ax.bar([i + 0.2 for i in x], b5, width=0.4, color=GREEN, label="B5 joint")
for i, (a, b) in enumerate(zip(b3, b5)):
    ax.text(i - 0.2, a + 1, str(a), ha="center", fontsize=9)
    ax.text(i + 0.2, b + 1, str(b), ha="center", fontsize=9)
ax.set_xticks(list(x), labels)
ax.set_ylabel("unique fragment experiments")
ax.set_title("Joint selection removes experiments (cold planning still loses: charge it)")
ax.legend()
save(fig, "task_reduction.png")

# ── Trajectory planning speedup ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9.5, 3.8))
groups, full, hint = [], [], []
for case, data in (("path/none", traj), ("star/none", traj),
                   ("path/zx", traj_zx), ("star/zx", traj_zx)):
    entry = next(s for s in data["summary"] if s["case"] == ("qaoa_path" if "path" in case else "qaoa_star"))
    groups.append(case)
    full.append(entry["median_full_ms"])
    hint.append(entry["median_hint_ms"])
x = range(len(groups))
ax.bar([i - 0.2 for i in x], full, width=0.4, color=GREY, label="full planning / binding")
ax.bar([i + 0.2 for i in x], hint, width=0.4, color=ACCENT, label="structural replay / binding")
for i, (a, b) in enumerate(zip(full, hint)):
    ax.text(i - 0.2, a + 2, f"{a:.0f}", ha="center", fontsize=9)
    ax.text(i + 0.2, b + 0.4, f"{b:.1f}", ha="center", fontsize=9)
    ax.text(i, max(a, b) * 1.12, f"{a/b:.0f}×", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x), [f"{g}\n(ms per binding)" for g in groups])
ax.set_ylabel("planning ms")
ax.set_ylim(0, max(full) * 1.3)
ax.set_title("RQ3: structural replay across VQE bindings (8/8 hits, oracle-exact)")
ax.legend()
save(fig, "trajectory_speedup.png")

# ── Tensor two-sided boundary ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
cats = ["path\nB4", "path\nB5", "star\nB4", "star\nB5"]
vals = []
rows = [r for r in tensor["runs"] if r["repetition"] >= 0 and r["status"] == "ok"]
for case, mode in [("qaoa_path", "B4_tensor"), ("qaoa_path", "B5_joint"),
                   ("qaoa_star", "B4_tensor"), ("qaoa_star", "B5_joint")]:
    vals.append(median(r["total_ns"] for r in rows if r["case"] == case and r["mode"] == mode) / 1e6)
ax.bar(cats, vals, color=[ORANGE, GREEN, ORANGE, GREEN])
for i, v in enumerate(vals):
    ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=10)
ax.set_ylabel("cold total (ms)")
ax.set_title("4-qubit fragments: tensor wins")
ax = axes[1]
cats2 = ["12q conn.\nB4 tensor", "12q conn.\ndense full"]
vals2 = [conn12["B4_tensor"]["total_ms"], conn12["v1_reference"]["full_median_ms"]]
ax.bar(cats2, vals2, color=[ORANGE, GREY])
ax.set_yscale("log")
for i, v in enumerate(vals2):
    ax.text(i, v * 1.3, f"{v:,.0f} ms" if v > 100 else f"{v:.1f} ms", ha="center", fontsize=10)
ax.set_ylabel("total (ms, log)")
ax.set_title("12-qubit connected batch: dense wins ~4700×")
fig.suptitle("Tensor boundary is two-sided: arena decides, not method")
save(fig, "tensor_boundary.png")

# ── Budget threshold on the star ─────────────────────────────────────────────
entry = next(s for s in fixed["summary"] if s["case"] == "qaoa_star")
modes = ["B5_budget1", "B5_budget8", "B5_budget32", "B5_joint"]
tasks = [entry[m]["unique_experiments"][0] for m in modes]
plans = [entry[m]["median_total_ms"] for m in modes]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
bars = ax.bar(["R=1", "R=8", "R=32", "uncapped"], plans, color=[GREY, BLUE, GREEN, GREEN])
for bar, t, p in zip(bars, tasks, plans):
    ax.text(bar.get_x() + bar.get_width() / 2, p + 3, f"{t} tasks\n{p:.0f} ms",
            ha="center", fontsize=10)
ax.set_ylabel("cold total (ms)")
ax.set_title("QAOA star: deterministic budget recovers sharing at R=32")
save(fig, "budget_threshold.png")

print("done")
