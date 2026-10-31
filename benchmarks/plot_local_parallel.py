#!/usr/bin/env python3
"""Render recorded measurements; requires matplotlib, never runs a simulation."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "results/local_parallel")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    summary = json.loads((args.input_dir / "summary.json").read_text())
    cases = {c["name"]: c for c in summary["cases"]}
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), constrained_layout=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (name, label), color in zip(
        [
            ("wide_16q", "16q, 8 tasks"),
            ("balanced_14q_8tasks", "14q, 8 tasks"),
            ("imbalanced_14q", "14q, uneven"),
            ("bridge_cut_8q", "8q cut fragments"),
        ],
        colors,
    ):
        values = [[1.0]] + [
            [
                v["paired_speedups"]["parallel"]["geomean"]
                for v in cases[name]["configurations"][f"w{w}_m{w}"]
            ]
            for w in (2, 4)
        ]
        axes[0].plot(
            [1, 2, 4], [median(v) for v in values], "o-", label=label, color=color, markersize=3
        )
        axes[0].fill_between(
            [1, 2, 4], [min(v) for v in values], [max(v) for v in values], alpha=0.18, color=color
        )
    axes[0].plot([1, 4], [1, 4], "--", color="0.5", linewidth=0.8)
    axes[0].set(
        xlabel="Workers",
        ylabel="Paired warm speedup",
        xticks=[1, 2, 4],
        title="(a) Fixed-workload scaling",
        ylim=(0, 4.2),
    )
    axes[0].legend(fontsize=6, loc="upper left")
    values = [
        [
            r["modes"]["parallel"]["median_ns"] / 1e6
            for r in cases["wide_16q"]["configurations"][f"w4_m{m}"]
        ]
        for m in (1, 2, 4)
    ]
    center = [median(v) for v in values]
    axes[1].bar([0, 1, 2], center, color="#0072B2", width=0.6)
    axes[1].errorbar(
        [0, 1, 2],
        center,
        yerr=[
            [c - min(v) for c, v in zip(center, values)],
            [max(v) - c for c, v in zip(center, values)],
        ],
        fmt="none",
        color="black",
        capsize=3,
    )
    axes[1].set(
        xticks=[0, 1, 2],
        xticklabels=["1", "2", "4"],
        xlabel="In-flight payload cap (MiB)",
        ylabel="Warm collection time (ms)",
        title="(b) Four workers, 16q tasks",
    )
    weak = summary["weak_scaling"]
    for session in weak:
        axes[2].plot(
            [p["workers"] for p in session],
            [p["weak_efficiency"] for p in session],
            "o-",
            linewidth=1,
            markersize=3,
        )
    axes[2].axhline(1, color="0.5", linestyle="--", linewidth=0.8)
    axes[2].set(
        xlabel="Workers (two tasks per worker)",
        ylabel="Weak-scaling efficiency",
        xticks=[1, 2, 4],
        ylim=(0, 1.1),
        title="(c) 14q, 40 layers per task",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "local_parallel_scaling.pdf")
    fig.savefig(args.output_dir / "local_parallel_scaling.png", dpi=200)
    plt.close(fig)

    raw = [
        json.loads(p.read_text()) for p in sorted(args.input_dir.glob("session*/evaluation.json"))
    ]
    selected = ["tiny_4q", "medium_10q", "balanced_14q_8tasks", "wide_16q"]
    fields = ["lowering_ns", "evolution_ns", "expectation_ns", "reconstruction_ns"]
    fractions = []
    for name in selected:
        rows = [
            r
            for d in raw
            for c in d["cases"]
            if c["case"]["name"] == name
            for cfg in c["configurations"]
            if cfg["workers"] == 1
            for r in cfg["runs"]
        ]
        first = [mean(r["stats"][field] / r["total_ns"] for r in rows) for field in fields]
        fractions.append(first + [1 - sum(first)])
    fig, ax = plt.subplots(figsize=(3.45, 2.6), constrained_layout=True)
    bottom = [0.0] * len(selected)
    for i, (label, color) in enumerate(
        zip(
            ["Lowering", "Evolution + norm", "Expectations", "Reconstruction", "Other host work"],
            ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#999999"],
        )
    ):
        heights = [100 * v[i] for v in fractions]
        ax.bar(range(len(selected)), heights, bottom=bottom, label=label, color=color, width=0.65)
        bottom = [a + b for a, b in zip(bottom, heights)]
    ax.set(
        xticks=range(4),
        xticklabels=["4q", "10q", "14q", "16q"],
        ylabel="Share of serial collection time (%)",
        xlabel="Eight-task workloads",
        ylim=(0, 100),
    )
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.27), ncol=2, frameon=False)
    fig.savefig(args.output_dir / "local_parallel_phases.pdf")
    fig.savefig(args.output_dir / "local_parallel_phases.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
