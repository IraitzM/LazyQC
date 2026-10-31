#!/usr/bin/env python3
"""Compare the pre-study planner with the current one on interleaved reruns.

Each version ran the frozen QML matrix and the exhaustive grouping reference in
two rounds, alternating the order (old, new, new, old), so slow drift of the
host affects both versions alike. Planning ratios use per-case medians.

    uv run --locked python benchmarks/compare_planner_versions.py \
        --input-dir results/qml_workloads/planner_ab
"""

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import summarize_grouping_oracle
import summarize_qml_workloads


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def geomean(values):
    values = list(values)
    return math.exp(sum(math.log(v) for v in values) / len(values))


def per_case_medians(payload, field):
    groups = defaultdict(list)
    for row in payload["runs"]:
        groups[(row["case_id"], row["strategy"])].append(row[field])
    return {key: median(values) for key, values in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    files = {path.name.split(".")[0]: path for path in sorted(args.input_dir.glob("*.json*"))}
    result = {"qml": {}, "oracle": {}, "planning_ratio_new_over_old": {}, "drift": {}}
    for version in ("old", "new"):
        heldout = defaultdict(list)
        for round_ in (1, 2):
            payload = load(files[f"qml_{version}_r{round_}"])
            for aggregate in summarize_qml_workloads.summarize(payload)["aggregates"]:
                if aggregate["split"] == "heldout":
                    heldout[aggregate["pattern"]].append(aggregate["geomean_auto_over_best_forced"])
        result["qml"][version] = {pattern: values for pattern, values in heldout.items()}
        oracle = defaultdict(list)
        for round_ in (1, 2):
            summary = summarize_grouping_oracle.summarize(
                load(files[f"oracle_{version}_r{round_}"])
            )
            for aggregate in summary["aggregates"]:
                if aggregate["split"] == "heldout":
                    for key, value in aggregate.items():
                        if key.startswith("geomean"):
                            oracle[key].append(value)
        result["oracle"][version] = dict(oracle)
    for field, label in (
        ("planning_ns", "planning_ratio_new_over_old"),
        ("collection_ns", "drift"),
    ):
        ratios = defaultdict(list)
        for round_ in (1, 2):
            old = per_case_medians(load(files[f"qml_old_r{round_}"]), field)
            new = per_case_medians(load(files[f"qml_new_r{round_}"]), field)
            for key, value in old.items():
                if value > 0 and new.get(key, 0) > 0:
                    ratios[key[1]].append(new[key] / value)
        result[label] = {strategy: geomean(values) for strategy, values in ratios.items()}
    lines = [
        "# Pre-study versus current planner (interleaved reruns)",
        "",
        "Held-out auto / best forced, geometric mean per round:",
        "",
        "| Pattern | Old r1 | Old r2 | New r1 | New r2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for pattern in sorted(result["qml"]["old"]):
        old, new = result["qml"]["old"][pattern], result["qml"]["new"][pattern]
        lines.append(f"| {pattern} | {old[0]:.3f} | {old[1]:.3f} | {new[0]:.3f} | {new[1]:.3f} |")
    lines += [
        "",
        "Planning time new / old (geometric mean over cases and rounds): "
        + ", ".join(
            f"{k} {v:.3f}" for k, v in sorted(result["planning_ratio_new_over_old"].items())
        ),
        "",
        "Collection time new / old (drift check, unchanged code path): "
        + ", ".join(f"{k} {v:.3f}" for k, v in sorted(result["drift"].items())),
        "",
        "Exhaustive grouping reference (held-out), per round:",
        "",
    ]
    for version in ("old", "new"):
        for key, values in sorted(result["oracle"][version].items()):
            lines.append(f"- {version} {key}: " + ", ".join(f"{v:.3f}" for v in values))
    (args.input_dir / "comparison.json").write_text(json.dumps(result, indent=1))
    (args.input_dir / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
