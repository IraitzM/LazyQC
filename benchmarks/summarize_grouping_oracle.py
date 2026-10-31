#!/usr/bin/env python3
"""Summarize exhaustive finite grouping-oracle results by seed split."""

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def geomean(values):
    values = tuple(values)
    return math.exp(sum(math.log(value) for value in values) / len(values))


def generated_best(case):
    medians = case["policy_medians_ns"]
    return min(medians[s]["collection_ns"] for s in ("full", "separate", "grouped"))


def summarize(payload, split_by="seed"):
    case_summaries = []
    for case in payload["cases"]:
        auto = case["policy_medians_ns"]["auto"]
        best = case["best_enumerated_collection_ns"]
        planning = auto["planning_ns"]
        execution = auto["collection_ns"]
        case_summaries.append(
            {
                **case,
                "auto_total_regret_at_R32": (planning + 32 * execution) / (32 * best),
            }
        )

    aggregates = []
    if split_by == "seed":
        splits = [
            (split, [case for case in case_summaries if case["seed_offset"] == offset])
            for split, offset in (("development", 0), ("heldout", 202))
        ]
    else:
        families = sorted({case["family"] for case in case_summaries})
        splits = [
            (f"heldout family {family}", [c for c in case_summaries if c["family"] == family])
            for family in families
        ]
    for split, cases in splits:
        aggregates.append(
            {
                "split": split,
                "case_count": len(cases),
                "candidate_count_median": statistics.median(
                    case["candidate_count"] for case in cases
                ),
                "enumeration_ms_median": statistics.median(
                    case["enumeration_ns"] for case in cases
                )
                / 1e6,
                "auto_graph_coverage": sum(
                    case["policy_graph_coverage"]["auto"] for case in cases
                ),
                "geomean_auto_execution_regret": geomean(
                    case["auto_collection_regret"] for case in cases
                ),
                "geomean_auto_total_regret_R1": geomean(
                    case["auto_total_regret_at_R1"] for case in cases
                ),
                "geomean_auto_total_regret_R5": geomean(
                    case["auto_total_regret_at_R5"] for case in cases
                ),
                "geomean_auto_total_regret_R32": geomean(
                    case["auto_total_regret_at_R32"] for case in cases
                ),
                "auto_execution_wins": sum(
                    case["auto_collection_regret"] <= 1 for case in cases
                ),
                "auto_R5_wins": sum(case["auto_total_regret_at_R5"] <= 1 for case in cases),
                # Ranking: auto against the best of its own generated candidates.
                # Coverage: that best against the best enumerated partition.
                "geomean_ranking_regret": geomean(
                    case["policy_medians_ns"]["auto"]["collection_ns"] / generated_best(case)
                    for case in cases
                ),
                "geomean_coverage_regret": geomean(
                    generated_best(case) / case["best_enumerated_collection_ns"]
                    for case in cases
                ),
            }
        )

    return {
        "split_by": split_by,
        "source_artifact": payload["metadata"].get(
            "artifact", "results/qml_workloads/grouping_oracle.json"
        ),
        "finite_portfolio": payload["metadata"]["oracle_scope"],
        "raw_rows": payload["metadata"]["raw_row_count"],
        "failures": len(payload["metadata"]["failures"]),
        "max_abs_error": payload["metadata"]["maximum_absolute_error"],
        "repetitions": payload["metadata"]["repeats"],
        "collections_per_plan": payload["metadata"]["collections_per_plan"],
        "enumeration_cost_treatment": payload["metadata"]["enumeration_cost_treatment"],
        "aggregates": aggregates,
        "cases": case_summaries,
    }


def render(summary):
    lines = [
        "# Exhaustive Small-Instance Grouping Reference",
        "",
        (
            f"Raw rows: {summary['raw_rows']}; numerical failures: "
            f"{summary['failures']}; maximum absolute error: "
            f"{summary['max_abs_error']:.3e}."
        ),
        "",
        (
            "The finite portfolio contains every set partition of five outputs, with one "
            "causal-cone task per partition block, plus an explicit full-state plan. It "
            "excludes cuts, rewrites, approximation and parallel workers. Ratios compare "
            "the current auto plan with the fastest measured candidate median in this "
            "declared finite set; the reference is not a global simulator optimum."
        ),
        "",
        (
            "Each measured plan is collected five times per timing block, eight blocks "
            "per case. `R` amortizes the auto planning median over repeated collections "
            "of the same bound batch. Exhaustive enumeration time is reported separately; "
            "it is not treated as a zero-cost selection algorithm. Execution regret "
            "factors into ranking (auto against the best of the full, separate and "
            "grouped candidates it generates) times coverage (that candidate against "
            "the best enumerated partition)."
        ),
        "",
        (
            "| Split | Cases | Candidates/case | Enumeration ms median | "
            "Auto graph covered | Auto execution / best | Ranking | Coverage | "
            "Auto total R=1 | R=5 | R=32 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["aggregates"]:
        lines.append(
            f"| {row['split']} | {row['case_count']} | {row['candidate_count_median']} | "
            f"{row['enumeration_ms_median']:.2f} | "
            f"{row['auto_graph_coverage']}/{row['case_count']} | "
            f"{row['geomean_auto_execution_regret']:.3f} | "
            f"{row['geomean_ranking_regret']:.3f} | "
            f"{row['geomean_coverage_regret']:.3f} | "
            f"{row['geomean_auto_total_regret_R1']:.3f} | "
            f"{row['geomean_auto_total_regret_R5']:.3f} | "
            f"{row['geomean_auto_total_regret_R32']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The results measure a narrow no-cut same-state grouping space on "
                "generated circuits. "
                + (
                    "Seed holdout is not family holdout. "
                    if summary.get("split_by", "seed") == "seed"
                    else "Both families were added after the planner was frozen. "
                )
                + "The best "
                "measured candidate is subject to minimum-selection timing bias across "
                "52–53 candidates."
            ),
            "",
            f"See `{Path(summary['source_artifact']).name}` for raw repetitions, "
            "candidate IDs, and per-case results.",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "results/qml_workloads/grouping_oracle.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/qml_workloads")
    parser.add_argument("--split-by", choices=("seed", "family"), default="seed")
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    payload["metadata"]["artifact"] = str(args.input.resolve().relative_to(ROOT))
    result = summarize(payload, args.split_by)
    stem = args.input.stem
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{stem}_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / f"{stem}_SUMMARY.md").write_text(render(result))
    print(render(result))


if __name__ == "__main__":
    main()
