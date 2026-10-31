#!/usr/bin/env python3
"""Summarize the matched-basis IQM Garnet probes (swap and line placement).

For each probe file, reports per-plan compiled gates, station execution
windows, per-output estimates against the exact ideals, mean absolute error,
the attenuation slope (least-squares fit of measured on ideal through the
origin) and paired absolute-error differences with a bootstrap over the
seed-by-output cells. Writes ``results/qml_workloads/iqm_garnet_basis_probes_summary.json``.
"""

import argparse
import json
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/qml_workloads"
DEFAULT_INPUTS = {
    "swap": RESULTS / "iqm_garnet_basis_probe.json",
    "line": RESULTS / "iqm_garnet_basis_probe_line.json",
}
PLANS = ("full", "separate", "grouped")


def parse(value):
    match = re.search(r"\((\d+), (\d+), (\d+), (\d+), (\d+), (\d+), (\d+)", value)
    return datetime(*map(int, match.groups()), tzinfo=timezone.utc)


def execution_window(row):
    events = {
        e.get("status"): e.get("timestamp")
        for e in row.get("timeline") or []
        if e.get("source") == "iqm-station-control"
    }
    return (parse(events["execution_ended"]) - parse(events["execution_started"])).total_seconds()


def summarize(payload, seed=9701, draws=20000):
    runs = payload["runs"]
    cells = defaultdict(list)
    for row in runs:
        for out in row["outputs"]:
            key = (row["seed_offset"], f"{out['qubit']}{out['pauli']}")
            cells[(row["strategy"], key)].append((out["expectation"], out["ideal_expectation"]))
    keys = sorted({key for _, key in cells})
    estimate = {
        plan: {key: mean(v for v, _ in cells[(plan, key)]) for key in keys} for plan in PLANS
    }
    ideal = {key: cells[(PLANS[0], key)][0][1] for key in keys}

    plans = {}
    for plan in PLANS:
        plan_rows = [r for r in runs if r["strategy"] == plan]
        circuits = [m for m in plan_rows[0]["circuits"] if m["repetition_index"] == 0]
        errors = [abs(estimate[plan][k] - ideal[k]) for k in keys]
        plans[plan] = {
            "jobs": len(plan_rows),
            "circuits_per_job": plan_rows[0]["circuit_count"],
            "circuit_shots": sum(r["total_shots"] for r in plan_rows),
            "two_qubit_gates_per_circuit": sorted(
                {m["compiled_two_qubit_gates"] for m in circuits}
            ),
            "compiled_depths": sorted({m["compiled_depth"] for m in circuits}),
            "correction_swaps": sorted({len(m["output_correction_swaps"]) for m in circuits}),
            "execution_windows_s": {
                str(r["seed_offset"]): execution_window(r) for r in plan_rows
            },
            "mean_abs_error": mean(errors),
            "rmse": mean(e * e for e in errors) ** 0.5,
            "attenuation_slope": sum(estimate[plan][k] * ideal[k] for k in keys)
            / sum(ideal[k] ** 2 for k in keys),
        }
    for plan in PLANS:
        windows = plans[plan]["execution_windows_s"]
        plans[plan]["window_over_full"] = {
            s: windows[s] / plans["full"]["execution_windows_s"][s] for s in windows
        }

    rng = random.Random(seed)
    contrasts = {}
    for a, b in (("grouped", "full"), ("grouped", "separate"), ("separate", "full")):
        diffs = [
            abs(estimate[a][k] - ideal[k]) - abs(estimate[b][k] - ideal[k]) for k in keys
        ]
        boots = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(draws))
        contrasts[f"{a}_minus_{b}"] = {
            "mean_abs_error_difference": mean(diffs),
            "interval": [boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]],
            "cells_where_first_is_closer": sum(d < 0 for d in diffs),
            "cells": len(diffs),
        }
    return {
        "placement": payload["metadata"].get("placement", "swap"),
        "shots_per_output": len(cells[(PLANS[0], keys[0])])
        * payload["metadata"]["shots_per_circuit"],
        "plans": plans,
        "contrasts": contrasts,
        "cells": [
            {
                "seed_offset": k[0],
                "output": k[1],
                "ideal": ideal[k],
                **{plan: estimate[plan][k] for plan in PLANS},
            }
            for k in keys
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "iqm_garnet_basis_probes_summary.json"
    )
    args = parser.parse_args()
    summary = {
        name: {"source": str(path.relative_to(ROOT)), **summarize(json.loads(path.read_text()))}
        for name, path in DEFAULT_INPUTS.items()
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for name, s in summary.items():
        print(f"== {name} (shots per output {s['shots_per_output']})")
        for plan, p in s["plans"].items():
            print(
                f"  {plan:9s} 2q/circ={p['two_qubit_gates_per_circuit']} "
                f"depth={p['compiled_depths']} "
                f"swaps={p['correction_swaps']} windows={p['execution_windows_s']} "
                f"vs_full={ {k: round(v, 3) for k, v in p['window_over_full'].items()} } "
                f"MAE={p['mean_abs_error']:.3f} slope={p['attenuation_slope']:.3f}"
            )
        for name_c, c in s["contrasts"].items():
            print(
                f"  {name_c}: {c['mean_abs_error_difference']:+.3f} "
                f"({c['interval'][0]:+.3f}, {c['interval'][1]:+.3f}) "
                f"closer {c['cells_where_first_is_closer']}/{c['cells']}"
            )


if __name__ == "__main__":
    main()
