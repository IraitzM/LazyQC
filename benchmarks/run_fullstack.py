"""E8 full-stack pilot: causal cones + grouping + ZX + cuts + cache + workers.

One matrix over two workload families where different features carry the win:

- ham12: 12q layered ansatz, 64 overlapping Pauli terms. Cuts are outside
  the bounded search (recorded NoFeasiblePlan); cones + grouping + cache
  carry the result. ZX and workers run as measured ablations, not assumptions.
- qaoa_path6: 6q path-placement QAOA, fragment cap 3q, up to 2 CX cuts.
  Joint cut search + cache reuse carry the result; dense (B0) and separate
  (B1) run as eager controls.

Eager = cold per-collection planning, no cache reuse. Lazy = plan once,
collect R times against a warm PlanningCache. All setup is charged to the
side that incurs it. A dense NumPy oracle checks every value.

Usage:
  uv run --locked --all-extras python benchmarks/run_fullstack.py \
      --repeats 3 --collections 10 --output results/fullstack_pilot.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import median
from time import perf_counter_ns

for key in ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_ham12(rng):
    from lazyqc import Program

    n, layers = 12, 3
    gates = []
    for _ in range(layers):
        for q in range(n):
            gates.append(("ry", q, rng.uniform(-math.pi, math.pi)))
        for a in range(n - 1):
            gates.append(("cx", a, a + 1))
    p = Program(n)
    for g in gates:
        p = p.ry(g[1], g[2]) if g[0] == "ry" else p.cx(g[1], g[2])
    terms, seen = [], set()
    while len(terms) < 64:
        center = rng.randrange(n)
        radius = rng.choice((0, 0, 1, 1, 2))
        obs = {
            q: rng.choice("XYZ")
            for q in range(max(0, center - radius), min(n, center + radius + 1))
        }
        key = tuple(sorted(obs.items()))
        if key in seen:
            continue
        seen.add(key)
        terms.append(obs)
    return p, terms, gates


def build_qaoa_path6():
    from lazyqc import Program

    n, edges = 6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    p = Program(n)
    for q in range(n):
        p = p.h(q)
    for a, b in edges:
        p = p.cx(a, b).rz(b, 0.37).cx(a, b)
    for q in range(n):
        p = p.rx(q, 0.53)
    terms = [{q: "Z"} for q in range(n)] + [
        {a: "Z", b: "Z"} for a, b in edges[:2]
    ]
    return p, terms


def oracle_values(n, operations, terms):
    """Dense statevector oracle via stride loops; supports H/RX/RY/RZ/CX."""
    import numpy as np

    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0
    mats = {
        "H": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    def single(mat, q):
        step = 1 << q
        for base in range(0, 1 << n, 2 * step):
            for i in range(base, base + step):
                a, b = state[i], state[i + step]
                state[i] = mat[0, 0] * a + mat[0, 1] * b
                state[i + step] = mat[1, 0] * a + mat[1, 1] * b

    def rot(axis, q, theta):
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        if axis == "X":
            single(np.array([[c, -1j * s], [-1j * s, c]]), q)
        elif axis == "Y":
            single(np.array([[c, -s], [s, c]]), q)
        else:
            single(np.array([[c - 1j * s, 0], [0, c + 1j * s]]), q)

    for g in operations:
        if g.name == "CX":
            a, b = g.qubits
            for i in range(1 << n):
                if (i >> a) & 1 and not (i >> b) & 1:
                    j = i | (1 << b)
                    state[i], state[j] = state[j], state[i]
        elif g.name in mats:
            single(mats[g.name], g.qubits[0])
        elif g.name in ("RX", "RY", "RZ"):
            rot(g.name[1], g.qubits[0], g.angle)
        else:  # pragma: no cover - closed gate set
            raise ValueError(g.name)
    out = []
    for obs in terms:
        v = state.copy()
        for q, axis in obs.items():
            if axis != "I":
                save = state.copy()
                state[:] = v
                single(mats[axis], q)
                v = state.copy()
                state[:] = save
        out.append(float(np.real(np.vdot(state, v))))
    return out


def plan_stats(plan):
    """Extract mechanism counters from explain(): search states, unique tasks."""
    try:
        info = plan.explain()
    except Exception:
        return {}
    stats = {"cache_status": info.get("cache_status")}
    search = info.get("search")
    if isinstance(search, dict) and "states" in search:
        stats["search_states"] = search["states"]
    dag = info.get("dag")
    if isinstance(dag, dict) and "experiment_nodes" in dag:
        stats["unique_experiments"] = len(dag["experiment_nodes"])
    groups = info.get("groups")
    if isinstance(groups, list):
        stats["groups"] = len(groups)
    candidate = info.get("selected_candidate")
    if isinstance(candidate, dict):
        for key in ("evolutions", "unique_tasks", "tasks", "experiments"):
            if key in candidate:
                stats[f"candidate_{key}"] = candidate[key]
    return stats


def collect_mode(program, terms, options, collections):
    """Plan once, collect R times; return totals in ms and values."""
    from lazyqc import plan_all

    requests = [program.expectation(t) for t in terms]
    t0 = perf_counter_ns()
    plan = plan_all(requests, **options)
    planning_ms = (perf_counter_ns() - t0) / 1e6
    stats = plan_stats(plan)
    collect_ms, values = [], None
    for _ in range(collections):
        t0 = perf_counter_ns()
        values = list(plan.collect().values)
        collect_ms.append((perf_counter_ns() - t0) / 1e6)
    return planning_ms, collect_ms, values, stats


def run_case(name, program, terms, modes, repeats, collections, oracle):
    from lazyqc import NoFeasiblePlan

    rows = []
    order = list(modes)
    for rep in range(repeats):
        rng = random.Random(1000 + rep)
        rep_order = order[:]
        rng.shuffle(rep_order)
        for mode in rep_order:
            options = dict(modes[mode])
            cache = options.pop("cache_factory", None)
            if cache is not None:
                options["cache"] = cache()
            try:
                planning_ms, collect_ms, values, stats = collect_mode(
                    program, terms, options, collections
                )
                err = max(abs(a - b) for a, b in zip(values, oracle))
                rows.append(
                    {
                        "case": name,
                        "mode": mode,
                        "rep": rep,
                        "feasible": True,
                        "planning_ms": planning_ms,
                        "collect_ms": collect_ms,
                        "lifetime_ms": planning_ms + sum(collect_ms),
                        "max_abs_error": err,
                        "plan_stats": stats,
                    }
                )
            except NoFeasiblePlan as exc:
                rows.append(
                    {
                        "case": name,
                        "mode": mode,
                        "rep": rep,
                        "feasible": False,
                        "reason": str(exc)[:160],
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--collections", type=int, default=10)
    parser.add_argument("--output", default="results/fullstack_pilot.json")
    args = parser.parse_args()

    from lazyqc import CutOptions, PlanningCache

    rng = random.Random(42)
    ham_program, ham_terms, _ = build_ham12(rng)
    qaoa_program, qaoa_terms = build_qaoa_path6()

    ham_oracle = oracle_values(
        ham_program.n_qubits, ham_program.resolved_operations(), ham_terms
    )
    qaoa_oracle = oracle_values(
        qaoa_program.n_qubits, qaoa_program.resolved_operations(), qaoa_terms
    )

    cut = lambda search: CutOptions(  # noqa: E731
        max_fragment_qubits=3,
        max_cuts=2,
        search=search,
        soft_timeout_ms=15000.0,
    )
    ham_modes = {
        "eager_separate_cold": {"strategy": "separate"},
        "lazy_grouped_cache": {
            "strategy": "grouped",
            "cache_factory": lambda: PlanningCache(64),
        },
        "lazy_grouped_zx_cache": {
            "strategy": "grouped",
            "optimization": "zx",
            "cache_factory": lambda: PlanningCache(64),
        },
        "lazy_grouped_cache_w2": {
            "strategy": "grouped",
            "cache_factory": lambda: PlanningCache(64),
            "workers": 2,
            "total_state_bytes": 1 << 20,
        },
    }
    qaoa_modes = {
        "eager_dense_cold": {"strategy": "full"},
        "eager_separate_cold": {"strategy": "separate"},
        "lazy_b3_cache": {
            "cutting": cut("independent"),
            "cache_factory": lambda: PlanningCache(64),
        },
        "lazy_b5_cache": {
            "cutting": cut("joint"),
            "cache_factory": lambda: PlanningCache(64),
        },
        "lazy_b5_zx_cache": {
            "cutting": cut("joint"),
            "optimization": "zx",
            "cache_factory": lambda: PlanningCache(64),
        },
    }

    rows = run_case(
        "ham12_64terms", ham_program, ham_terms, ham_modes,
        args.repeats, args.collections, ham_oracle,
    )
    rows += run_case(
        "qaoa_path6", qaoa_program, qaoa_terms, qaoa_modes,
        args.repeats, args.collections, qaoa_oracle,
    )

    summary = {}
    for row in rows:
        key = (row["case"], row["mode"])
        summary.setdefault(
            key,
            {"feasible": 0, "infeasible": 0, "lifetimes": [], "errors": [],
             "warm": [], "planning": [], "stats": []},
        )
        entry = summary[key]
        if row["feasible"]:
            entry["feasible"] += 1
            entry["lifetimes"].append(row["lifetime_ms"])
            entry["errors"].append(row["max_abs_error"])
            entry["warm"].append(median(row["collect_ms"][1:]))
            entry["planning"].append(row["planning_ms"])
            entry["stats"].append(row["plan_stats"])
        else:
            entry["infeasible"] += 1
    table = [
        {
            "case": case,
            "mode": mode,
            "feasible_runs": entry["feasible"],
            "infeasible_runs": entry["infeasible"],
            "median_lifetime_ms": median(entry["lifetimes"]) if entry["lifetimes"] else None,
            "median_planning_ms": median(entry["planning"]) if entry["planning"] else None,
            "median_warm_collect_ms": median(entry["warm"]) if entry["warm"] else None,
            "worst_error": max(entry["errors"]) if entry["errors"] else None,
            "plan_stats_rep0": entry["stats"][0] if entry["stats"] else None,
        }
        for (case, mode), entry in summary.items()
    ]

    payload = {
        "config": {
            "repeats": args.repeats,
            "collections": args.collections,
            "threads": "1 per runtime",
        },
        "summary": table,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    for entry in table:
        print(entry)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
