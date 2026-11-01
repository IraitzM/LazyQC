"""QAOA MaxCut with native RZZ gates, against the same circuit translated to CX, RZ, CX.

Each cost layer applies RZZ(gamma) on every edge. Writing it as CX, RZ, CX triples the
two-qubit gate count; the native gate keeps one gate per edge. The cost function is
the sum of ZZ on the edges, evaluated per edge so cones stay local.

    uv run python examples/07_qaoa_native_gates.py --qubits 12 16 20 --layers 1 2 3
"""

import argparse
from statistics import median
from time import perf_counter

from lazyqc import Program, plan_all


def graph(n):
    """A ring with a chord to the opposite vertex: 3-regular for even n."""
    edges = {tuple(sorted((i, (i + 1) % n))) for i in range(n)}
    edges |= {tuple(sorted((i, (i + n // 2) % n))) for i in range(n // 2)}
    return sorted(edges)


def qaoa(n, layers, native):
    program = Program(n)
    for q in range(n):
        program = program.h(q)
    for layer in range(layers):
        gamma, beta = 0.35 + 0.1 * layer, 0.62 - 0.07 * layer
        for a, b in graph(n):
            if native:
                program = program.rzz(a, b, 2 * gamma)
            else:
                program = program.cx(a, b).rz(b, 2 * gamma).cx(a, b)
        for q in range(n):
            program = program.rx(q, 2 * beta)
    return program


def measure(n, layers, native, repeats, outputs):
    program = qaoa(n, layers, native)
    edges = graph(n)[:outputs] if outputs else graph(n)
    requests = [program.expectation({a: "Z", b: "Z"}) for a, b in edges]
    plan = plan_all(requests)
    plan.collect()  # warm-up
    plan_times, collect_times = [], []
    for _ in range(repeats):
        start = perf_counter()
        plan = plan_all(requests)
        plan_times.append(perf_counter() - start)
        start = perf_counter()
        result = plan.collect()
        collect_times.append(perf_counter() - start)
    executed = sum(len(t.operations) for t in plan.selected.tasks)
    return {
        "gates": len(program.operations),
        "executed": executed,
        "strategy": plan.selected.strategy,
        "width": plan.selected.max_width,
        "plan_ms": median(plan_times) * 1e3,
        "collect_ms": median(collect_times) * 1e3,
        "values": result.values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--outputs", type=int, default=0, help="read only the first edges (default: every edge)"
    )
    args = parser.parse_args()

    header = (
        f"{'n':>3} {'p':>2} {'form':>10} {'gates':>6} {'executed':>9} {'plan':>9} "
        f"{'width':>5} {'collect ms':>11} {'plan ms':>8}"
    )
    print(header)
    for n in args.qubits:
        for layers in args.layers:
            rows = {
                form: measure(n, layers, form == "native", args.repeats, args.outputs)
                for form in ("translated", "native")
            }
            for form, row in rows.items():
                print(
                    f"{n:>3} {layers:>2} {form:>10} {row['gates']:>6} {row['executed']:>9} "
                    f"{row['strategy']:>9} {row['width']:>5} {row['collect_ms']:>11.2f} "
                    f"{row['plan_ms']:>8.2f}"
                )
            error = max(
                abs(a - b) for a, b in zip(rows["native"]["values"], rows["translated"]["values"])
            )
            speedup = rows["translated"]["collect_ms"] / rows["native"]["collect_ms"]
            print(
                f"{'':>6} native/translated collect speedup {speedup:.2f}x, "
                f"max |difference| {error:.1e}"
            )


if __name__ == "__main__":
    main()
