"""Gradient descent with a plan built once, against planning again at every step.

A parameter-shift Jacobian evaluates the circuit at 2P shifted angle sets. The
structure never changes between steps, only the angles, so the plan is built from
symbolic parameters and `collect(bindings=...)` supplies the values of each step.
Values are identical to planning from scratch.

    uv run python examples/08_plan_once_training.py --qubits 12 16 20 --layers 2 --steps 10
"""

import argparse
import math
from dataclasses import replace
from time import perf_counter

import numpy as np

from lazyqc import Pauli, Program, parameter_shift_requests, plan_all


def ansatz(n, layers, rng):
    program = Program(n)
    for _ in range(layers):
        for q in range(n):
            program = program.ry(q, rng.uniform(-1, 1))
        for q in range(0, n - 1, 2):
            program = program.cx(q, q + 1)
        for q in range(n):
            program = program.rz(q, rng.uniform(-1, 1))
        for q in range(1, n - 1, 2):
            program = program.cx(q, q + 1)
    return program


def descend(program, observables, targets, indices, steps, rate, collect):
    """Run `steps` updates; `collect(angles)` returns the (1 + 2P) x outputs value table."""
    angles = np.array([program.operations[i].angle for i in indices])
    n = len(observables)
    seconds, losses = [], []
    for _ in range(steps):
        start = perf_counter()
        table = collect(angles).reshape(1 + 2 * len(indices), n)
        forward, jacobian = table[0], 0.5 * (table[1::2] - table[2::2]).T
        angles = angles - rate * jacobian.T @ (2 * (forward - targets) / n)
        seconds.append(perf_counter() - start)
        losses.append(float(np.mean((forward - targets) ** 2)))
    return seconds, losses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, nargs="+", default=[12, 16, 20])
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    print(
        f"{'qubits':>6} {'params':>6} {'planning each step':>19} {'plan once':>10} {'speedup':>8}"
    )
    for n in args.qubits:
        rng = np.random.default_rng(n)
        program = ansatz(n, args.layers, rng)
        observables = [Pauli(((q, "Z"),)) for q in range(n)]
        targets = rng.uniform(-0.5, 0.5, n)
        indices = [i for i, g in enumerate(program.operations) if g.name in ("RY", "RZ")]
        sym, values = program.parametrize(indices)
        names = list(values)

        def fresh(angles):
            """Numeric shifted programs, built and planned again at every step."""
            operations = list(sym.bind(**dict(zip(names, angles))).resolved_operations())
            programs = [Program(n, tuple(operations))]
            for i in indices:
                for sign in (1, -1):
                    shifted = list(operations)
                    shifted[i] = replace(shifted[i], angle=shifted[i].angle + sign * math.pi / 2)
                    programs.append(Program(n, tuple(shifted)))
            requests = [p.expectation(o) for p in programs for o in observables]
            return np.array(plan_all(requests).collect().values)

        requests = parameter_shift_requests(sym, observables, indices)
        start = perf_counter()
        plan = plan_all(requests, symbolic=True)
        build = perf_counter() - start

        def rebind(angles):
            return np.array(plan.collect(bindings=dict(zip(names, angles))).values)

        t_fresh, l_fresh = descend(program, observables, targets, indices, args.steps, 0.2, fresh)
        t_once, l_once = descend(program, observables, targets, indices, args.steps, 0.2, rebind)
        assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in zip(l_fresh, l_once))
        total_once = build + sum(t_once)
        print(
            f"{n:>6} {len(indices):>6} {sum(t_fresh):>17.3f} s {total_once:>8.3f} s "
            f"{sum(t_fresh) / total_once:>7.2f}x"
        )


if __name__ == "__main__":
    main()
