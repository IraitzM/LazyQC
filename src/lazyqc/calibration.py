"""Opt-in synthetic calibration. Call before planning, with thread settings fixed."""

import random
from dataclasses import asdict, replace
from statistics import median
from time import perf_counter_ns

from .cost import features, fit_cost_model
from .ir import Gate, Pauli
from .plan import Candidate, Plan, Task


def calibrate(*, widths=(1, 2, 3, 5, 7, 9, 11, 13, 15, 17), repeats=7, seed=1729):
    """Benchmark representative circuits and return a calibrated execution profile.

    Return (model, raw report); includes setup, warmup and fit in calibration_ns.

    Imports and native startup are included if not already completed by the caller.
    Width coverage is a bound, not a guarantee for arbitrary depths or gate mixes.
    """
    start = perf_counter_ns()
    widths = tuple(widths)
    if (
        not widths
        or len(set(widths)) != len(widths)
        or any(type(w) is not int or not 1 <= w <= 20 for w in widths)
    ):
        raise ValueError("Calibration widths must be distinct integers between 1 and 20")
    if type(repeats) is not int or repeats < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    candidates = []
    for width in widths:
        for layers, count, mixed in (
            (0, 1, False),
            (0, 8, True),
            (1, 1, True),
            (1, 8, False),
            (3, 4, True),
            (3, 12, False),
        ):
            operations = []
            for _ in range(layers):
                operations.extend(
                    Gate(rng.choice(("RX", "RY", "RZ")), (q,), rng.uniform(-2, 2))
                    for q in range(width)
                )
                operations.extend(Gate("CX", (q, q + 1)) for q in range(width - 1))
            outputs = tuple(
                (
                    i,
                    Pauli(
                        tuple(
                            (q, rng.choice("XYZ") if mixed else "Z")
                            for q in rng.sample(range(width), min(width, 1 + i % 3))
                        )
                    ),
                )
                for i in range(count)
            )
            task = Task(tuple(range(width)), tuple(operations), outputs)
            candidates.append(Candidate("calibration", (task,)))
            if width <= 3:
                for copies in (2, 5):
                    tasks = tuple(
                        replace(task, outputs=tuple((j * count + i, obs) for i, obs in outputs))
                        for j in range(copies)
                    )
                    candidates.append(Candidate("calibration", tasks))
    plans = [
        Plan(
            c,
            (c,),
            sum(len(t.outputs) for t in c.tasks),
            (),
            "calibration",
            max(widths),
            16 * 2 ** max(widths),
            0,
        )
        for c in candidates
    ]
    # Warm every shape, but preserve first runs separately.
    first = [asdict(p.collect().stats) for p in plans]
    rows = []
    for repetition in range(repeats):
        order = list(range(len(plans)))
        rng.shuffle(order)
        for i in order:
            stats = plans[i].collect().stats
            rows.append({"candidate": i, "repetition": repetition, **asdict(stats)})
    targets = [
        median(r["execution_ns"] for r in rows if r["candidate"] == i)
        for i in range(len(candidates))
    ]
    model = fit_cost_model(candidates, targets)
    model = replace(model, calibration_ns=perf_counter_ns() - start)
    return model, {
        "seed": seed,
        "widths": widths,
        "repeats": repeats,
        "calibration_ns": model.calibration_ns,
        "profile_id": model.profile_id,
        "fit": "nonnegative least squares weighted by inverse measured execution time",
        "targets": targets,
        "features": [features(c) for c in candidates],
        "candidates": [asdict(c) for c in candidates],
        "first_calls": first,
        "runs": rows,
        "predicted_ns": [model.predict(c) for c in candidates],
    }
