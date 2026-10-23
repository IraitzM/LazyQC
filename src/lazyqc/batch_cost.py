"""Instrument-aware execution calibration, separated from held-out batch workloads."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import median
from time import perf_counter_ns

from .cost import environment, source_digest
from .instruments import Projector


@lru_cache(maxsize=1)
def _implementation_hash():
    """Return a digest of batch implementation files used by calibration."""
    return source_digest("batch_executor.py", "batch_cost.py", "instruments.py")


def batch_environment():
    """Return the runtime environment fingerprint for batch calibration."""
    result = environment()
    result["batch_executor_sha256"] = _implementation_hash()
    return tuple(sorted(result.items()))


def task_features(task):
    """Return structural features used by the batch duration model."""
    size = 1 << task.width
    projections = sum(isinstance(g, Projector) for g in task.operations)
    gates = len(task.operations) - projections
    return (
        1,
        gates,
        size,
        size * gates,
        projections,
        size * projections,
        len(task.outputs),
        size * len(task.outputs),
    )


@dataclass(frozen=True)
class BatchCostModel:
    coefficients: tuple[float, ...]
    reconstruction_term_ns: float
    reconstruction_factor_ns: float
    environment_items: tuple[tuple[str, str], ...]
    max_width: int
    calibration_ns: int
    planning_ns_per_expansion: float | None = None
    scheduling_ns_per_task: float | None = None

    def __post_init__(self):
        """Normalize and validate batch calibration coefficients and metadata."""
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(
            self, "environment_items", tuple(tuple(item) for item in self.environment_items)
        )
        if len(self.coefficients) != 8 or any(
            not math.isfinite(v) or v < 0
            for v in (
                *self.coefficients,
                self.reconstruction_term_ns,
                self.reconstruction_factor_ns,
            )
        ):
            raise ValueError("Invalid batch cost coefficients")
        if self.planning_ns_per_expansion is not None and (
            not math.isfinite(self.planning_ns_per_expansion)
            or self.planning_ns_per_expansion <= 0
        ):
            raise ValueError("Invalid planning cost per expansion")
        if self.scheduling_ns_per_task is not None and (
            not math.isfinite(self.scheduling_ns_per_task) or self.scheduling_ns_per_task < 0
        ):
            raise ValueError("Invalid scheduling cost per task")
        if type(self.max_width) is not int or not 1 <= self.max_width <= 24:
            raise ValueError("Invalid calibrated width")
        if type(self.calibration_ns) is not int or self.calibration_ns < 0:
            raise ValueError("Invalid calibration duration")

    def validate_environment(self):
        """Raise when the current batch runtime differs from the profile runtime."""
        if self.environment_items != batch_environment():
            raise ValueError("Batch calibration environment mismatch; recalibrate")

    @property
    def profile_id(self):
        """Return the stable identifier of this batch calibration profile."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def task_ns(self, task):
        """Predict one task's execution time in nanoseconds."""
        if task.width > self.max_width:
            raise ValueError(
                "Batch task outside calibrated width; extend calibration or omit cost_model"
            )
        return max(
            1.0, sum(a * b for a, b in zip(self.coefficients, task_features(task), strict=True))
        )

    def reconstruction_ns(self, terms, factors):
        """Predict reconstruction time from term and factor counts."""
        return self.reconstruction_term_ns * terms + self.reconstruction_factor_ns * factors

    def save(self, path):
        """Serialize this batch calibration profile to JSON."""
        Path(path).write_text(
            json.dumps({"schema_version": 1, **asdict(self)}, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, path):
        """Load and environment-check a batch calibration profile from JSON."""
        data = json.loads(Path(path).read_text())
        if data.pop("schema_version") != 1:
            raise ValueError("Unsupported batch profile schema")
        data.setdefault("planning_ns_per_expansion", None)
        data.setdefault("scheduling_ns_per_task", None)
        result = cls(**data)
        result.validate_environment()
        return result


def calibrate_planning(*, repeats=3):
    """Fit nominal planning cost per joint beam expansion on tiny batches.

    Runs unbudgeted joint searches (planning only, no execution) on small
    synthetic circuits that are disjoint from the evaluation workloads, and
    fits the median marginal planning time per beam expansion. Returns the
    fitted ns-per-expansion rate and a details dict. Timing varies run to
    run; the fitted constant is fixed once per profile, so searches using it
    stay deterministic.
    """
    from .cutting import CutOptions
    from .ir import Program
    from .planner import plan_all

    start = perf_counter_ns()
    if type(repeats) is not int or repeats < 1:
        raise ValueError("Use positive repeats")
    chains = Program(3).ry(0, 0.3).cx(0, 1).ry(1, 0.5).cx(1, 2)
    path = (
        Program(4).h(0).h(1).h(2).h(3).cx(0, 1).cx(1, 2).cx(2, 3).rz(1, 0.4).rz(2, 0.7)
    )
    star = Program(4).h(0).h(1).h(2).h(3).cx(0, 1).cx(0, 2).cx(0, 3).rz(0, 0.6)
    specs = [
        ([chains.expectation({q: "Z"}) for q in range(3)], 2, 1),
        (
            [
                path.expectation({0: "Z", 1: "Z"}),
                path.expectation({1: "Z", 2: "Z"}),
                path.expectation({2: "Z", 3: "Z"}),
            ],
            2,
            2,
        ),
        (
            [
                star.expectation({0: "Z", 1: "Z"}),
                star.expectation({0: "Z", 2: "Z"}),
                star.expectation({0: "Z", 3: "Z"}),
            ],
            2,
            2,
        ),
    ]
    samples, skipped = [], []
    for bi, (requests, fragment_width, max_cuts) in enumerate(specs):
        batch_samples = []
        for _ in range(repeats):
            plan = plan_all(
                requests,
                cutting=CutOptions(max_fragment_qubits=fragment_width, max_cuts=max_cuts),
                optimization="zx",
                strategy="auto",
                workers=1,
                max_state_qubits=fragment_width,
                max_state_bytes=16 * 2**fragment_width,
            )
            analysis = dict(plan.analysis_stats)
            marginal_ns = plan.planning_ns - analysis["incumbent_build_ns"]
            if plan.search_states > 0 and marginal_ns > 0:
                batch_samples.append(marginal_ns / plan.search_states)
        if batch_samples:
            samples.extend(batch_samples)
        else:
            skipped.append(bi)
    if not samples:
        raise ValueError("Calibration batches explored no joint states")
    rate = median(samples)
    details = {
        "repeats": repeats,
        "batches": len(specs),
        "skipped_batches": skipped,
        "samples": len(samples),
        "median_ns_per_expansion": rate,
        "min_ns_per_expansion": min(samples),
        "max_ns_per_expansion": max(samples),
        "calibration_ns": perf_counter_ns() - start,
    }
    return rate, details


def calibrate_batch(*, widths=(1, 2, 4, 6, 8, 10), repeats=5, extra_mix=False):
    """Benchmark batch tasks and fit a nonnegative instrument-aware cost model.

    With ``extra_mix``, an H/RZ QAOA-like task family is added alongside the
    default RY/CX family to test whether gate-type diversity reduces bias on
    cut fragment workloads. Opt-in: default profiles are unchanged.
    """
    import random

    import numpy as np
    from scipy.optimize import nnls

    from .batch_executor import _one
    from .ir import Gate, Pauli
    from .plan import Task

    start = perf_counter_ns()
    if (
        not widths
        or any(type(w) is not int or not 1 <= w <= 16 for w in widths)
        or type(repeats) is not int
        or repeats < 1
    ):
        raise ValueError("Use widths 1..16 and positive repeats")
    tasks = []
    for width in widths:
        for layers, projections, outputs in (
            (0, 0, 1),
            (2, 0, 4),
            (6, 0, 2),
            (1, 1, 1),
            (3, 2, 4),
            (6, 3, 2),
        ):
            ops = []
            for layer in range(layers):
                ops.extend(Gate("RY", (q,), 0.29 + 0.1 * layer) for q in range(width))
                ops.extend(Gate("CX", (q, q + 1)) for q in range(width - 1))
            ops.extend(Projector((i % width,), i % 2) for i in range(projections))
            tasks.append(
                Task(
                    tuple(range(width)),
                    tuple(ops),
                    tuple((i, Pauli(((i % width, "XYZ"[i % 3]),))) for i in range(outputs)),
                )
            )
    if extra_mix:
        for width in widths:
            for layers, projections, outputs in (
                (1, 0, 2),
                (2, 1, 1),
                (3, 2, 3),
            ):
                ops = []
                for layer in range(layers):
                    ops.extend(Gate("H", (q,)) for q in range(width))
                    ops.extend(Gate("CX", (q, q + 1)) for q in range(width - 1))
                    ops.extend(Gate("RZ", (q,), 0.41 + 0.13 * layer) for q in range(width))
                ops.extend(Projector((i % width,), (i + 1) % 2) for i in range(projections))
                tasks.append(
                    Task(
                        tuple(range(width)),
                        tuple(ops),
                        tuple((i, Pauli(((i % width, "XYZ"[(i + 1) % 3]),)))
                              for i in range(outputs)),
                    )
                )
    first = [_one(t)[1:4] for t in tasks]
    rows = []
    rng = random.Random(4401)
    for repeat in range(repeats):
        order = list(range(len(tasks)))
        rng.shuffle(order)
        for i in order:
            row = _one(tasks[i])
            rows.append(
                {
                    "task": i,
                    "repetition": repeat,
                    "lowering_ns": row[1],
                    "evolution_ns": row[2],
                    "expectation_ns": row[3],
                }
            )
    target = np.array(
        [
            median(
                sum(r[k] for k in ("lowering_ns", "evolution_ns", "expectation_ns"))
                for r in rows
                if r["task"] == i
            )
            for i in range(len(tasks))
        ]
    )
    matrix = np.array([task_features(t) for t in tasks], dtype=float)
    scale = np.maximum(matrix.max(axis=0), 1)
    coeffs, _ = nnls(matrix / scale / target[:, None], np.ones(len(target)), maxiter=1000)
    # Fit the actual fsum/product reconstruction loop on synthetic scalar tables.
    rrows = []
    for terms, factors in ((32, 1), (32, 4), (128, 2), (128, 6)):
        values = tuple(0.17 + 0.01 * i for i in range(factors))
        samples = []
        for _ in range(7):
            mark = perf_counter_ns()
            math.fsum(0.5 * math.prod(values[j] for j in range(factors)) for _ in range(terms))
            samples.append(perf_counter_ns() - mark)
        rrows.append((terms, terms * factors, median(samples)))
    ra = np.array([r[:2] for r in rrows], dtype=float)
    rb = np.array([r[2] for r in rrows])
    rc, _ = nnls(ra / rb[:, None], np.ones(len(rb)))
    # Fit serial dispatch overhead per task through the real execute path.
    # A throwaway candidate over a width spread of calibration tasks isolates
    # the loop/record cost that price() must charge per collection.
    from .batch import BatchCandidate, BatchPlan
    from .batch_executor import _execute_batch
    from .cutting import CutOptions

    probe_tasks = [tasks[i] for i in range(0, len(tasks), max(1, len(tasks) // 8))][:8]
    probe = BatchCandidate("calibration-probe", tuple(probe_tasks), (), ())
    sched_samples = []
    for _ in range(3):
        probe_plan = BatchPlan(
            selected=probe,
            alternatives=(probe,),
            request_count=0,
            constant_outputs=(),
            max_state_qubits=16,
            max_state_bytes=256 * 2**20,
            workers=1,
            total_state_bytes=256 * 2**20,
            cutting=CutOptions(max_fragment_qubits=8),
            planning_ns=0,
            search_states=0,
            search_exhausted=False,
        )
        stats = _execute_batch(probe_plan).stats
        sched_samples.append(stats.scheduling_ns / len(probe_tasks))
    scheduling_rate = median(sched_samples)
    model = BatchCostModel(
        tuple(float(v) for v in coeffs / scale),
        float(rc[0]),
        float(rc[1]),
        batch_environment(),
        max(widths),
        perf_counter_ns() - start,
        None,
        float(scheduling_rate),
    )
    return model, {
        "seed": 4401,
        "widths": widths,
        "repeats": repeats,
        "extra_mix": extra_mix,
        "tasks": [asdict(t) for t in tasks],
        "first_calls": first,
        "runs": rows,
        "targets_ns": target.tolist(),
        "reconstruction_samples": rrows,
        "scheduling_ns_per_task_samples": sched_samples,
        "profile_id": model.profile_id,
        "calibration_ns": model.calibration_ns,
    }
