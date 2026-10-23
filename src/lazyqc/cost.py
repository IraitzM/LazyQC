"""Explicit, machine-specific execution cost profiles; prediction never runs a backend."""

import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

FEATURES = (
    "collection",
    "tasks",
    "wires",
    "gates",
    "observables",
    "pauli_factors",
    "state_amplitudes",
    "one_qubit_work",
    "two_qubit_work",
    "diagonal_expectation_work",
    "mixed_expectation_work",
)
THREAD_KEYS = ("QULACS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def features(candidate):
    """Return the public feature tuple used to predict candidate execution cost."""
    return candidate.cost_features


def _compute_features(candidate):
    """Compute structural cost features without requiring an environment profile."""
    row = [1.0] + [0.0] * (len(FEATURES) - 1)
    for task in candidate.tasks:
        size = 1 << task.width
        diagonal = sum(all(p == "Z" for _, p in obs.factors) for _, obs in task.outputs)
        one = sum(len(g.qubits) == 1 for g in task.operations)
        values = (
            1,
            task.width,
            len(task.operations),
            len(task.outputs),
            sum(len(obs.factors) for _, obs in task.outputs),
            size,
            size * one,
            size * (len(task.operations) - one),
            size * diagonal,
            size * (len(task.outputs) - diagonal),
        )
        for i, value in enumerate(values, 1):
            row[i] += value
    return tuple(row)


def source_digest(*names):
    """Return the SHA-256 of the named package source files, in order."""
    digest = hashlib.sha256()
    for name in names:
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _static_environment():
    """Return stable environment fields that affect calibrated timings."""
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "qulacs": version("qulacs"),
        "numpy": version("numpy"),
        "implementation_sha256": source_digest("backends/qulacs.py", "executor.py", "cost.py"),
    }


def environment():
    """Return the complete runtime environment fingerprint for calibration checks."""
    return {**_static_environment(), **{key: os.environ.get(key, "") for key in THREAD_KEYS}}


@dataclass(frozen=True)
class CalibratedCostModel:
    coefficients: tuple[float, ...]
    environment_items: tuple[tuple[str, str], ...]
    min_width: int
    max_width: int
    calibration_ns: int
    schema_version: int = 1

    def __post_init__(self):
        """Normalize profile mappings and validate their numeric contents."""
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "environment_items", tuple(sorted(self.environment_items)))
        if self.schema_version != 1:
            raise ValueError("Unsupported calibration schema")
        if len(self.coefficients) != len(FEATURES) or any(
            not math.isfinite(v) or v < 0 for v in self.coefficients
        ):
            raise ValueError("Cost coefficients must be finite, nonnegative and match features")
        if (
            type(self.min_width) is not int
            or type(self.max_width) is not int
            or not 1 <= self.min_width <= self.max_width <= 30
        ):
            raise ValueError("Invalid calibration width domain")
        if type(self.calibration_ns) is not int or self.calibration_ns < 0:
            raise ValueError("Invalid calibration duration")
        if len(dict(self.environment_items)) != len(self.environment_items) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.environment_items
        ):
            raise ValueError("Invalid calibration environment")

    @property
    def profile_id(self):
        """Return the stable identifier of this calibration profile."""
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def validate_environment(self):
        """Raise when the current runtime differs from the calibrated environment."""
        current = environment()
        if dict(self.environment_items) != current:
            raise ValueError("Calibration environment mismatch; recalibrate in this environment")

    def covers(self, candidate):
        """Return whether the profile contains features required by a candidate."""
        return not candidate.tasks or (
            self.min_width <= candidate.min_width and candidate.max_width <= self.max_width
        )

    def predict(self, candidate):
        """Predict candidate execution time in nanoseconds."""
        if not self.covers(candidate):
            raise ValueError("Candidate width outside calibration domain")
        return sum(a * b for a, b in zip(self.coefficients, features(candidate), strict=True))

    def save(self, path):
        """Serialize this calibration profile to a JSON file."""
        Path(path).write_text(
            json.dumps({**asdict(self), "features": FEATURES}, indent=2, allow_nan=False) + "\n"
        )

    @classmethod
    def load(cls, path):
        """Load and validate a calibration profile from a JSON file."""
        data = json.loads(Path(path).read_text())
        if tuple(data.pop("features")) != FEATURES:
            raise ValueError("Calibration feature schema mismatch")
        data["environment_items"] = tuple(tuple(item) for item in data["environment_items"])
        model = cls(**data)
        model.validate_environment()
        return model


def fit_cost_model(candidates, durations_ns, *, calibration_ns=0):
    """Fit a calibrated cost model from candidates and measured durations.

    Relative-error weighted NNLS; targets are measured execution times, not scores.
    """
    import numpy as np
    from scipy.optimize import nnls

    candidates, durations_ns = tuple(candidates), tuple(durations_ns)
    if len(candidates) != len(durations_ns) or not candidates:
        raise ValueError("Provide matching nonempty candidates and timings")
    target = np.asarray(durations_ns, dtype=float)
    if not np.all(np.isfinite(target)) or np.any(target <= 0):
        raise ValueError("Timings must be finite and positive")
    matrix = np.asarray([features(c) for c in candidates])
    widths = [t.width for c in candidates for t in c.tasks]
    if not widths:
        raise ValueError("Calibration must include tasks")
    scale = np.maximum(np.max(matrix, axis=0), 1)
    # Weight each observation by 1/time so large circuits do not dominate the fit.
    coefficients, _ = nnls(matrix / scale / target[:, None], np.ones(len(target)), maxiter=1000)
    return CalibratedCostModel(
        tuple(float(v) for v in coefficients / scale),
        tuple(environment().items()),
        min(widths),
        max(widths),
        calibration_ns,
    )
