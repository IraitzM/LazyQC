"""Benchmark artifacts must be reproducible and contain every validated timing row."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_calibrated_runner(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_calibrated.py"),
            "--repeats",
            "1",
            "--calibration-repeats",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    evaluation = json.loads((tmp_path / "evaluation.json").read_text())
    training = json.loads((tmp_path / "training.json").read_text())
    profile = json.loads((tmp_path / "profile.json").read_text())
    assert len(evaluation["runs"]) == len(evaluation["cases"]) * 5 == 75
    assert len(evaluation["first_calls"]) == 75
    assert len(training["runs"]) == 96
    assert profile["calibration_ns"] == training["calibration_ns"]
    assert evaluation["metadata"]["profile_id"] == training["profile_id"]
    for row in evaluation["runs"]:
        assert row["max_abs_error"] < 1e-10
        assert row["allocations"] == row["evolutions"]
        assert row["total_ns"] >= sum(
            row[k] for k in ("construction_ns", "planning_ns", "execution_ns")
        )
    for summary in evaluation["summary"]:
        metrics = summary["metrics"]
        assert summary["calibrated_vs_heuristic_speedup"] == pytest.approx(
            metrics["heuristic"]["median_ns"]["total_ns"]
            / metrics["calibrated"]["median_ns"]["total_ns"]
        )
    # Freeze definitions without oracle values before fitting, and label the older family.
    manifest = evaluation["metadata"]["precalibration_manifest"]
    assert all("expected" not in case for case in manifest)
    assert sum(case["family"] == "pair_regression" for case in manifest) == 5
    oracles = {c["oracle"] for c in evaluation["cases"]}
    assert oracles == {"analytic", "dense_numpy", "qulacs_full_consistency"}
