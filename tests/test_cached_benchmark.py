"""Exercise recorded cold misses, warm hits and a changing-parameter negative control."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cached_benchmark_artifacts(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_cached.py"),
            "--repeats",
            "2",
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
    data = json.loads((tmp_path / "evaluation.json").read_text())
    training = json.loads((tmp_path / "training.json").read_text())
    assert len(data["cases"]) == 12
    assert len(data["runs"]) == 192
    assert len(data["first_calls"]) == 96
    assert data["metadata"]["profile_id"] == training["profile_id"]
    for row in data["runs"]:
        assert row["max_abs_error"] < 1e-10
        assert row["allocations"] == row["evolutions"] > 0
        assert row["total_ns"] >= sum(
            row[k] for k in ("construction_ns", "planning_ns", "execution_ns")
        )
        if row["mode"] == "calibrated_cache_cold":
            assert row["cache_status"] == "miss"
        elif row["mode"] in ("calibrated_cache", "heuristic_cache"):
            assert row["cache_status"] == (
                "miss" if row["case"] == "changing_binding_6q" else "hit"
            )
        else:
            assert row["cache_status"] == "disabled"
    for stats in data["cache_statistics"].values():
        for info in stats.values():
            assert info["hits"] + info["misses"] == 3
            assert info["entries"] <= info["max_entries"] == 4
    for summary in data["summary"]:
        metrics = summary["metrics"]
        assert summary["cache_total_speedup"] == pytest.approx(
            metrics["calibrated"]["median_ns"]["total_ns"]
            / metrics["calibrated_cache"]["median_ns"]["total_ns"]
        )
        assert (
            metrics["calibrated"]["selected_strategies"]
            == metrics["calibrated_cache"]["selected_strategies"]
        )
    stream = next(s for s in data["summary"] if s["family"] == "changing_binding")
    assert stream["estimated_extra_repeats_to_repay_first_miss"] is None
    assert len(data["oracles"]["changing_binding_6q"]["expected_by_repetition"]) == 3
