"""Validate benchmark measurements and retained negative controls, not speed claims."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyzx")
ROOT = Path(__file__).resolve().parents[1]


def test_zx_benchmark(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_zx.py"),
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
    data = json.loads((tmp_path / "evaluation.json").read_text())
    assert len(data["runs"]) == len(data["first_calls"]) == 48
    assert len(data["cases"]) == 8
    assert data["metadata"]["pyzx"]
    for row in data["runs"]:
        assert row["max_abs_error"] < 1e-10
        assert row["total_ns"] >= sum(
            row[k] for k in ("construction_ns", "planning_ns", "execution_ns")
        )
        assert row["planning_ns"] >= row["optimization_ns"]
        assert row["allocations"] == row["evolutions"]
        if row["mode"] == "zx_cached":
            assert row["cache_status"] == "hit"
            assert row["optimization_ns"] == 0 < row["optimization_build_ns"]
    for summary in data["summary"]:
        m = summary["metrics"]
        assert summary["cold_zx_speedup"] == pytest.approx(
            m["baseline"]["median_ns"]["total_ns"] / m["zx"]["median_ns"]["total_ns"]
        )
    phase = data["plans"]["commuting_phases_14q/zx"]
    assert phase["optimization"]["records"][0]["status"] == "verified"
    assert (
        phase["optimization"]["records"][0]["zx_gates"]
        < phase["optimization"]["records"][0]["local_gates"]
    )
    assert any(c["optimization"] == "none" for c in phase["alternatives"])
    assert any(c["optimization"] == "zx" for c in phase["alternatives"])
    assert (
        data["oracles"]["clean_uncomputation"]["expected"]
        != data["oracles"]["omitted_uncomputation_control"]["expected"]
    )


def test_invalid_benchmark_tolerance(tmp_path):
    fixture = json.loads((ROOT / "benchmarks/zx.json").read_text())
    fixture["absolute_tolerance"] = -1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(fixture))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_zx.py"),
            "--fixture",
            str(path),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "tolerance must be finite and positive" in result.stderr
    assert not (tmp_path / "out").exists()
