import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_runner_saves_raw_runs_and_valid_timing_boundaries(tmp_path):
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_initial.py"),
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(output.read_text())
    assert len(data["runs"]) == len(data["first_calls"]) == 4
    assert data["metadata"]["packages"]["qulacs"]
    assert len(data["metadata"]["source_sha256"]) == 64
    assert len(data["summary"]) == 4
    for row in data["runs"]:
        assert row["max_abs_error"] < 1e-10
        phases = [
            "construction_ns",
            "planning_ns",
            "lowering_ns",
            "evolution_ns",
            "expectation_ns",
            "assembly_ns",
        ]
        assert all(row[key] >= 0 for key in phases)
        assert sum(row[key] for key in phases) <= row["total_ns"]
        assert row["allocations"] == row["evolutions"]


def test_runner_fails_instead_of_reporting_incorrect_golden_values(tmp_path):
    data = json.loads((ROOT / "benchmarks/initial.json").read_text())
    data["case"]["requests"][0]["expected"] = -0.9
    fixture = tmp_path / "incorrect.json"
    fixture.write_text(json.dumps(data))
    output = tmp_path / "should_not_exist.json"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_initial.py"),
            "--fixture",
            str(fixture),
            "--repeats",
            "1",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "AssertionError" in process.stderr
    assert not output.exists()
