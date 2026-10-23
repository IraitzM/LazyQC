"""Setup-inclusive measurements must charge misses and preserve honest controls."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("pyzx")
pytest.importorskip("qiskit_addon_cutting")
ROOT = Path(__file__).resolve().parents[1]


def test_priority_benchmark_records_cumulative_cost_and_worker_choices(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_priorities.py"),
            "--collections",
            "4",
            "--seeds",
            "1",
            "--cases",
            "overlapping_uncut_batch16",
            "--unique-bindings",
            "2",
            "--worker-trials",
            "1",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "evaluation.json").read_text())
    for summary in data["summary"]:
        for mode, cumulative in summary["cumulative_ns"].items():
            rows = [r for r in data["runs"] if r["case"] == summary["case"] and r["mode"] == mode]
            assert cumulative[-1] == sum(r["total_ns"] for r in rows)
            assert rows[0]["cache_status"] == "miss"
            assert all(r["cache_status"] == "hit" for r in rows[1:])
            assert summary["cache_info"][mode]["accounted_bytes"] > 0
    assert all(r["max_abs_error"] < 1e-10 for r in data["runs"] + data["unique_bindings"])
    for workload in data["workers"].values():
        assert workload["transport_profiles"]["auto"]["startup_ns"] > 0
        for row in workload["runs"]:
            assert row["max_abs_error"] < 1e-10
            assert row["allocations"] == workload["unique_experiments"]
            if row["mode"].startswith("persistent"):
                assert row["pool_reused"] and row["pool_startup_ns"] == 0
