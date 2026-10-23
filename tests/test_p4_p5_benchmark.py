"""Ensure timing/failure records are trustworthy without asserting performance wins."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("qiskit_addon_cutting")
pytest.importorskip("pyzx")
pytest.importorskip("quimb")
ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_smoke(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_p4_p5.py"),
            "--repeats",
            "1",
            "--calibration-repeats",
            "1",
            "--cases",
            "bell_1q_fragments",
            "dense_negative",
            "--modes",
            "B0_full",
            "B1_cones",
            "B3_independent",
            "B4_tensor",
            "B5_joint",
            "B3_cached",
            "B5_cached",
            "workers2_cached",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads((tmp_path / "evaluation.json").read_text())
    assert data["metadata"]["numerical_cache"] is False
    assert data["metadata"]["manifest_sha256"]
    assert data["frontend"]["decoded_probabilities_and_leakage"] == pytest.approx(
        (0.5, 0.25, 0.25, 0), abs=1e-10
    )
    assert any(r["status"] == "infeasible_within_limits" for r in data["runs"])
    for row in data["runs"]:
        if row["status"] != "ok":
            assert row["reason"]
            continue
        assert row["max_abs_error"] < 1e-10
        assert row["total_ns"] == sum(
            row[k] for k in ("construction_ns", "planning_ns", "execution_ns")
        )
        if "allocations" in row:
            assert row["allocations"] == row["evolutions"] == row["unique_experiments"]
        if row["mode"].endswith("cached"):
            assert row["cache_status"] == "hit"
    assert data["summary"][0]["comparisons"]["B3_independent/B5_joint"]["paired_speedups"]


def test_tensor_baseline_noncommuting_axes_and_target_order():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from tensor_baseline import prepare

    from lazyqc import Program, plan_all

    p = Program(3).h(2).ry(0, 0.43).cx(2, 0).rx(1, 0.8).rz(2, 0.32)
    requests = tuple(
        p.expectation(o) for o in ({2: "Y", 0: "X"}, {1: "Y"}, {0: "Z", 2: "Y"}, {}, {2: "X"})
    )
    reference = plan_all(requests, strategy="full").collect().values
    plan = prepare(requests)
    assert plan.collect() == pytest.approx(reference, abs=1e-10)
    assert plan.collect() == pytest.approx(reference, abs=1e-10)
