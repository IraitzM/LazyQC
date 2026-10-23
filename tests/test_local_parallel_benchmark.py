"""Independent reference validation and real worker/payload accounting checks."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from lazyqc.ir import Gate

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    # Benchmark scripts import their sibling helper modules.
    if str(ROOT / "benchmarks") not in sys.path:
        sys.path.insert(0, str(ROOT / "benchmarks"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vector_oracle_matches_dense_matrices_with_noncommuting_gates():
    vector = load("vector", ROOT / "benchmarks/local_parallel_support.py")
    dense = load("dense", ROOT / "tests/reference_numpy.py")
    rng = np.random.default_rng(9011)
    for width in (2, 3, 4):
        gates = []
        for _ in range(4):
            for q in range(width):
                gates.extend(Gate("R" + axis, (q,), rng.uniform(-2, 2)) for axis in "XYZ")
            gates.append(Gate("CX", tuple(rng.choice(width, 2, replace=False).tolist())))
        actual = vector.numpy_state(width, gates)
        expected = dense.statevector(width, [(g.name, g.qubits, g.angle) for g in gates])
        np.testing.assert_allclose(actual, expected, atol=1e-13)
        for factors in ((), ((0, "Y"),), ((0, "X"), (width - 1, "Z"))):
            assert (
                abs(
                    vector.numpy_expectation(actual, factors) - dense.expectation(expected, factors)
                )
                < 1e-13
            )


def test_real_worker_benchmark_preserves_oracle_payload_and_timing_scopes(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/run_local_parallel.py"),
            "--smoke",
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = json.loads((tmp_path / "evaluation.json").read_text())
    summary_module = load("parallel_summary", ROOT / "benchmarks/summarize_local_parallel.py")
    aggregate = summary_module.summarize([result])
    assert aggregate["timed_warm_collections"] == 26
    assert aggregate["validated_collections"] == 40
    for configs in aggregate["cases"][0]["configurations"].values():
        if configs[0]["workers"] > 1:
            assert configs[0]["lifetime"]["collections"] == 3
    case = result["cases"][0]
    assert len(case["dag"]["tasks"]) == 2
    assert case["planning_ns"] > 0
    assert result["metadata"]["calibration_wall_ns"] > 0
    for config in case["configurations"]:
        for row in [config["first"], *config["runs"], config["resources"]["collection"]]:
            assert row["max_abs_error"] < 1e-10
            assert row["stats"]["allocations"] == 2
            assert row["stats"]["peak_active_state_bytes"] <= config["total_state_bytes"]
            if config["payload_slots"] == 1:
                assert row["stats"]["peak_in_flight"] <= 1
        assert config["resources"]["sample_count"] >= 2
        assert config["resources"]["sampled_peak_sum_rss_bytes"] > 0
        if config["workers"] > 1:
            assert config["warmup_ns"] >= config["transport_profile"]["startup_ns"]
            assert config["shutdown_ns"] > 0
            assert config["first"]["stats"]["pool_reused"]


def test_summary_charges_setup_and_requires_sustained_break_even():
    summary = load("parallel_summary", ROOT / "benchmarks/summarize_local_parallel.py")
    assert summary.sustained_crossing([5, 10, 15], [7, 9, 16]) is None
    assert summary.sustained_crossing([5, 10, 15], [7, 9, 12]) == 2
    assert summary.cumulative(100, [20, 30, 40]) == [120, 150, 190]
    assert summary.paired_interval([2, 2, 2], 0) == [2, 2]
