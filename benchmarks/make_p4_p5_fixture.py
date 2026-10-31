"""Reproduce the small P4/P5 validation manifest, without executing or tuning workloads."""

import json
import math
import random
from pathlib import Path

from lazyqc import Program

ROOT = Path(__file__).resolve().parents[1]


def record(name, p, observables, width, *, max_cuts=2, **extra):
    return {
        "name": name,
        "qubits": p.n_qubits,
        "initial_state": "zero",
        "operations": [
            {
                "gate": g.name,
                "qubits": g.qubits,
                **({"angle_radians": g.angle} if g.angle is not None else {}),
            }
            for g in p.resolved_operations()
        ],
        "requests": [
            {"id": str(i), "observable": [{"qubit": q, "pauli": a} for q, a in o.items()]}
            for i, o in enumerate(observables)
        ],
        "fragment_width": width,
        "max_cuts": max_cuts,
        **extra,
    }


def main():
    cases = []
    p = Program(2).h(0).cx(0, 1)
    cases.append(
        record("bell_1q_fragments", p, [{0: "Z"}, {1: "Z"}, {0: "X", 1: "X"}, {0: "Y", 1: "Y"}], 1)
    )
    p = Program(3).ry(0, 0.4).h(1).cx(0, 1).ry(1, 0.7).cx(1, 2).rx(2, 0.3)
    cases.append(
        record(
            "two_cut_chain", p, [{0: "X", 2: "Y"}, {0: "Z", 1: "Z", 2: "Z"}, {1: "X"}, {2: "Z"}], 1
        )
    )
    for placement, edges in (
        ("path", [(0, 1), (1, 2), (2, 3)]),
        ("star", [(0, 1), (0, 2), (0, 3)]),
        ("ring_negative", [(0, 1), (1, 2), (2, 3), (3, 0)]),
    ):
        p = Program(4)
        for q in range(4):
            p = p.h(q)
        for a, b in edges:
            p = p.cx(a, b).rz(b, 0.37).cx(a, b)
        for q in range(4):
            p = p.rx(q, 0.53)
        cases.append(
            record(
                f"qaoa_{placement}",
                p,
                [{a: "Z", b: "Z"} for a, b in edges],
                3 if placement == "star" else 2,
            )
        )
    rng = random.Random(9107)
    p = Program(5)
    for q in range(5):
        p = p.ry(q, rng.uniform(-1, 1)).rz(q, rng.uniform(-1, 1))
    for a, b in ((0, 1), (3, 4), (1, 2), (2, 3)):
        p = p.cx(a, b).ry(b, rng.uniform(-1, 1))
    obs = [{i % 5: "XYZ"[i % 3], (i + 2) % 5: "XYZ"[(i + 1) % 3]} for i in range(16)]
    cases.append(record("local_pauli_batch16", p, obs, 3))
    cases.append(
        record("changing_angle_encoding", p, obs[:4], 3, binding_deltas=[0.0, 0.23, -0.31])
    )
    p = Program(8)
    for q in range(8):
        p = p.ry(q, 0.17 + 0.03 * q)
    for q in range(7):
        p = p.cx(q, q + 1).rz(q + 1, 0.31)
    cases.append(
        record(
            "overlapping_uncut_batch16",
            p,
            [{q: axis} for q in range(8) for axis in "XZ"],
            8,
            max_cuts=0,
        )
    )
    p = Program(8)
    for q in range(8):
        p = p.ry(q, 0.13 + q * 0.07)
    cases.append(
        record(
            "disjoint_batch64", p, [{i % 8: "XYZ"[(i // 8) % 3]} for i in range(64)], 1, max_cuts=0
        )
    )
    p = Program(4)
    for _ in range(3):
        for q in range(4):
            p = p.rx(q, rng.uniform(-2, 2)).ry(q, rng.uniform(-2, 2))
        for a, b in ((0, 1), (1, 2), (2, 3), (0, 3)):
            p = p.cx(a, b)
    cases.append(record("dense_negative", p, [{0: "Z", 3: "X"}], 2))
    # Non-adjacent commuting phase cancellation, inherited from the earlier ZX pilot.
    p = Program(6)
    for q in range(6):
        p = p.ry(q, 0.21 + q * 0.03)
    for q in range(5):
        p = p.cx(q, q + 1)
    for q in range(6):
        p = p.rz(q, math.pi / 4).rz(q, math.pi / 8)
    for q in reversed(range(5)):
        p = p.cx(q, q + 1)
    cases.append(record("clifford_phase_network", p, [{q: "Z"} for q in range(6)], 3))
    fixture = {
        "schema_version": 1,
        "rotation_convention": "R_P(theta) = exp(-i * theta * P / 2)",
        "amplitude_index": "sum(bit[q] * 2**q)",
        "absolute_tolerance": 1e-10,
        "seed": 9107,
        "purpose": "Frozen small validation suite, not a scalability study",
        "cut_limits": {
            "max_branches": 100,
            "max_cut_candidates": 32,
            "beam_width": 4,
            "max_search_states": 256,
            "soft_timeout_ms": 1000,
        },
        "cases": cases,
    }
    (ROOT / "tests/fixtures/p4_p5.json").write_text(json.dumps(fixture, indent=2) + "\n")


if __name__ == "__main__":
    main()
