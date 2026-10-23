"""Load the versioned validation fixtures without executing them."""

import json
from pathlib import Path

from .ir import Gate, Pauli, Program

ROTATION_CONVENTION = "R_P(theta) = exp(-i * theta * P / 2)"


def load_fixture(path):
    """Load a JSON benchmark fixture from `path`."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported fixture schema")
    if data.get("rotation_convention") != ROTATION_CONVENTION:
        raise ValueError("Unsupported rotation convention")
    if data.get("amplitude_index") != "sum(bit[q] * 2**q)":
        raise ValueError("Unsupported amplitude ordering")
    cases = data.get("cases")
    if cases is None:
        cases = [data["case"]]
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("Cases must have unique names and not be empty")
    return data, cases


def build_requests(case):
    """Build immutable expectation requests from a fixture description."""
    if case["initial_state"] != "zero":
        raise ValueError("Only zero product-state preparation is supported")
    program = Program(
        case["qubits"],
        tuple(
            Gate(op["gate"], tuple(op["qubits"]), op.get("angle_radians"))
            for op in case["operations"]
        ),
    )
    return tuple(
        program.expectation(
            Pauli(tuple((factor["qubit"], factor["pauli"]) for factor in request["observable"]))
        )
        for request in case["requests"]
    )
