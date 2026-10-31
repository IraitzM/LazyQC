"""Conservative observable support analysis for unitary, product-input circuits."""

from .ir import Gate, Pauli


def cone_indices(qubits: tuple[tuple[int, ...], ...], support):
    """Return live wires and retained operation indices for a backward support cone.

    Only gate wires matter, so programs sharing a topology share the result.
    """
    live = set(support)
    retained = []
    for index in range(len(qubits) - 1, -1, -1):
        if live.intersection(qubits[index]):
            live.update(qubits[index])
            retained.append(index)
    return tuple(sorted(live)), tuple(reversed(retained))


def causal_cone(operations: tuple[Gate, ...], observable: Pauli):
    """Extract the conservative backward causal cone for a Pauli observable."""
    wires, retained = cone_indices(
        tuple(g.qubits for g in operations), (q for q, _ in observable.factors)
    )
    return wires, tuple(operations[i] for i in retained)
