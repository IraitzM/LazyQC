"""Conservative observable support analysis for unitary, product-input circuits."""

from .ir import Gate, Pauli


def causal_cone(operations: tuple[Gate, ...], observable: Pauli):
    """Extract the conservative backward causal cone for a Pauli observable."""
    live = {q for q, _ in observable.factors}
    retained = []
    for gate in reversed(operations):
        if live.intersection(gate.qubits):
            live.update(gate.qubits)
            retained.append(gate)
    return tuple(sorted(live)), tuple(reversed(retained))
