"""Static bond-dimension model for exact matrix-product-state (MPS) execution.

The model needs no contraction. quimb's ``CircuitMPS(cutoff=0.0)`` keeps every
singular value, so applying a two-qubit gate across bond ``k`` sets

    b[k] = min(2 * b[k - 1], 2 * b[k + 1], 2 ** min(k + 1, n - k - 1))

(boundary bonds count as 1). Single-qubit gates leave bonds unchanged. A gate on
distant qubits is applied by swapping one qubit next to the other, applying the
gate, and swapping back, so every bond in between sees the same update twice.
On the 48- to 128-qubit memory-wall families this reproduces the recorded
maximum bond exactly (8, 16, 32, 128 and 512).
"""


def mps_bond_profile(n_qubits, operations):
    """Return the bond size after the circuit for each of the ``n - 1`` cuts."""
    bonds = [1] * max(n_qubits - 1, 0)
    caps = [1 << min(k + 1, n_qubits - k - 1) for k in range(len(bonds))]

    def grow(k):
        left = bonds[k - 1] if k > 0 else 1
        right = bonds[k + 1] if k < len(bonds) - 1 else 1
        bonds[k] = min(2 * left, 2 * right, caps[k])

    for gate in operations:
        if len(gate.qubits) != 2:
            continue
        low, high = sorted(gate.qubits)
        for k in range(low, high - 1):
            grow(k)
        grow(high - 1)
        for k in range(high - 2, low - 1, -1):
            grow(k)
    return tuple(bonds)


def mps_max_bond(n_qubits, operations):
    """Return the largest bond the exact MPS reaches while applying the circuit."""
    # Intermediate bonds are never larger than the final ones: bonds only grow.
    return max(mps_bond_profile(n_qubits, operations), default=1)


def mps_payload_bytes(n_qubits, bond):
    """Upper bound on the MPS tensor payload: ``n`` tensors of ``2 * bond**2`` complex numbers."""
    return 16 * 2 * n_qubits * bond * bond
