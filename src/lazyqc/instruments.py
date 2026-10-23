"""Exact local projection branches. These are physical tasks, not unitary Program gates."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Projector:
    qubits: tuple[int, ...]
    outcome: int

    def __post_init__(self):
        """Validate the projector outcome and its single target wire."""
        object.__setattr__(self, "qubits", tuple(self.qubits))
        if len(self.qubits) != 1 or type(self.qubits[0]) is not int or self.qubits[0] < 0:
            raise ValueError("A projector acts on one nonnegative qubit index")
        if type(self.outcome) is not int or self.outcome not in (0, 1):
            raise ValueError("Projector outcome must be 0 or 1")

    @property
    def name(self):
        """Return the backend gate name corresponding to this projector."""
        return f"P{self.outcome}"
