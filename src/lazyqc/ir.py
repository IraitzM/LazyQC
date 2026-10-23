"""Immutable circuit descriptions and intermediate representation (IR).

This module defines the core data structures for quantum programs, gates, symbolic
parameters, Pauli observables, and deferred expectation requests.

Rotations follow the standard physics convention:
    .. math::
        R_P(\\theta) = \\exp\\left(-i \\frac{\\theta}{2} P\\right)
where :math:`P \\in \\{X, Y, Z\\}` and :math:`\\theta` is given in radians.

Effect annotations track basis-preservation and unitary characteristics, drawing
inspiration from quantum type and effect systems such as Silq.

References:
    - Nielsen, M. A., & Chuang, I. L. (2010). "Quantum Computation and Quantum Information".
      Cambridge University Press.
    - Bichsel, B., Baader, M., Gehr, T., & Vechev, M. (2020). "Silq: a high-level quantum
      language with safe uncomputation". In Proceedings of the 41st ACM SIGPLAN Conference
      on Programming Language Design and Implementation (PLDI '20), pp. 921-936.
    - McArdle, S., et al. (2020). "Quantum computational chemistry". Reviews of Modern
      Physics, 92(1), 015003. (Variational parameterized circuits).
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real

GATES = frozenset({"H", "X", "Y", "Z", "RX", "RY", "RZ", "CX"})
ROTATIONS = frozenset({"RX", "RY", "RZ"})


def finite_angle(value: Real) -> float:
    """Validate and convert an angular rotation value to a finite float.

    Args:
        value: The angle in radians, which must be a finite real number (not boolean).

    Returns:
        float: The validated angle in radians as a float.

    Raises:
        ValueError: If value is a boolean, not a Real instance, or not finite (NaN/inf).

    Examples:
        >>> finite_angle(3.14159)
        3.14159
        >>> finite_angle(0)
        0.0
    """
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError("Angles must be finite real numbers in radians")
    return float(value)


@dataclass(frozen=True)
class Parameter:
    """Symbolic parameter for parameterized quantum gates (e.g. in VQE or QAOA).

    Parameters are bound to concrete values at planning or execution time via
    `Program.bind(name=value)`.

    Args:
        name: A nonempty string identifying the parameter.

    Examples:
        >>> theta = Parameter("theta")
        >>> theta.name
        'theta'
    """
    name: str

    def __post_init__(self):
        """Reject empty or non-string symbolic parameter names."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a nonempty string")


@dataclass(frozen=True)
class Effects:
    """Operational effects and semantic properties of an operation.

    Inspired by quantum programming language type systems (such as Silq),
    tracking whether an operation is unitary, measurement-free, or preserves
    computational basis states.

    References:
        - Bichsel, B., et al. (2020). "Silq: a high-level quantum language with safe
          uncomputation". PLDI '20.

    Args:
        unitary: True if the operation is unitary.
        measurement_free: True if the operation contains no wave-function collapses.
        basis_preserving: True if the gate maps computational basis states to computational
            basis states (e.g., Pauli-Z, X, CX, RZ up to phase).
    """
    unitary: bool = True
    measurement_free: bool = True
    basis_preserving: bool = False


@dataclass(frozen=True)
class Gate:
    """An immutable quantum gate descriptor.

    Supported gates include:
        - 1-qubit Cliffords: 'H', 'X', 'Y', 'Z'
        - 1-qubit rotations: 'RX', 'RY', 'RZ' with angle in radians
        - 2-qubit entanglers: 'CX' (Controlled-X / CNOT)

    Args:
        name: Gate identifier string in {'H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CX'}.
        qubits: Tuple of nonnegative integer wire indices.
        angle: Angle in radians (for rotations) or a symbolic Parameter instance.

    Raises:
        ValueError: If gate name is unknown, wire count is invalid, or angle type mismatches.

    Examples:
        >>> g1 = Gate("H", (0,))
        >>> g2 = Gate("CX", (0, 1))
        >>> g3 = Gate("RZ", (0,), angle=1.57)
    """
    name: str
    qubits: tuple[int, ...]
    angle: float | Parameter | None = None

    def __post_init__(self):
        """Normalize and validate the gate's wires and rotation parameter."""
        object.__setattr__(self, "qubits", tuple(self.qubits))
        if self.name not in GATES:
            raise ValueError(f"Unsupported gate: {self.name}")
        arity = 2 if self.name == "CX" else 1
        if len(self.qubits) != arity or len(set(self.qubits)) != arity:
            raise ValueError(f"{self.name} needs {arity} distinct qubits")
        if any(type(q) is not int or q < 0 for q in self.qubits):
            raise ValueError("Gate indices must be nonnegative integers")
        if self.name in ROTATIONS:
            if not isinstance(self.angle, Parameter):
                object.__setattr__(self, "angle", finite_angle(self.angle))
        elif self.angle is not None:
            raise ValueError(f"{self.name} does not accept an angle")

    @property
    def effects(self) -> Effects:
        """Derive operational effects for this gate."""
        # This is a basis-support property, not a claim to implement Silq's qfree typing.
        return Effects(basis_preserving=self.name in {"X", "Y", "Z", "RZ", "CX"})

    def inverse(self) -> "Gate":
        """Compute the Hermitian adjoint / inverse of this gate.

        Returns:
            Gate: Inverted gate with negated angle for rotations, or self for self-inverse gates.

        Raises:
            ValueError: If the gate contains an unbound symbolic Parameter.
        """
        if isinstance(self.angle, Parameter):
            raise ValueError("Bind parameters before requesting an explicit inverse")
        return replace(self, angle=-self.angle) if self.name in ROTATIONS else self


@dataclass(frozen=True)
class VersionedOperation:
    """An operation annotated with SSA-like input and output wire version numbers.

    Used for causal dependency analysis and conservative backward cone extraction.

    Args:
        gate: The underlying Gate operation.
        inputs: Tuple of (qubit, version_in) pairs for each operated wire.
        outputs: Tuple of (qubit, version_out) pairs for each operated wire.
    """
    gate: Gate
    inputs: tuple[tuple[int, int], ...]
    outputs: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Pauli:
    """One Hermitian multi-qubit Pauli product string.

    Represents an operator of the form :math:`P = \\bigotimes_{q} P_q`, where
    :math:`P_q \\in \\{I, X, Y, Z\\}`. An empty product represents the identity operator.

    Args:
        factors: Sequence of `(qubit_index, axis)` pairs, where axis is in `{'I', 'X', 'Y', 'Z'}`.

    Raises:
        ValueError: If wire index is negative, axis is invalid, or a qubit is duplicated.

    Examples:
        >>> p = Pauli(((0, "Z"), (1, "X")))
        >>> p.factors
        ((0, 'Z'), (1, 'X'))
        >>> id_op = Pauli()
        >>> id_op.factors
        ()
    """

    factors: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        """Normalize factors and reject invalid or duplicate Pauli wires."""
        factors = tuple(tuple(f) for f in self.factors)
        if any(len(f) != 2 for f in factors):
            raise ValueError("Pauli factors must be (qubit, axis) pairs")
        if any(type(q) is not int or q < 0 or p not in {"I", "X", "Y", "Z"} for q, p in factors):
            raise ValueError("Invalid Pauli factor")
        if len({q for q, _ in factors}) != len(factors):
            raise ValueError("Specify each Pauli qubit only once")
        object.__setattr__(self, "factors", tuple(sorted((q, p) for q, p in factors if p != "I")))


@dataclass(frozen=True)
class Program:
    """Immutable quantum program description with named or indexed qubits.

    Initialized to the all-zero state :math:`|0\\dots 0\\rangle`. All gate application
    and parameter binding methods return new immutable `Program` instances.

    Args:
        qubits: Either an integer qubit count (generating names 'q0', 'q1', ...) or
            a sequence of distinct logical qubit names (strings).
        operations: Tuple of Gate operations in chronological execution order.
        bindings: Tuple of (parameter_name, float_value) bindings.

    Examples:
        >>> p = Program(2).h(0).cx(0, 1)
        >>> p.n_qubits
        2
        >>> len(p.operations)
        2
    """

    qubits: tuple[str, ...] | int
    operations: tuple[Gate, ...] = ()
    bindings: tuple[tuple[str, float], ...] = ()

    def __post_init__(self):
        """Normalize qubit names and validate operations and parameter bindings."""
        names = self.qubits
        if type(names) is int:
            if names < 1:
                raise ValueError("A program needs at least one qubit")
            names = tuple(f"q{i}" for i in range(names))
        elif isinstance(names, (tuple, list)):
            names = tuple(names)
        else:
            raise ValueError("Qubits must be a positive count or sequence of names")
        if not names or any(not isinstance(q, str) or not q for q in names):
            raise ValueError("Qubit names must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("Qubit names must be unique")
        ops = tuple(self.operations)
        if any(not isinstance(g, Gate) for g in ops):
            raise ValueError("Operations must be Gate descriptions")
        if any(q >= len(names) for g in ops for q in g.qubits):
            raise ValueError("Gate qubit is outside the program")
        bindings = tuple((k, finite_angle(v)) for k, v in self.bindings)
        if len({k for k, _ in bindings}) != len(bindings):
            raise ValueError("Duplicate parameter bindings")
        params = {g.angle.name for g in ops if isinstance(g.angle, Parameter)}
        if any(k not in params for k, _ in bindings):
            raise ValueError("Binding names must occur in the program")
        object.__setattr__(self, "qubits", names)
        object.__setattr__(self, "operations", ops)
        object.__setattr__(self, "bindings", tuple(sorted(bindings)))

    @property
    def n_qubits(self) -> int:
        """Return the number of logical qubits in the program."""
        return len(self.qubits)

    def index(self, qubit: int | str) -> int:
        """Resolve a logical qubit name or index to an integer wire index.

        Args:
            qubit: Either a string name or a nonnegative integer index.

        Returns:
            int: The 0-based integer wire index.

        Raises:
            ValueError: If the qubit name or index is not found in the program.
        """
        if isinstance(qubit, str):
            try:
                return self.qubits.index(qubit)
            except ValueError:
                raise ValueError(f"Unknown logical qubit: {qubit}") from None
        if type(qubit) is not int or not 0 <= qubit < self.n_qubits:
            raise ValueError(f"Invalid qubit index: {qubit}")
        return qubit

    def gate(self, name: str, *qubits: int | str, angle=None) -> "Program":
        """Append a generic gate to the program, returning a new immutable Program.

        Args:
            name: Gate identifier string in {'H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'CX'}.
            *qubits: Target qubit names or indices.
            angle: Rotation angle in radians or a symbolic Parameter (for RX, RY, RZ).

        Returns:
            Program: A new Program instance containing the appended gate.
        """
        gate = Gate(name, tuple(self.index(q) for q in qubits), angle)
        return replace(self, operations=(*self.operations, gate))

    def h(self, q: int | str) -> "Program":
        """Append a Hadamard gate :math:`H` on qubit `q`.

        .. math::
            H = \\frac{1}{\\sqrt{2}} \\begin{pmatrix} 1 & 1 \\\\ 1 & -1 \\end{pmatrix}

        Args:
            q: Target qubit name or integer index.

        Returns:
            Program: New Program with H appended.
        """
        return self.gate("H", q)

    def x(self, q: int | str) -> "Program":
        """Append a Pauli-X (NOT) gate on qubit `q`.

        Args:
            q: Target qubit name or integer index.

        Returns:
            Program: New Program with X appended.
        """
        return self.gate("X", q)

    def y(self, q: int | str) -> "Program":
        """Append a Pauli-Y gate on qubit `q`.

        Args:
            q: Target qubit name or integer index.

        Returns:
            Program: New Program with Y appended.
        """
        return self.gate("Y", q)

    def z(self, q: int | str) -> "Program":
        """Append a Pauli-Z (phase flip) gate on qubit `q`.

        Args:
            q: Target qubit name or integer index.

        Returns:
            Program: New Program with Z appended.
        """
        return self.gate("Z", q)

    def rx(self, q: int | str, angle: float | Parameter) -> "Program":
        """Append an X-axis rotation
        :math:`R_X(\\theta) = \\exp(-i \\frac{\\theta}{2} X)` on qubit `q`.

        Args:
            q: Target qubit name or integer index.
            angle: Angle in radians (float) or symbolic Parameter.

        Returns:
            Program: New Program with RX appended.
        """
        return self.gate("RX", q, angle=angle)

    def ry(self, q: int | str, angle: float | Parameter) -> "Program":
        """Append a Y-axis rotation
        :math:`R_Y(\\theta) = \\exp(-i \\frac{\\theta}{2} Y)` on qubit `q`.

        Args:
            q: Target qubit name or integer index.
            angle: Angle in radians (float) or symbolic Parameter.

        Returns:
            Program: New Program with RY appended.
        """
        return self.gate("RY", q, angle=angle)

    def rz(self, q: int | str, angle: float | Parameter) -> "Program":
        """Append a Z-axis rotation
        :math:`R_Z(\\theta) = \\exp(-i \\frac{\\theta}{2} Z)` on qubit `q`.

        Args:
            q: Target qubit name or integer index.
            angle: Angle in radians (float) or symbolic Parameter.

        Returns:
            Program: New Program with RZ appended.
        """
        return self.gate("RZ", q, angle=angle)

    def cx(self, control: int | str, target: int | str) -> "Program":
        """Append a Controlled-X (CNOT) gate with specified control and target qubits.

        Args:
            control: Control qubit name or integer index.
            target: Target qubit name or integer index.

        Returns:
            Program: New Program with CX appended.
        """
        return self.gate("CX", control, target)

    def bind(self, **values: float) -> "Program":
        """Bind named symbolic parameters to concrete floating-point angle values.

        Args:
            **values: Keyword mapping from parameter names to numeric values in radians.

        Returns:
            Program: New Program with updated parameter bindings.

        Examples:
            >>> p = Program(1).rx(0, Parameter("theta")).bind(theta=0.5)
            >>> p.bindings
            (('theta', 0.5),)
        """
        return replace(self, bindings=tuple((dict(self.bindings) | values).items()))

    def resolved_operations(self) -> tuple[Gate, ...]:
        """Return gate operations with all symbolic parameters substituted with bound values.

        Returns:
            tuple[Gate, ...]: Concrete Gate operations ready for simulation or planning.

        Raises:
            ValueError: If an operation references a Parameter that has not been bound.
        """
        values = dict(self.bindings)
        result = []
        for g in self.operations:
            if isinstance(g.angle, Parameter):
                if g.angle.name not in values:
                    raise ValueError(f"Unbound parameter: {g.angle.name}")
                g = replace(g, angle=values[g.angle.name])
            result.append(g)
        return tuple(result)

    @property
    def versioned_operations(self) -> tuple[VersionedOperation, ...]:
        """Conservative dependency versions; no automatic uncomputation inference.

        Returns:
            tuple[VersionedOperation, ...]: Operations annotated with input and output
                wire versions.
        """
        versions = [0] * self.n_qubits
        result = []
        for gate in self.operations:
            inputs = tuple((q, versions[q]) for q in gate.qubits)
            for q in gate.qubits:
                versions[q] += 1
            outputs = tuple((q, versions[q]) for q in gate.qubits)
            result.append(VersionedOperation(gate, inputs, outputs))
        return tuple(result)

    def expectation(self, observable: Pauli | Mapping[int | str, str]) -> "Request":
        """Construct a deferred expectation request for a Pauli observable on this program.

        Args:
            observable: Either a `Pauli` instance or a mapping from qubit name/index
                to a Pauli character in {'I', 'X', 'Y', 'Z'}.

        Returns:
            Request: A deferred expectation request :math:`\\langle \\psi | O | \\psi \\rangle`.

        Examples:
            >>> p = Program(2).h(0).cx(0, 1)
            >>> req = p.expectation({0: "Z", 1: "Z"})
            >>> req.observable.factors
            ((0, 'Z'), (1, 'Z'))
        """
        if isinstance(observable, Mapping):
            observable = Pauli(tuple((self.index(q), p) for q, p in observable.items()))
        return Request(self, observable)


@dataclass(frozen=True)
class Request:
    """A deferred quantum expectation request :math:`\\langle \\psi | O | \\psi \\rangle`.

    Consists of a `Program` preparing state :math:`|\\psi\\rangle` and a Hermitian `Pauli`
    operator :math:`O`. Computation does not occur when instantiated; requests are
    passed to `plan_all()` or `collect_all()` for batched optimization and execution.

    Args:
        program: The quantum circuit generating the state.
        observable: The Pauli product observable to evaluate.

    Raises:
        ValueError: If program or observable types are invalid, or observable wires exceed program.

    Examples:
        >>> p = Program(1).x(0)
        >>> req = p.expectation({0: "Z"})
        >>> isinstance(req, Request)
        True
    """
    program: Program
    observable: Pauli

    def __post_init__(self):
        """Validate the program and ensure observable wires are in range."""
        if not isinstance(self.program, Program) or not isinstance(self.observable, Pauli):
            raise ValueError("A request requires a Program and a Pauli observable")
        if any(q >= self.program.n_qubits for q, _ in self.observable.factors):
            raise ValueError("Observable qubit is outside the program")
