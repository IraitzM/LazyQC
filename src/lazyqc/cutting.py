"""Exact CX channel cuts, bounded cut search, and immutable reconstruction formulas."""

import hashlib
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache
from importlib.metadata import PackageNotFoundError, version
from itertools import chain, combinations, product
from math import fsum, isfinite, pi
from pathlib import Path

from .instruments import Projector
from .ir import Gate, Pauli


@dataclass(frozen=True)
class CutOptions:
    max_fragment_qubits: int = 8
    max_cuts: int = 2
    max_branches: int = 1000
    max_cut_candidates: int = 32
    max_fragments: int = 2048
    max_terms: int = 16384
    max_requests: int = 64
    max_group_size: int = 4
    beam_width: int = 4
    max_search_states: int = 256
    soft_timeout_ms: float = 5000.0
    search: str = "joint"
    reuse: bool = True
    partition: tuple[tuple[int, ...], ...] | None = None
    expected_collections: int | None = None

    def __post_init__(self):
        """Validate cutting limits, search mode, and optional wire partition."""
        if self.expected_collections is not None and (
            type(self.expected_collections) is not int or self.expected_collections < 1
        ):
            raise ValueError("expected_collections must be a positive integer or None")
        for name in (
            "max_fragment_qubits",
            "max_branches",
            "max_cut_candidates",
            "max_fragments",
            "max_terms",
            "max_requests",
            "max_group_size",
            "beam_width",
            "max_search_states",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_cuts) is not int or not 0 <= self.max_cuts <= 6:
            raise ValueError("max_cuts must be between 0 and 6")
        if (
            isinstance(self.soft_timeout_ms, bool)
            or not isfinite(self.soft_timeout_ms)
            or self.soft_timeout_ms <= 0
        ):
            raise ValueError("soft_timeout_ms must be finite and positive")
        if self.search not in ("joint", "independent", "fixed") or type(self.reuse) is not bool:
            raise ValueError("Choose joint, independent or fixed search and boolean reuse")
        if self.partition is not None:
            parts = tuple(tuple(p) for p in self.partition)
            wires = [q for part in parts for q in part]
            if (
                not parts
                or any(not part for part in parts)
                or len(set(wires)) != len(wires)
                or any(type(q) is not int or q < 0 for q in wires)
            ):
                raise ValueError("partition must contain nonempty disjoint nonnegative wire groups")
            object.__setattr__(self, "partition", parts)


@dataclass(frozen=True)
class Recipe:
    width: int
    operations: tuple[Gate | Projector, ...]

    @cached_property
    def _hash(self):
        """Return the memoized structural hash of this recipe."""
        return hash((self.width, self.operations))

    def __hash__(self):
        """Return the recipe's structural hash."""
        return self._hash

    def __reduce__(self):
        """Return a process-safe reconstruction tuple for this recipe."""
        # Python string hashes are process-specific. Recompute memoized hashes
        # after a plan crosses an isolation/process boundary.
        return type(self), (self.width, self.operations)

    @cached_property
    def sort_key(self):
        """Return a deterministic ordering key for recipe deduplication."""
        return (
            self.width,
            tuple((g.name, g.qubits, getattr(g, "angle", None)) for g in self.operations),
        )


@dataclass(frozen=True)
class ValueRef:
    recipe: Recipe
    observable: Pauli


@dataclass(frozen=True)
class Term:
    coefficient: float
    factors: tuple[ValueRef, ...]


@dataclass(frozen=True)
class Formula:
    output: int
    terms: tuple[Term, ...]


@dataclass(frozen=True)
class GroupPlan:
    formulas: tuple[Formula, ...]
    cut_indices: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    optimization: str
    qpd_terms: int
    branches: int

    @cached_property
    def _hash(self):
        """Return the memoized structural hash of this group plan."""
        return hash((self.formulas, self.cut_indices, self.components, self.optimization))

    def __hash__(self):
        """Return the group plan's structural hash."""
        return self._hash

    def __reduce__(self):
        """Return a process-safe reconstruction tuple for this group plan."""
        return type(self), (
            self.formulas,
            self.cut_indices,
            self.components,
            self.optimization,
            self.qpd_terms,
            self.branches,
        )

    @property
    def outputs(self):
        """Return request indices produced by this group."""
        return tuple(f.output for f in self.formulas)


class CutLimitError(ValueError):
    pass


@lru_cache(maxsize=1)
def cutting_identity():
    """Return the source and optional-package identity for cutting caches."""
    try:
        dependency = version("qiskit-addon-cutting")
    except PackageNotFoundError as exc:
        raise ImportError("Install exact cutting support: uv sync --extra cutting") from exc
    return (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        + f"/qiskit-addon-cutting-{dependency}"
    )


@lru_cache(maxsize=1)
def cx_branches():
    """Use the upstream six-term CX basis; enumerate signed measurement outcomes.

    This yields ten deterministic branch combinations, not nine exact experiments.
    Nine is the sampling overhead gamma**2, not an exact enumeration count.
    """
    from qiskit.circuit.library import CXGate
    from qiskit_addon_cutting.qpd import QPDBasis

    basis = QPDBasis.from_instruction(CXGate())

    def expand(operations, wire):
        """Expand one upstream decomposition side into signed IR branches."""
        branches = [(1.0, ())]
        for op in operations:
            if op.name == "qpd_measure":
                branches = [
                    (sign * (1 if bit == 0 else -1), gates + (Projector((wire,), bit),))
                    for sign, gates in branches
                    for bit in (0, 1)
                ]
                continue
            if op.name in ("h", "x", "y", "z"):
                gate = Gate(op.name.upper(), (wire,))
            elif op.name in ("s", "sdg"):
                gate = Gate("RZ", (wire,), pi / 2 if op.name == "s" else -pi / 2)
            else:
                raise ValueError(f"Unsupported CX decomposition operation: {op.name}")
            branches = [(sign, gates + (gate,)) for sign, gates in branches]
        return branches

    result = []
    for coefficient, sides in zip(basis.coeffs, basis.maps, strict=True):
        for (sa, a), (sb, b) in product(expand(sides[0], 0), expand(sides[1], 1)):
            result.append((float(coefficient) * sa * sb, a + b))
    if len(basis.maps) != 6 or len(result) != 10 or abs(basis.kappa - 3) > 1e-12:
        raise ValueError("Unsupported upstream CX decomposition schema")
    return tuple(result)


def components(task, cuts):
    """Return connected wire components after removing the selected CX gates."""
    parents = {q: q for q in task.wires}

    def root(q):
        """Find the representative of a wire's union-find component."""
        while parents[q] != q:
            q = parents[q]
        return q

    for i, gate in enumerate(task.operations):
        if i not in cuts and len(gate.qubits) == 2:
            a, b = map(root, gate.qubits)
            parents[b] = a
    groups = {}
    for q in task.wires:
        groups.setdefault(root(q), []).append(q)
    return tuple(sorted(tuple(g) for g in groups.values()))


def cut_choices(task, options, diagnostics=None):
    """Enumerate bounded gate-removal proposals; actual channel maps replace gates later."""
    indices = tuple(i for i, g in enumerate(task.operations) if g.name == "CX")
    if options.partition is not None:
        labels = {q: i for i, part in enumerate(options.partition) for q in part}
        if not set(task.wires) <= labels.keys():
            raise ValueError("partition does not cover the task's logical wires")
        choices = [
            tuple(i for i in indices if len({labels[q] for q in task.operations[i].qubits}) > 1)
        ]
    else:
        # Seed contiguous partitions before individual occurrence combinations:
        # repeated boundary gates must be cut together to disconnect components.
        seeds = [()]
        width = options.max_fragment_qubits
        for shift in range(min(width, len(task.wires))):
            labels = {q: (j + shift) // width for j, q in enumerate(task.wires)}
            seeds.append(
                tuple(i for i in indices if len({labels[q] for q in task.operations[i].qubits}) > 1)
            )
        subsets = (
            choice
            for k in range(min(options.max_cuts, len(indices)) + 1)
            for choice in combinations(indices, k)
        )
        choices = chain(seeds, subsets)
    attempted = set()
    for cuts in choices:
        if cuts in attempted:
            continue
        if len(attempted) >= options.max_cut_candidates:
            if diagnostics is not None:
                diagnostics.append("Cut proposal budget exhausted; search is incomplete")
            break
        attempted.add(cuts)
        if len(cuts) > options.max_cuts or 10 ** len(cuts) > options.max_branches:
            continue
        parts = components(task, cuts)
        if max(map(len, parts), default=0) > options.max_fragment_qubits:
            continue
        # Cutting an edge whose endpoints remain connected cannot shrink a fragment.
        labels = {q: i for i, p in enumerate(parts) for q in p}
        if any(len({labels[q] for q in task.operations[i].qubits}) == 1 for i in cuts):
            continue
        yield cuts, parts


def compile_recipes(task, cuts, parts, options):
    """Compile branch preparations once; independent of requested observables."""
    if 10 ** len(cuts) > options.max_branches:
        raise CutLimitError("Exact branch cap exceeded")
    location = {q: (i, j) for i, part in enumerate(parts) for j, q in enumerate(part)}
    cut_slots = {gate_index: position for position, gate_index in enumerate(cuts)}
    branches = cx_branches() if cuts else ()

    def lower(gates):
        """Lower gates to fragment-local wire indices and reject crossing gates."""
        lowered = []
        for g in gates:
            group = location[g.qubits[0]][0]
            if any(location[q][0] != group for q in g.qubits):
                raise ValueError("Uncut gate crosses a fragment boundary")
            lowered.append((group, replace(g, qubits=tuple(location[q][1] for q in g.qubits))))
        return tuple(lowered)

    # Remap each gate/branch replacement once, not for every Cartesian product.
    ordinary, replacements = {}, {}
    for i, gate in enumerate(task.operations):
        if i in cut_slots:
            replacements[i] = tuple(
                lower(
                    tuple(replace(g, qubits=tuple(gate.qubits[q] for q in g.qubits)) for g in gates)
                )
                for _, gates in branches
            )
        else:
            ordinary[i] = lower((gate,))
    registry, result = {}, []
    for branch_ids in product(range(len(branches)), repeat=len(cuts)):
        coefficient = 1.0
        local_ops = [[] for _ in parts]
        for i in range(len(task.operations)):
            if i in cut_slots:
                branch = branch_ids[cut_slots[i]]
                coefficient *= branches[branch][0]
                gates = replacements[i][branch]
            else:
                gates = ordinary[i]
            for group, g in gates:
                local_ops[group].append(g)
        recipes = []
        for part, ops in zip(parts, local_ops, strict=True):
            key = (len(part), tuple(ops))
            if key not in registry:
                registry[key] = Recipe(*key)
            recipes.append(registry[key])
        if len(registry) > options.max_fragments:
            raise CutLimitError("Unique fragment cap exceeded")
        result.append((coefficient, tuple(recipes)))
    return tuple(result)


def compile_group(
    task, cuts, parts, options, optimization="none", *, recipes=None, formula_cache=None
):
    """Compile cut recipes and observable formulas for one task group."""
    if (10 ** len(cuts)) * len(task.outputs) > options.max_terms:
        raise CutLimitError("Reconstruction term cap exceeded")
    if recipes is None:
        recipes = compile_recipes(task, cuts, parts, options)
    formula_cache = {} if formula_cache is None else formula_cache
    location = {q: (i, j) for i, part in enumerate(parts) for j, q in enumerate(part)}
    formulas = []
    for index, observable in task.outputs:
        if observable not in formula_cache:
            local_obs = tuple(
                Pauli(
                    tuple(
                        (location[q][1], axis)
                        for q, axis in observable.factors
                        if location[q][0] == i
                    )
                )
                for i in range(len(parts))
            )
            terms, refs = {}, {}
            for coefficient, branch_recipes in recipes:
                factors = []
                for recipe, pauli in zip(branch_recipes, local_obs, strict=True):
                    key = (recipe, pauli)
                    if key not in refs:
                        # Measured identity factors retain branch probability.
                        refs[key] = (
                            ValueRef(recipe, pauli)
                            if pauli.factors
                            or any(isinstance(g, Projector) for g in recipe.operations)
                            else None
                        )
                    if refs[key] is not None:
                        factors.append(refs[key])
                key = tuple(
                    sorted(factors, key=lambda r: (r.recipe.sort_key, r.observable.factors))
                )
                terms.setdefault(key, []).append(coefficient)
            formula_cache[observable] = tuple(
                Term(weight, key) for key, values in terms.items() if (weight := fsum(values)) != 0
            )
        formulas.append(Formula(index, formula_cache[observable]))
    return GroupPlan(
        tuple(formulas), tuple(cuts), parts, optimization, 6 ** len(cuts), 10 ** len(cuts)
    )
