"""A two-level physical DAG: unique local experiments feed weighted result nodes."""

import json
from dataclasses import asdict, dataclass, replace
from functools import cached_property

from .cutting import CutLimitError, GroupPlan
from .instruments import Projector
from .plan import Task


@dataclass(frozen=True)
class Product:
    coefficient: float
    factors: tuple[tuple[int, int], ...]  # (experiment node, output slot); multiplicity matters


@dataclass(frozen=True)
class Reconstruction:
    output: int
    terms: tuple[Product, ...]


@dataclass(frozen=True)
class BatchCandidate:
    strategy: str
    tasks: tuple[Task, ...]
    reconstruction: tuple[Reconstruction, ...]
    groups: tuple[GroupPlan, ...]
    estimated_cost: float = 0.0
    cost_units: str = "heuristic work units"

    @property
    def optimization(self):
        """Return the combined optimization label of the group's fragments."""
        return "+".join(sorted({g.optimization for g in self.groups})) or "none"

    @property
    def label(self):
        """Return the stable strategy, optimization, and cut-count label."""
        return f"{self.strategy}/{self.optimization}/cuts-{self.cut_count}"

    @property
    def score(self):
        """Return the candidate's estimated execution and reconstruction cost."""
        return self.estimated_cost

    @cached_property
    def max_width(self):
        """Return the widest experiment task, or zero when there are no tasks."""
        return max((t.width for t in self.tasks), default=0)

    @property
    def state_bytes(self):
        """Return the largest raw state payload required by any task."""
        return max((t.state_bytes for t in self.tasks), default=0)

    @property
    def cut_count(self):
        """Return the total number of cut gates across the groups."""
        return sum(len(g.cut_indices) for g in self.groups)

    @cached_property
    def term_count(self):
        """Return the total number of reconstruction terms."""
        return sum(len(r.terms) for r in self.reconstruction)

    @cached_property
    def factor_count(self):
        """Return the total number of factor references in reconstruction terms."""
        return sum(len(term.factors) for r in self.reconstruction for term in r.terms)


def assemble(groups, options, name):
    """Assemble group recipes into a deduplicated batch candidate."""
    groups = tuple(groups)
    registry, recipes, observables = {}, [], []
    reconstructions = []
    for group_index, group in enumerate(groups):

        def intern(ref):
            """Intern a recipe and observable reference in the batch DAG."""
            key = ref.recipe if options.reuse else (group_index, ref.recipe)
            if key not in registry:
                registry[key] = len(recipes)
                recipes.append(ref.recipe)
                observables.append({})
            index = registry[key]
            outputs = observables[index]
            if ref.observable not in outputs:
                outputs[ref.observable] = len(outputs)
            return index, outputs[ref.observable]

        for formula in group.formulas:
            terms = tuple(
                Product(term.coefficient, tuple(intern(r) for r in term.factors))
                for term in formula.terms
            )
            reconstructions.append(Reconstruction(formula.output, terms))
    if len(recipes) > options.max_fragments:
        raise CutLimitError("Batch unique-fragment cap exceeded")
    if sum(len(r.terms) for r in reconstructions) > options.max_terms:
        raise CutLimitError("Batch reconstruction-term cap exceeded")
    tasks = tuple(
        Task(
            tuple(range(recipe.width)), recipe.operations, tuple((i, p) for p, i in outputs.items())
        )
        for recipe, outputs in zip(recipes, observables, strict=True)
    )
    return BatchCandidate(name, tasks, tuple(reconstructions), groups)


def task_work(task):
    """Estimate task work from state size, operations, and projections."""
    projections = sum(isinstance(g, Projector) for g in task.operations)
    return 128 + task.score + 16 * projections * (1 << task.width)


def task_durations(tasks, cost_model=None):
    """Return calibrated task durations, or heuristic work without a cost model."""
    return [cost_model.task_ns(t) if cost_model else task_work(t) for t in tasks]


def schedule_estimate(tasks, workers, memory, durations):
    """Deterministic longest-job-first list scheduling with aggregate state-payload limits."""
    pending = sorted(range(len(tasks)), key=lambda i: (-durations[i], i))
    active, now, peak = [], 0.0, 0
    while pending or active:
        live = sum(tasks[i].state_bytes for _, i in active)
        for i in list(pending):
            if len(active) >= workers:
                break
            if live + tasks[i].state_bytes <= memory:
                pending.remove(i)
                active.append((now + durations[i], i))
                live += tasks[i].state_bytes
                peak = max(peak, live)
        if not active:
            raise CutLimitError("A task exceeds aggregate state-payload memory")
        now = min(end for end, _ in active)
        active = [(end, i) for end, i in active if end > now]
    return now, peak


def price(candidate, workers, memory, cost_model=None):
    """Price a candidate with calibrated durations or the heuristic cost model."""
    durations = task_durations(candidate.tasks, cost_model)
    makespan, _ = schedule_estimate(candidate.tasks, workers, memory, durations)
    reconstruction = (
        cost_model.reconstruction_ns(candidate.term_count, candidate.factor_count)
        if cost_model
        else 8 * candidate.term_count + 4 * candidate.factor_count
    )
    scheduling = (
        cost_model.scheduling_ns_per_task * len(candidate.tasks)
        if cost_model and cost_model.scheduling_ns_per_task is not None
        else 0
    )
    return replace(
        candidate,
        estimated_cost=makespan + reconstruction + scheduling,
        cost_units="estimated ns" if cost_model else "heuristic work units",
    )


@dataclass(frozen=True)
class BatchPlan:
    selected: BatchCandidate
    alternatives: tuple[BatchCandidate, ...]
    request_count: int
    constant_outputs: tuple[tuple[int, float], ...]
    max_state_qubits: int
    max_state_bytes: int
    workers: int
    total_state_bytes: int
    cutting: object
    planning_ns: int
    search_states: int
    search_exhausted: bool
    cost_model: object = None
    cache_status: str = "disabled"
    optimization_ns: int = 0
    optimization_build_ns: int = 0
    rewrite_records: tuple = ()
    provenance: tuple[tuple[str, str], ...] = ()
    rejections: tuple[str, ...] = ()
    analysis_stats: tuple[tuple[str, int], ...] = ()
    beam_truncated: bool = False
    structural_template: str = ""

    def explain(self):
        """Return a serializable explanation of batch selection and reconstruction."""
        c = self.selected
        return {
            "selected_candidate": c.label,
            "selected_strategy": c.strategy,
            "request_count": self.request_count,
            "cache_status": self.cache_status,
            "cost_units": c.cost_units,
            "estimated_cost": c.estimated_cost,
            "prediction_scope": (
                "makespan plus serial dispatch plus reconstruction; "
                "excludes search, cache-hit planning and process startup"
            ),
            "limits": {
                "max_state_qubits": self.max_state_qubits,
                "max_state_bytes": self.max_state_bytes,
                "workers": self.workers,
                "total_state_bytes": self.total_state_bytes,
                **asdict(self.cutting),
            },
            "memory_scope": "active raw state payload; not process RSS or plan memory",
            "search": {
                "states": self.search_states,
                "budget_exhausted": self.search_exhausted,
                "beam_truncated": self.beam_truncated,
                "structural_template": (
                    json.loads(self.structural_template) if self.structural_template else {}
                ),
                "optimality": "bounded heuristic; no global optimality claim",
                "analysis": dict(self.analysis_stats),
                "expected_collections": self.cutting.expected_collections,
            },
            "optimization": {
                "elapsed_ns": self.optimization_ns,
                "original_build_ns": self.optimization_build_ns,
                "records": [asdict(r) for r in self.rewrite_records],
            },
            "provenance": dict(self.provenance),
            "rejections": self.rejections,
            "alternatives": [
                {
                    "candidate": a.label,
                    "score": a.estimated_cost,
                    "unique_experiments": len(a.tasks),
                    "terms": a.term_count,
                    "cut_count_across_groups": a.cut_count,
                    "max_state_qubits": a.max_width,
                }
                for a in self.alternatives
            ],
            "dag": {
                "experiment_nodes": [
                    {
                        "id": i,
                        "width": t.width,
                        "gates": len(t.operations),
                        "projectors": sum(isinstance(g, Projector) for g in t.operations),
                        "outputs": len(t.outputs),
                    }
                    for i, t in enumerate(c.tasks)
                ],
                "reconstruction_nodes": [
                    {
                        "request": r.output,
                        "terms": len(r.terms),
                        "dependencies": sorted({i for term in r.terms for i, _ in term.factors}),
                    }
                    for r in c.reconstruction
                ],
            },
            "groups": [
                {
                    "requests": g.outputs,
                    "cut_gate_indices": g.cut_indices,
                    "components": g.components,
                    "optimization": g.optimization,
                    "qpd_terms": g.qpd_terms,
                    "enumerated_branches": g.branches,
                    "gamma": 3 ** len(g.cut_indices),
                    "sampling_overhead_not_exact_work": 9 ** len(g.cut_indices),
                }
                for g in c.groups
            ],
        }

    def collect(self, *, backend=None, session=None):
        """Execute the selected batch plan and return reconstructed results."""
        from .batch_executor import execute_batch

        return execute_batch(self, backend=backend, session=session)
