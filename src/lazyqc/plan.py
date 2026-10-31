"""Backend-independent physical plans. They contain no quantum state objects."""

from dataclasses import asdict, dataclass
from functools import cached_property

from .cost import CalibratedCostModel, _compute_features
from .instruments import Projector
from .ir import Gate, Pauli
from .tensor_cost import mps_payload_bytes


@dataclass(frozen=True)
class Task:
    wires: tuple[int, ...]
    operations: tuple[Gate | Projector, ...]  # logical indices; compacted during lowering
    outputs: tuple[tuple[int, Pauli], ...]
    # Largest exact-MPS bond for tensor-network tasks; 0 marks a state-vector task.
    bond: int = 0

    @property
    def representation(self):
        """Return ``mps`` for tensor-network tasks and ``statevector`` otherwise."""
        return "mps" if self.bond else "statevector"

    @property
    def width(self):
        """Return the number of local wires in the task."""
        return len(self.wires)

    @property
    def state_bytes(self):
        """Return the raw complex-state payload size in bytes."""
        if self.bond:
            return mps_payload_bytes(self.width, self.bond)
        return 16 * (1 << self.width)

    @property
    def score(self):
        """Return the heuristic work score used for candidate comparison."""
        if self.bond:
            return (1 + len(self.operations)) * self.bond**3 + (
                len(self.outputs) * self.width * self.bond**2
            )
        return (1 << self.width) * (1 + len(self.operations) + len(self.outputs))


@dataclass(frozen=True)
class Candidate:
    strategy: str
    tasks: tuple[Task, ...]
    rejection: str | None = None
    predicted_execution_ns: float | None = None
    optimization: str = "none"

    @property
    def label(self):
        """Return the human-readable strategy and optimization label."""
        return (
            self.strategy if self.optimization == "none" else f"{self.strategy}+{self.optimization}"
        )

    @cached_property
    def score(self):
        """Return the total heuristic work score of all tasks."""
        return sum(t.score for t in self.tasks)

    @cached_property
    def max_width(self):
        """Return the widest state-vector task, or zero when there is none."""
        return max((t.width for t in self.tasks if not t.bond), default=0)

    @property
    def representation(self):
        """Return ``mps`` when any task is a tensor-network task."""
        return "mps" if any(t.bond for t in self.tasks) else "statevector"

    @property
    def max_bond(self):
        """Return the largest exact-MPS bond over tensor-network tasks, or zero."""
        return max((t.bond for t in self.tasks), default=0)

    @cached_property
    def min_width(self):
        """Return the narrowest task, or zero when the candidate is empty."""
        return min((t.width for t in self.tasks), default=0)

    @cached_property
    def cost_features(self):
        """Return the calibrated-model features for this candidate."""
        return _compute_features(self)

    @cached_property
    def state_bytes(self):
        """Return the largest raw state payload required by the candidate."""
        return max((t.state_bytes for t in self.tasks), default=0)


@dataclass(frozen=True)
class Plan:
    selected: Candidate
    alternatives: tuple[Candidate, ...]
    request_count: int
    constant_outputs: tuple[tuple[int, float], ...]
    requested_strategy: str
    max_state_qubits: int
    max_state_bytes: int
    planning_ns: int
    cost_model: CalibratedCostModel | None = None
    cost_model_status: str = "heuristic"
    cache_status: str = "disabled"
    requested_optimization: str = "none"
    optimization_ns: int = 0
    optimization_build_ns: int = 0
    rewrite_records: tuple = ()
    optimizer_id: str | None = None
    zx_options: object | None = None
    analysis_budget: tuple = ()

    def explain(self) -> dict:
        """Return a serializable explanation of selection, limits, and alternatives."""
        return {
            "requested_strategy": self.requested_strategy,
            "selected_strategy": self.selected.strategy,
            "selected_candidate": self.selected.label,
            "optimization": {
                "requested": self.requested_optimization,
                "selected": self.selected.optimization,
                "elapsed_ns": self.optimization_ns,
                "original_build_ns": self.optimization_build_ns,
                "optimizer_id": self.optimizer_id,
                "options": asdict(self.zx_options) if self.zx_options is not None else None,
                "records": [asdict(r) for r in self.rewrite_records],
                "timing_scope": "included in planning; build time is historical on cache hits",
            },
            "request_count": self.request_count,
            "score_units": "heuristic work units; not seconds",
            "cost_model_status": self.cost_model_status,
            "cache_status": self.cache_status,
            "profile_id": self.cost_model.profile_id if self.cost_model else None,
            "prediction_scope": "execution only; excludes construction, planning and calibration",
            "selection_reason": (
                "analysis skipped: dense execution predicted below cone analysis cost"
                if dict(self.analysis_budget).get("skipped")
                else "no state-vector plan is feasible; exact MPS fallback with bond "
                f"{self.selected.max_bond} predicted statically, without contracting"
                if self.selected.representation == "mps" and self.requested_strategy == "auto"
                else "lowest predicted execution time, then fewest tasks"
                if self.cost_model_status == "calibrated"
                else "lowest work score, then fewest tasks"
            )
            if self.requested_strategy == "auto" or self.requested_optimization != "none"
            else "explicitly requested strategy",
            "analysis_budget": dict(self.analysis_budget) or None,
            "memory_scope": "raw state or MPS tensor payload only; not process RSS",
            "representation": self.selected.representation,
            "max_mps_bond": self.selected.max_bond or None,
            "max_state_qubits": self.selected.max_width,
            "estimated_state_bytes": self.selected.state_bytes,
            "limits": {
                "max_state_qubits": self.max_state_qubits,
                "max_state_bytes": self.max_state_bytes,
            },
            "constant_outputs": list(self.constant_outputs),
            "alternatives": [
                {
                    "strategy": c.strategy,
                    "candidate": c.label,
                    "optimization": c.optimization,
                    "score": c.score,
                    "predicted_execution_ns": c.predicted_execution_ns,
                    "evolutions": len(c.tasks),
                    "representation": c.representation,
                    "max_mps_bond": c.max_bond or None,
                    "max_state_qubits": c.max_width,
                    "estimated_state_bytes": c.state_bytes,
                    "rejection": c.rejection,
                }
                for c in self.alternatives
            ],
            "groups": [
                {
                    "request_indices": [i for i, _ in t.outputs],
                    "logical_to_local": {str(q): i for i, q in enumerate(t.wires)},
                    "gate_count": len(t.operations),
                    "state_qubits": t.width,
                    "score": t.score,
                }
                for t in self.selected.tasks
            ],
        }

    def collect(self, *, backend=None, wide_backend=None, wide_from=None, mps_backend=None):
        """Execute the selected plan and return its values and statistics.

        Tasks of at least `wide_from` qubits run on `wide_backend` when given.
        Tensor-network tasks run on `mps_backend` (quimb exact MPS by default).
        """
        from .executor import execute

        return execute(
            self,
            backend=backend,
            wide_backend=wide_backend,
            wide_from=wide_from,
            mps_backend=mps_backend,
        )
