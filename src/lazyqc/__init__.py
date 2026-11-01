"""LazyQC: Lazy, Observable-Centric Quantum Simulation & Batch Planning.

LazyQC is an experimental quantum simulation framework based on lazy evaluation,
causal-cone reduction, exact quasi-probability decomposition (QPD) circuit cutting,
ZX-calculus graph simplification, and bounded batch optimization.

Users describe quantum programs and construct deferred expectation value requests.
Classical planning (subcircuit identification, grouping, cut decomposition, and
resource allocation) occurs upfront; native simulation via high-performance backends
(e.g., Qulacs) executes only upon calling `collect()`.

References:
    - Munich Quantum Toolkit (MQT). Toolstack reference for compiler,
      simulation, mapping, verification, and benchmarking layers; see
      docs/MQT_COMPARISON.md. https://mqt.readthedocs.io/en/stable/
    - Armbrust, M., et al. (2015). "Spark SQL: Relational data processing in Spark".
      In Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data,
      pp. 1383-1394.
    - Peng, T., Harrow, A. W., Ozols, M., & Wu, X. (2020). "Simulating Large Quantum Circuits
      on a Small Quantum Computer". Physical Review Letters, 125(15), 150504.
      arXiv:1904.00102.
    - Suzuki, Y., et al. (2021). "Qulacs: a fast versatile quantum circuit simulator for
      Python/C++". Quantum, 5, 559.
    - Duncan, R., Kissinger, A., Perdrix, S., & van de Wetering, J. (2020). "Graph-theoretic
      Simplification of Quantum Circuits with the ZX-calculus". Quantum, 4, 279.
      arXiv:1902.03178.
    - Tang, W., et al. (2021). "CutQC: using small Quantum computers for large Quantum circuits".
      In Proceedings of the 26th ACM International Conference on Architectural Support for
      Programming Languages and Operating Systems (ASPLOS '21), pp. 473-486.
"""

from .batch_cost import BatchCostModel, calibrate_batch
from .cache import PlanningCache
from .cost import CalibratedCostModel
from .cutting import CutOptions
from .ir import Gate, Parameter, Pauli, PauliSum, Program, Request, SumRequest
from .isolation import PlanningResourceError, plan_isolated
from .planner import NoFeasiblePlan, plan_all
from .rewrites import ZXOptions
from .sensitivity import adjoint_gradient, parameter_shift_requests, profile_sensitivity
from .session import ExecutionSession

__all__ = [
    "ExecutionSession",
    "PlanningResourceError",
    "plan_isolated",
    "CutOptions",
    "BatchCostModel",
    "calibrate_batch",
    "ZXOptions",
    "PlanningCache",
    "CalibratedCostModel",
    "Gate",
    "Parameter",
    "Pauli",
    "PauliSum",
    "Program",
    "Request",
    "SumRequest",
    "NoFeasiblePlan",
    "plan_all",
    "adjoint_gradient",
    "parameter_shift_requests",
    "profile_sensitivity",
    "collect_all",
]


def collect_all(requests, **planning_options):
    """Convenience helper to plan a sequence of requests and immediately collect results.

    Shorthand for `plan_all(requests, **planning_options).collect()`.

    Args:
        requests: An iterable of Request objects representing expectation values.
        **planning_options: Arbitrary planning parameters passed directly to `plan_all`,
            such as `strategy`, `cost_model`, `cache`, `max_state_qubits`, etc.

    Returns:
        ExecutionResult: Evaluated expectation values and execution statistics.

    Examples:
        >>> from lazyqc import Program, collect_all
        >>> p = Program(2).h(0).cx(0, 1)
        >>> req = p.expectation({0: "Z", 1: "Z"})
        >>> res = collect_all([req])
        >>> round(res.values[0], 6)
        1.0
    """
    return plan_all(requests, **planning_options).collect()
