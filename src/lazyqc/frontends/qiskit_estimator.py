"""Qiskit ``BaseEstimatorV2`` primitive backed by LazyQC.

``Estimator.run`` turns every PUB of a job into requests over one translated
circuit, plans the whole job with a single ``plan_all`` call, collects, and
reshapes the values to each PUB's broadcast shape. Qiskit code that already
goes through the Estimator primitive, including gradient classes and
``EstimatorQNN``, then reaches the planner without being rewritten.

Values are exact. A ``precision`` above zero adds Gaussian noise to the exact
values, as ``StatevectorEstimator`` does; the plan is still chosen for exact
simulation and nothing is sampled from a distribution.

Performance note: Qulacs defaults to an OpenMP thread team, which is far slower than one
thread on the small states cone reduction produces. Set ``OMP_NUM_THREADS=1`` and
``QULACS_NUM_THREADS=1`` before importing Qulacs, as the benchmarks do.
"""

import numpy as np
from qiskit.primitives import BaseEstimatorV2, PrimitiveJob, PrimitiveResult, PubResult
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from ..ir import Pauli, PauliSum, SumRequest
from ..planner import plan_all
from .qiskit import ParametricTemplate

_IMAGINARY_TOLERANCE = 1e-12
_TEMPLATE_CACHE = 8


def _pauli_sum(observable, n_qubits):
    """Convert one ``{label: coefficient}`` observable into a `PauliSum`.

    Qiskit labels put qubit 0 rightmost, so character ``i`` from the right is qubit ``i``.
    """
    terms = []
    for label, coefficient in observable.items():
        if len(label) != n_qubits:
            raise ValueError(f"Observable acts on {len(label)} qubits; the circuit has {n_qubits}")
        coefficient = complex(coefficient)
        if abs(coefficient.imag) > _IMAGINARY_TOLERANCE:
            raise ValueError("Observable has a complex coefficient; it must be Hermitian")
        factors = tuple((q, p) for q, p in enumerate(reversed(label)) if p != "I")
        terms.append((coefficient.real, Pauli(factors)))
    return PauliSum(tuple(terms))


class Estimator(BaseEstimatorV2):
    """Exact Pauli-expectation Estimator that plans each job as one batch.

    Args:
        default_precision: Precision used when neither the PUB nor ``run`` gives one.
            Zero returns exact values.
        seed: Seed or ``numpy.random.Generator`` for the emulated noise. Each PUB draws
            from ``numpy.random.default_rng(seed)`` in C order over its broadcast shape,
            the convention of ``StatevectorEstimator``, so equal seeds give equal values.
        plan_options: Keyword arguments for `plan_all`, such as ``strategy`` or
            ``max_state_qubits``.
        collect_options: Keyword arguments for ``collect``, such as ``wide_backend``.
    """

    def __init__(
        self,
        *,
        default_precision: float = 0.0,
        seed=None,
        plan_options: dict | None = None,
        collect_options: dict | None = None,
    ):
        """Create an Estimator; options are validated when a job runs."""
        self._default_precision = default_precision
        self._seed = seed
        self._plan_options = dict(plan_options or {})
        self._collect_options = dict(collect_options or {})
        self._templates = {}  # id(circuit) -> (private copy, ParametricTemplate), oldest first

    @property
    def default_precision(self) -> float:
        """Return the default precision."""
        return self._default_precision

    @property
    def seed(self):
        """Return the seed or generator used for emulated noise."""
        return self._seed

    def run(self, pubs, *, precision: float | None = None):
        """Submit PUBs and return a job whose result holds one `PubResult` per PUB."""
        if precision is None:
            precision = self._default_precision
        coerced = [EstimatorPub.coerce(pub, precision) for pub in pubs]
        job = PrimitiveJob(self._run, coerced)
        job._submit()
        return job

    def _run(self, pubs):
        """Plan every PUB together, collect once, and split the values by PUB."""
        requests, layouts = [], []
        seen = {}  # circuits cannot change during a run, so each is validated once per job
        for pub in pubs:
            layout = self._requests_for(pub, seen)
            layouts.append((layout[0], len(layout[1])))
            requests.extend(layout[1])
        values, plan_info = (), {}
        if requests:
            plan = plan_all(requests, **self._plan_options)
            values = plan.collect(**self._collect_options).values
            plan_info = {
                "selected_strategy": plan.selected.strategy,
                "tasks": len(plan.selected.tasks),
                "requests": len(requests),
            }
        results, start = [], 0
        for pub, (shape, count) in zip(pubs, layouts, strict=True):
            evs = np.asarray(values[start : start + count], dtype=np.float64).reshape(shape)
            start += count
            results.append(self._pub_result(pub, evs))
        return PrimitiveResult(results, metadata={"version": 2, "lazyqc": plan_info})

    def _template_for(self, circuit, seen):
        """Translate a circuit once and reuse it while it stays unchanged.

        ``EstimatorQNN`` and Qiskit's gradient classes send the same circuit object on every
        call. Circuits are mutable, so the cache keeps a private copy and compares against it.
        """
        if id(circuit) in seen:
            return seen[id(circuit)]
        entry = self._templates.pop(id(circuit), None)
        if entry is None or entry[0] != circuit:
            entry = (circuit.copy(), ParametricTemplate(circuit))
        self._templates[id(circuit)] = entry  # most recently used goes last
        while len(self._templates) > _TEMPLATE_CACHE:
            self._templates.pop(next(iter(self._templates)))
        seen[id(circuit)] = entry[1]
        return entry[1]

    def _requests_for(self, pub, seen):
        """Return ``(shape, requests)`` for one PUB, in C order over its broadcast shape."""
        circuit, shape = pub.circuit, pub.shape
        template = self._template_for(circuit, seen)
        observables = pub.observables
        if observables.num_qubits != circuit.num_qubits:
            raise ValueError("Observables and circuit act on different numbers of qubits")
        sums = [
            _pauli_sum(observables[index], circuit.num_qubits)
            for index in np.ndindex(*observables.shape)
        ]
        sum_ids = np.broadcast_to(np.arange(len(sums)).reshape(observables.shape), shape)
        if template.parameters:
            rows = pub.parameter_values.as_array(template.parameters)
            row_shape = rows.shape[:-1]
            row_ids = np.broadcast_to(np.arange(int(np.prod(row_shape))).reshape(row_shape), shape)
            flat = rows.reshape(-1, rows.shape[-1])
        else:
            row_ids, flat = np.zeros(shape, dtype=int), [()]
        programs = {}
        requests = []
        for index in np.ndindex(*shape):
            row = int(row_ids[index])
            if row not in programs:
                programs[row] = template.program(flat[row])
            requests.append(SumRequest(programs[row], sums[int(sum_ids[index])]))
        return shape, requests

    def _pub_result(self, pub, evs):
        """Attach emulated noise when the PUB asks for a precision above zero."""
        precision = pub.precision
        stds = np.zeros_like(evs)
        if precision != 0:
            rng = np.random.default_rng(self._seed)
            evs = evs.copy()
            for index in np.ndindex(*evs.shape):
                evs[index] = rng.normal(evs[index], precision)
            stds = np.full_like(evs, float(precision))
        data = DataBin(evs=evs, stds=stds, shape=evs.shape)
        return PubResult(
            data,
            metadata={"target_precision": precision, "circuit_metadata": pub.circuit.metadata},
        )
