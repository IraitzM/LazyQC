"""Qiskit ``BaseEstimatorV2`` primitive backed by LazyQC.

``Estimator.run`` turns every PUB of a job into requests over one translated
circuit, plans the whole job with a single ``plan_all`` call, collects, and
reshapes the values to each PUB's broadcast shape. Qiskit code that already
goes through the Estimator primitive, including gradient classes and
``EstimatorQNN``, then reaches the planner without being rewritten.

Values are exact. A ``precision`` above zero adds Gaussian noise to the exact
values, as ``StatevectorEstimator`` does; the plan is still chosen for exact
simulation and nothing is sampled from a distribution.

Repeated jobs: training loops send the same circuit and observables with new parameter
values. The Estimator keeps a symbolic plan per job structure (`plan_all(symbolic=True)`) and
passes the values as ``bindings``, so later jobs skip circuit construction, planning and
lowering. Values equal the fresh-plan path exactly. Options that need numeric angles
(``optimization``, ``cutting``, ``workers``, ``total_state_bytes``, ``cost_model``) keep the
fresh-plan path.

Performance note: ``collect`` runs Qulacs on one thread when every task is narrower than
``lazyqc.threads.SMALL_TASK_QUBITS`` and caps the team for wider ones (``native_threads="auto"``).
Pass ``collect_options={"native_threads": "default"}`` to leave Qulacs' team alone.
"""

import numpy as np
from qiskit.primitives import BaseEstimatorV2, PrimitiveJob, PrimitiveResult, PubResult
from qiskit.primitives.containers import DataBin
from qiskit.primitives.containers.estimator_pub import EstimatorPub

from ..ir import Pauli, PauliSum, SumRequest
from ..planner import plan_all, symbolic_compatible
from .qiskit import ParametricTemplate

_IMAGINARY_TOLERANCE = 1e-12
_TEMPLATE_CACHE = 8
_PLAN_CACHE = 4


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
        self._plans = {}  # job structure -> (templates, symbolic plan), oldest first
        self._once = {}  # job structures seen once, oldest first; a repeat earns a symbolic plan
        self.last_job = {}

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
        seen = {}  # circuits cannot change during a run, so each is validated once per job
        prepared = [self._prepare(pub, seen) for pub in pubs]
        counts = [len(item["sums_at"]) for item in prepared]
        values, plan_info = (), {}
        if sum(counts):
            if symbolic_compatible(self._plan_options):
                plan, values, reused = self._collect_symbolic(prepared)
            else:
                requests = [r for item in prepared for r in self._requests(item)]
                plan = plan_all(requests, **self._plan_options)
                values, reused = plan.collect(**self._collect_options).values, False
            plan_info = {
                "selected_strategy": plan.selected.strategy,
                "tasks": len(plan.selected.tasks),
                "requests": sum(counts),
                "plan_reused": reused,
            }
        self.last_job = plan_info
        results, start = [], 0
        for pub, item, count in zip(pubs, prepared, counts, strict=True):
            evs = np.asarray(values[start : start + count], dtype=np.float64).reshape(item["shape"])
            start += count
            results.append(self._pub_result(pub, evs))
        return PrimitiveResult(results, metadata={"version": 2, "lazyqc": plan_info})

    def _collect_symbolic(self, prepared):
        """Collect a job through a cached symbolic plan, building it on first sight."""
        key = tuple(
            (id(item["template"]), item["sums"], item["shape"], item["row_ids"].tobytes())
            for item in prepared
        )
        templates = tuple(item["template"] for item in prepared)
        entry = self._plans.pop(key, None)
        reused = entry is not None and all(a is b for a, b in zip(entry[0], templates, strict=True))
        if not reused:
            if key not in self._once:
                # First sight of this structure: a one-off job should not pay for a plan that
                # is never reused, so it takes the fresh-plan path and only remembers the key.
                self._once[key] = templates
                while len(self._once) > 4 * _PLAN_CACHE:
                    self._once.pop(next(iter(self._once)))
                requests = [r for item in prepared for r in self._requests(item)]
                plan = plan_all(requests, **self._plan_options)
                return plan, plan.collect(**self._collect_options).values, False
            self._once.pop(key)
            requests = []
            for number, item in enumerate(prepared):
                programs = {
                    row: item["template"].symbolic_program(f"{number}.{row}:")
                    for row in item["used"]
                }
                requests.extend(self._requests(item, programs))
            entry = (templates, plan_all(requests, symbolic=True, **self._plan_options))
        self._plans[key] = entry  # most recently used goes last
        while len(self._plans) > _PLAN_CACHE:
            self._plans.pop(next(iter(self._plans)))
        bindings = {}
        for number, item in enumerate(prepared):
            for row in item["used"]:
                bindings.update(item["template"].slot_values(item["flat"][row], f"{number}.{row}:"))
        plan = entry[1]
        return plan, plan.collect(bindings=bindings, **self._collect_options).values, reused

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

    def _prepare(self, pub, seen):
        """Validate one PUB and return what both planning paths need, in C order."""
        circuit, shape = pub.circuit, pub.shape
        template = self._template_for(circuit, seen)
        observables = pub.observables
        if observables.num_qubits != circuit.num_qubits:
            raise ValueError("Observables and circuit act on different numbers of qubits")
        sums = tuple(
            _pauli_sum(observables[index], circuit.num_qubits)
            for index in np.ndindex(*observables.shape)
        )
        sum_ids = np.broadcast_to(np.arange(len(sums)).reshape(observables.shape), shape)
        if template.parameters:
            rows = pub.parameter_values.as_array(template.parameters)
            row_shape = rows.shape[:-1]
            row_ids = np.broadcast_to(np.arange(int(np.prod(row_shape))).reshape(row_shape), shape)
            flat = rows.reshape(-1, rows.shape[-1])
        else:
            row_ids, flat = np.zeros(shape, dtype=int), [()]
        return {
            "shape": shape,
            "template": template,
            "sums": sums,
            "flat": flat,
            "row_ids": np.ascontiguousarray(row_ids),
            "sums_at": [int(sum_ids[index]) for index in np.ndindex(*shape)],
            "rows_at": [int(row_ids[index]) for index in np.ndindex(*shape)],
            "used": sorted(set(int(r) for r in row_ids.flat)),
        }

    @staticmethod
    def _requests(item, programs=None):
        """Return the requests of one PUB; numeric programs are built per row when not given."""
        if programs is None:
            programs = {}
            for row in item["used"]:
                programs[row] = item["template"].program(item["flat"][row])
        return [
            SumRequest(programs[row], item["sums"][s])
            for row, s in zip(item["rows_at"], item["sums_at"], strict=True)
        ]

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
