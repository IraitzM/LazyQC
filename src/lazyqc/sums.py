"""Weighted Pauli sums: expand into term requests, plan them together, recombine.

A `SumRequest` has no execution path of its own. `plan_all` expands it into one
`Request` per term, so the planner shares tasks between terms whose cones coincide,
and wraps the resulting plan in a `SumPlan` that adds the weighted term values.
"""

from dataclasses import dataclass, replace
from math import fsum

from .ir import Request, SumRequest


def expand_requests(requests):
    """Flatten sum requests into term requests.

    Returns ``(expanded, layout)``. ``layout`` is ``None`` when no request is a sum.
    Otherwise it holds, per original request, either ``("plain", position)`` or
    ``("sum", ((position, coefficient), ...))`` with positions into ``expanded``.
    """
    if not any(isinstance(r, SumRequest) for r in requests):
        return requests, None
    expanded, layout = [], []
    for request in requests:
        if isinstance(request, SumRequest):
            weights = []
            for (coefficient, _), term in zip(
                request.observable.terms, request.expand(), strict=True
            ):
                weights.append((len(expanded), coefficient))
                expanded.append(term)
            layout.append(("sum", tuple(weights)))
        elif isinstance(request, Request):
            layout.append(("plain", len(expanded)))
            expanded.append(request)
        else:
            raise ValueError("plan_all accepts expectation Requests")
    return tuple(expanded), tuple(layout)


def recombine(values, layout):
    """Return one value per original request, summing each sum's weighted terms."""
    combined = []
    for kind, payload in layout:
        if kind == "plain":
            combined.append(values[payload])
        else:
            combined.append(
                fsum(coefficient * values[position] for position, coefficient in payload)
            )
    return tuple(combined)


@dataclass(frozen=True)
class SumPlan:
    """A plan over expanded term requests that returns one value per original request.

    Everything not defined here (``selected``, ``alternatives``, ``request_count`` and
    so on) is read from the underlying plan, which counts term requests.
    """

    plan: object
    layout: tuple

    def __getattr__(self, name):
        """Forward attribute reads to the underlying plan."""
        return getattr(self.plan, name)

    def collect(self, **options):
        """Execute the underlying plan and recombine term values per sum."""
        result = self.plan.collect(**options)
        return replace(result, values=recombine(result.values, self.layout))

    def explain(self) -> dict:
        """Return the underlying explanation plus the sum-to-term mapping."""
        explanation = self.plan.explain()
        explanation["sums"] = [
            {
                "request_index": index,
                "term_request_indices": [position for position, _ in payload],
                "coefficients": [coefficient for _, coefficient in payload],
            }
            for index, (kind, payload) in enumerate(self.layout)
            if kind == "sum"
        ]
        explanation["original_request_count"] = len(self.layout)
        return explanation
