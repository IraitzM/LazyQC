"""quimb lightcones with stacked observables and shared lazy contraction intermediates."""

from time import perf_counter_ns


def prepare(requests):
    import autoray as ar
    import cotengra as ctg
    import numpy as np
    import quimb as qu
    import quimb.tensor as qtn

    started = perf_counter_ns()
    circuits, groups, constants = {}, {}, []
    for i, request in enumerate(requests):
        p = request.program
        key = (p.n_qubits, p.resolved_operations())
        if not request.observable.factors:
            constants.append(i)
            continue
        if key not in circuits:
            circuit = qtn.Circuit(p.n_qubits)
            for g in key[1]:
                args = (*g.qubits,) if g.angle is None else (g.angle, *g.qubits)
                circuit.apply_gate(g.name, *args)
            circuits[key] = circuit
        where = tuple(q for q, _ in request.observable.factors)
        matrix = np.ones((1, 1), dtype=complex)
        for _, axis in request.observable.factors:
            matrix = np.kron(matrix, qu.pauli(axis))
        groups.setdefault((key, where), []).append((i, matrix))
    optimizer = ctg.ReusableRandomGreedyOptimizer(max_repeats=32, seed=5201, parallel=False)
    expressions, output_groups, widths = [], [], []
    arrays = {}  # Preserve array identity across rehearsed networks when available.
    with ar.lazy.shared_intermediates():
        for (key, where), outputs in groups.items():
            rehearsed = circuits[key].local_expectation(
                [matrix for _, matrix in outputs], where, optimize=optimizer, rehearse=True
            )
            tn, tree = rehearsed["tn"], rehearsed["tree"]
            widths.append(float(tree.contraction_width()))

            def wrap(array):
                if id(array) not in arrays:
                    arrays[id(array)] = ar.lazy.array(array)
                return arrays[id(array)]

            tn.apply_to_arrays(wrap)
            value = tn.contract(all, output_inds=tn.outer_inds(), optimize=tree)
            expressions.append(value.data if isinstance(value, qtn.Tensor) else value)
            output_groups.append(tuple(i for i, _ in outputs))
    # Do not constant-fold: every call performs the scalar/tensor contraction again.
    function = ar.lazy.Function((), tuple(expressions), fold_constants=False)
    return TensorPlan(
        function,
        tuple(output_groups),
        tuple(constants),
        len(requests),
        perf_counter_ns() - started,
        max(widths, default=0),
    )


class TensorPlan:
    def __init__(self, function, groups, constants, count, planning_ns, contraction_width):
        self.function = function
        self.groups = groups
        self.constants = constants
        self.count = count
        self.planning_ns = planning_ns
        self.contraction_width = contraction_width

    def collect(self):
        import numpy as np

        values = [None] * self.count
        for i in self.constants:
            values[i] = 1.0
        for group, result in zip(self.groups, self.function(), strict=True):
            for i, value in zip(group, np.asarray(result).reshape(-1), strict=True):
                if abs(complex(value).imag) > 1e-10:
                    raise ArithmeticError("Complex tensor expectation")
                values[i] = float(complex(value).real)
        return tuple(values)
