"""Bounded candidate selection; grouping shares identical cones, not quantum clones."""

from dataclasses import dataclass, replace
from time import perf_counter_ns

from .cache import PlanningCache
from .cost import CalibratedCostModel
from .ir import Request
from .passes import cone_indices
from .plan import Candidate, Plan, Task
from .rewrites import ZXOptions, optimizer_identity, rewrite_programs
from .tensor_cost import mps_max_bond

STRATEGIES = ("full", "separate", "grouped", "mps", "auto")
ANALYSES = ("budgeted", "always")
# Host defaults fitted by benchmarks/calibrate_analysis_budget.py: dense serial
# execution (per program, per gate, per amplitude-gate, per amplitude-output)
# and cone analysis (per call, per request gate), in nanoseconds. They only
# decide when dense execution is too cheap for analysis to matter.
DENSE_NS = (40_700.0, 1_200.0, 1.0, 4.9)
ANALYSIS_NS = (5_200.0, 220.0)


class NoFeasiblePlan(ValueError):
    pass


def validate_limits(workers, max_state_qubits, max_state_bytes):
    """Reject worker counts and per-state limits outside the supported range."""
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if type(max_state_qubits) is not int or not 1 <= max_state_qubits <= 30:
        raise ValueError("max_state_qubits must be between 1 and 30")
    if type(max_state_bytes) is not int or max_state_bytes < 1:
        raise ValueError("max_state_bytes must be a positive integer")


@dataclass(frozen=True)
class _Analysis:
    candidates: tuple[Candidate, ...]
    constants: tuple[tuple[int, float], ...]
    rewrite_records: tuple = ()
    optimization_ns: int = 0


def _analyze(requests, strategy):
    """Build unoptimized full, separate, or causal-cone candidates."""
    # Cone membership depends only on gate wires, so it is computed once per
    # topology and observable support. Grouping keys pair an interned circuit
    # shape with concrete angles: equal keys mean equal retained circuits.
    full_slots, full_groups, cone_groups = {}, [], {}
    separate, constants = [], []
    programs, layouts, cones, shapes = {}, {}, {}, {}
    for index, request in enumerate(requests):
        program = request.program
        known = programs.get(id(program))
        if known is None:
            operations = program.resolved_operations()
            shape_ops = tuple((g.name, g.qubits) for g in operations)
            layout = layouts.setdefault((program.n_qubits, shape_ops), len(layouts))
            angles = tuple(g.angle for g in operations)
            known = programs[id(program)] = [operations, shape_ops, layout, angles, None]
        operations, shape_ops, layout, angles, slot = known
        if not request.observable.factors:
            constants.append((index, 1.0))
            continue
        output = (index, request.observable)
        if strategy in ("full", "auto"):
            if slot is None:
                # Full groups open at a program's first nonconstant output.
                slot = known[4] = full_slots.setdefault((layout, angles), len(full_slots))
                if slot == len(full_groups):
                    full_groups.append((tuple(range(program.n_qubits)), operations, []))
            full_groups[slot][2].append(output)
        if strategy != "full":
            support = tuple(q for q, _ in request.observable.factors)
            cone = cones.get((layout, support))
            if cone is None:
                wires, retained = cone_indices(tuple(q for _, q in shape_ops), support)
                shape = (wires, tuple(shape_ops[i] for i in retained))
                cone = cones[(layout, support)] = (
                    wires,
                    retained,
                    shapes.setdefault(shape, len(shapes)),
                )
            wires, retained, shape = cone
            key = (shape, tuple(angles[i] for i in retained))
            group = cone_groups.get(key)
            if group is None:
                group = cone_groups[key] = (wires, tuple(operations[i] for i in retained), [])
            if strategy in ("grouped", "auto"):
                group[2].append(output)
            if strategy in ("separate", "auto"):
                separate.append(Task(group[0], group[1], (output,)))

    def grouped_tasks(groups):
        """Convert grouped wire and gate descriptions into immutable tasks."""
        return tuple(Task(wires, gates, tuple(outputs)) for wires, gates, outputs in groups)

    candidates = []
    if strategy in ("full", "auto"):
        candidates.append(Candidate("full", grouped_tasks(full_groups)))
    if strategy in ("separate", "auto"):
        candidates.append(Candidate("separate", tuple(separate)))
    if strategy in ("grouped", "auto"):
        candidates.append(Candidate("grouped", grouped_tasks(cone_groups.values())))
    return _Analysis(tuple(candidates), tuple(constants))


def _mps_candidate(candidates, max_state_bytes):
    """Build the exact-MPS candidate from the whole-program (full) tasks.

    Each task keeps all wires and gates and is tagged with the bond that exact
    MPS execution reaches, predicted without contracting. The candidate is
    rejected only when its tensor payload would exceed the byte limit.
    """
    full = next((c for c in candidates if c.strategy == "full" and c.optimization == "none"), None)
    if full is None or not full.tasks:
        return None
    tasks = tuple(replace(t, bond=mps_max_bond(t.width, t.operations)) for t in full.tasks)
    candidate = Candidate("mps", tasks)
    if candidate.state_bytes > max_state_bytes:
        candidate = replace(
            candidate,
            rejection=(
                f"MPS payload {candidate.state_bytes} exceeds {max_state_bytes} bytes "
                f"(bond bound {candidate.max_bond})"
            ),
        )
    return candidate


def _analysis_budget(requests, max_state_qubits, max_state_bytes):
    """Predict whether cone analysis can repay itself against dense execution.

    Analysis can save at most the dense execution it replaces. When a feasible
    dense plan is predicted to cost less than analysing the requests, no cone
    plan can repay the analysis, so auto keeps the dense plan.
    """
    programs, request_gates = {}, 0
    for request in requests:
        program = request.program
        entry = programs.get(id(program))
        if entry is None:
            entry = programs[id(program)] = [program.n_qubits, len(program.operations), 0]
        if request.observable.factors:
            entry[2] += 1
            request_gates += entry[1]
    widest = max((n for n, _, _ in programs.values()), default=0)
    if widest > max_state_qubits or 16 * (1 << widest) > max_state_bytes:
        return None
    base, per_gate, per_amplitude_gate, per_amplitude_output = DENSE_NS
    dense = sum(
        base + per_gate * gates + (per_amplitude_gate * gates + per_amplitude_output * k) * (1 << n)
        for n, gates, k in programs.values()
    )
    analysis = ANALYSIS_NS[0] + ANALYSIS_NS[1] * request_gates
    return {
        "predicted_dense_ns": round(dense),
        "predicted_analysis_ns": round(analysis),
        "skipped": analysis >= dense,
    }


def _analyze_optimized(requests, strategy, optimization, options):
    """Add distinct locally or ZX-rewritten candidates to the baseline analysis."""
    baseline = _analyze(requests, strategy)
    if optimization == "none":
        return baseline
    local, zx, records, elapsed = rewrite_programs(requests, optimization, options)
    candidates = list(baseline.candidates)
    for name, batch in (("local", local), ("zx", zx)):
        if name == "local" and all(r.local_gates == r.input_gates for r in records):
            continue
        if name == "zx" and (optimization != "zx" or zx == local):
            continue
        additional = _analyze(batch, strategy)
        for c in additional.candidates:
            # Keep cheaper unoptimized/local representatives when structures coincide.
            if not any(
                c.strategy == prior.strategy and c.tasks == prior.tasks for prior in candidates
            ):
                candidates.append(replace(c, optimization=name))
    return _Analysis(tuple(candidates), baseline.constants, records, elapsed)


def _cache_key(requests, strategy):
    """Build a value-based cache key that preserves request order and duplicates."""
    # Canonicalize programs by value, resolving each object once per call. Full
    # equality resolves hash collisions; request order and duplicates remain in key.
    programs, outputs, indices, program_indices = [], [], {}, {}
    for request in requests:
        identity = id(request.program)
        if identity not in indices:
            index = program_indices.get(request.program)
            if index is None:
                index = len(programs)
                programs.append(request.program)
                program_indices[request.program] = index
            indices[identity] = index
        outputs.append((indices[identity], request.observable))
    return strategy, tuple(programs), tuple(outputs)


def _plan_lifetime(requests, cutting, **options):
    """Select the lifetime-optimal plan between independent and joint search.

    Runs both searches deterministically, then picks the least predicted
    execution cost. Both searches are charged: the reported lifetime is
    H_inc + H_joint + R * S_pick, so invoking this selector only pays off at
    high reuse (measured: R > ~21 on QAOA path, R > ~540 on QAOA star).
    Callers should gate invocation on a reuse threshold from history and use
    plain independent search below it; pre-search skipping needs a savings
    model and stays out of scope.
    """
    from .batch_planner import plan_batch

    expected = cutting.expected_collections
    if expected is None:
        raise ValueError("select='lifetime' requires cutting.expected_collections")
    if options["cost_model"] is None:
        raise ValueError("select='lifetime' requires a calibrated cost model")
    results, failures = {}, {}
    for search in ("independent", "joint"):
        try:
            results[search] = plan_batch(
                requests, cutting=replace(cutting, search=search), **options
            )
        except NoFeasiblePlan as exc:
            failures[search] = str(exc)
    if not results:
        raise NoFeasiblePlan("; ".join(f"{s}: {failures[s]}" for s in ("independent", "joint")))
    if len(results) == 1:
        only = next(iter(results))
        note = {"selection": "lifetime", "pick": only, "other_search": "infeasible"}
        winner = results[only]
        return replace(winner, provenance=winner.provenance + tuple(note.items()))
    inc, joint = results["independent"], results["joint"]
    h_total = inc.planning_ns + joint.planning_ns
    s_inc, s_joint = inc.selected.score, joint.selected.score
    pick = "independent" if s_inc <= s_joint else "joint"
    winner, loser = (inc, joint) if pick == "independent" else (joint, inc)
    alternatives = winner.alternatives
    if loser.selected.tasks != winner.selected.tasks:
        alternatives = alternatives + (loser.selected,)
    decision = {
        "selection": "lifetime",
        "expected_collections": str(expected),
        "pick": pick,
        "planning_inc_ns": str(inc.planning_ns),
        "planning_joint_ns": str(joint.planning_ns),
        "predicted_inc_ns": str(s_inc),
        "predicted_joint_ns": str(s_joint),
        "lifetime_pick_ns": str(h_total + expected * winner.selected.score),
        "lifetime_other_ns": str(h_total + expected * loser.selected.score),
        "lifetime_independent_alone_ns": str(inc.planning_ns + expected * s_inc),
    }
    if failures:
        decision["other_search_note"] = "single-sided; see pick"
    return replace(
        winner, provenance=winner.provenance + tuple(decision.items()), alternatives=alternatives
    )


def plan_all(
    requests,
    *,
    strategy="auto",
    max_state_qubits=24,
    max_state_bytes=256 * 2**20,
    cost_model=None,
    cache=None,
    optimization="none",
    zx_options=None,
    cutting=None,
    workers=1,
    total_state_bytes=None,
    select=None,
    analysis="budgeted",
) -> Plan:
    """Plan requests under strategy, resource, cache, and optimization constraints.

    Args:
        requests: Iterable of expectation `Request` objects.
        strategy: One of ``full``, ``separate``, ``grouped``, or ``auto``.
        analysis: ``budgeted`` lets uncached heuristic ``auto`` planning keep a
            feasible dense plan without cone analysis when dense execution is
            predicted to cost less than the analysis; ``always`` analyses.
        max_state_qubits: Maximum width of an executable state vector.
        max_state_bytes: Maximum raw state payload for one experiment.
        cost_model: Optional calibrated execution cost model.
        cache: Optional planning cache.
        optimization: One of ``none``, ``local``, or ``zx``.
        zx_options: Options for ZX optimization.
        cutting: Optional batch-cutting limits.
        workers: Number of batch workers.
        total_state_bytes: Aggregate active state-payload limit for batch execution.
        select: Optional lifetime selection. ``"lifetime"`` runs independent and
            joint search and keeps the least predicted execution cost, charging
            both searches; it requires ``cutting.expected_collections`` and a
            calibrated cost model.

    Returns:
        Plan: An immutable plan containing the selected candidate and alternatives.

    Raises:
        ValueError: If requests or planning options are invalid.
        NoFeasiblePlan: If no candidate satisfies the requested limits.
    """
    if select is not None and select != "lifetime":
        raise ValueError("select must be None or 'lifetime'")
    if select == "lifetime" and cutting is None and workers == 1 and total_state_bytes is None:
        raise ValueError("select='lifetime' requires batch planning options")
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    if cutting is not None or workers != 1 or total_state_bytes is not None:
        from .batch_planner import plan_batch
        from .cutting import CutOptions

        if cutting is None:
            cutting = CutOptions(max_fragment_qubits=max_state_qubits, max_cuts=0)
        options = dict(
            strategy=strategy,
            optimization=optimization,
            zx_options=zx_options,
            workers=workers,
            max_state_qubits=max_state_qubits,
            max_state_bytes=max_state_bytes,
            total_state_bytes=total_state_bytes,
            cache=cache,
            cost_model=cost_model,
        )
        if select == "lifetime":
            return _plan_lifetime(requests, cutting, **options)
        return plan_batch(requests, cutting=cutting, **options)
    start = perf_counter_ns()
    if optimization not in ("none", "local", "zx"):
        raise ValueError("optimization must be none, local or zx")
    if zx_options is not None and not isinstance(zx_options, ZXOptions):
        raise ValueError("zx_options must be ZXOptions")
    if zx_options is not None and optimization != "zx":
        raise ValueError("zx_options requires optimization='zx'")
    options = zx_options or ZXOptions()
    identity = optimizer_identity(optimization) if optimization != "none" else None
    if cache is not None and not isinstance(cache, PlanningCache):
        raise ValueError("cache must be a PlanningCache")
    if cost_model is not None:
        if not isinstance(cost_model, CalibratedCostModel):
            raise ValueError("cost_model must be a CalibratedCostModel")
        cost_model.validate_environment()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if analysis not in ANALYSES:
        raise ValueError(f"Unknown analysis {analysis!r}; choose from {ANALYSES}")
    validate_limits(workers, max_state_qubits, max_state_bytes)
    requests = tuple(requests)
    if any(not isinstance(r, Request) for r in requests):
        raise ValueError("plan_all accepts expectation Requests")
    budget = None
    if (
        analysis == "budgeted"
        and strategy == "auto"
        and optimization == "none"
        and cost_model is None
        and cache is None
    ):
        budget = _analysis_budget(requests, max_state_qubits, max_state_bytes)
    cache_status = "disabled"
    analysed = "full" if strategy == "mps" else strategy
    if budget is not None and budget["skipped"]:
        analysis = _analyze(requests, "full")
    elif cache is None:
        analysis = _analyze_optimized(requests, analysed, optimization, options)
    else:
        key = (_cache_key(requests, analysed), optimization, options, identity)
        analysis = cache._get(key)
        cache_status = "hit" if analysis is not None else "miss"
        if analysis is None:
            analysis = _analyze_optimized(requests, analysed, optimization, options)
            cache._put(key, analysis)
    candidates = list(analysis.candidates)
    for i, candidate in enumerate(candidates):
        reasons = []
        if candidate.max_width > max_state_qubits:
            reasons.append(f"width {candidate.max_width} exceeds {max_state_qubits}")
        if candidate.state_bytes > max_state_bytes:
            reasons.append(f"state payload {candidate.state_bytes} exceeds {max_state_bytes} bytes")
        if reasons:
            candidates[i] = replace(candidate, rejection="; ".join(reasons))
    eligible = [
        c
        for c in candidates
        if c.rejection is None and (strategy == "auto" or c.strategy == strategy)
    ]
    # Exact MPS is built lazily: only as a fallback when no state-vector plan is
    # feasible, or when requested explicitly, so ordinary plans pay nothing for it.
    if strategy == "mps" or (strategy == "auto" and not eligible):
        mps = _mps_candidate(candidates, max_state_bytes)
        if mps is not None:
            candidates = [mps] if strategy == "mps" else [*candidates, mps]
            eligible = [mps] if mps.rejection is None else []
    if not eligible:
        detail = "; ".join(
            f"{c.strategy}: {c.rejection}"
            for c in candidates
            if strategy == "auto" or c.strategy == strategy
        )
        raise NoFeasiblePlan(detail)
    model_status = "heuristic"
    if eligible and eligible[0].representation == "mps":
        # Calibrated models cover state-vector widths only.
        model_status = "heuristic: exact MPS fallback, no state-vector plan is feasible"
    elif cost_model is not None:
        if all(cost_model.covers(c) for c in eligible):
            model_status = "calibrated"
            candidates = [
                replace(c, predicted_execution_ns=cost_model.predict(c))
                if c.rejection is None
                else c
                for c in candidates
            ]
            eligible = [c for c in candidates if c.rejection is None]
        else:
            model_status = "heuristic fallback: eligible candidate outside calibration width domain"
    selected = min(
        eligible,
        key=lambda c: (
            c.predicted_execution_ns if model_status == "calibrated" else c.score,
            len(c.tasks),
        ),
    )
    return Plan(
        selected,
        tuple(candidates),
        len(requests),
        analysis.constants,
        strategy,
        max_state_qubits,
        max_state_bytes,
        perf_counter_ns() - start,
        cost_model,
        model_status,
        cache_status,
        optimization,
        0 if cache_status == "hit" else analysis.optimization_ns,
        analysis.optimization_ns,
        analysis.rewrite_records,
        identity,
        options if optimization == "zx" else None,
        tuple(budget.items()) if budget is not None else (),
    )
