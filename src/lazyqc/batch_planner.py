"""Bounded batch search over request groups, bound rewrites and exact CX cuts."""

import json
from dataclasses import asdict, replace
from functools import lru_cache
from time import perf_counter_ns

from .batch import BatchPlan, assemble, price
from .cache import PlanningCache
from .cost import THREAD_KEYS, source_digest
from .cutting import (
    CutLimitError,
    CutOptions,
    compile_group,
    compile_recipes,
    cut_choices,
    cutting_identity,
)
from .ir import Pauli, Request
from .passes import causal_cone
from .plan import Task
from .rewrites import ZXOptions, optimizer_identity, rewrite_programs

# Nominal planning cost charged per joint beam expansion when converting a
# reuse-based time allowance into a deterministic expansion budget. Order of
# the per-expansion cost measured on reference hardware (~1 ms/state on the
# QAOA cutting workloads); the Weeks 3-5 cost model replaces it with a
# calibrated value.
PLANNING_NS_PER_EXPANSION = 1_000_000


@lru_cache(maxsize=1)
def implementation_identity():
    """Return the implementation fingerprint used by batch planning caches."""
    return source_digest("batch_planner.py", "batch.py", "cutting.py", "instruments.py")


def _freeze(value):
    """Convert nested lists to tuples for immutable structural templates."""
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    return value


def _erased_program_key(n_qubits, operations):
    """Return binding-free program shape: gate names and wires only."""
    return (n_qubits, tuple((g.name, tuple(g.qubits)) for g in operations))


class _HintStale(ValueError):
    """Recorded structural decisions do not apply to the current batch."""


def plan_batch(
    requests,
    *,
    cutting,
    strategy="auto",
    optimization="none",
    zx_options=None,
    workers=1,
    max_state_qubits=24,
    max_state_bytes=256 * 2**20,
    total_state_bytes=None,
    cache=None,
    cost_model=None,
    _structural_hints=None,
):
    """Plan a bounded batch of requests with optional cutting and parallel workers.

    The experimental ``_structural_hints`` template (as produced by
    :func:`structural_template` from a previous plan's explanation) replays
    recorded group partitions and cut choices instead of searching, for
    repeated topologies such as VQE angle updates. Mismatched topologies or
    options fall back to full planning.
    """
    from .planner import STRATEGIES, NoFeasiblePlan, _cache_key, validate_limits

    start = perf_counter_ns()
    if not isinstance(cutting, CutOptions):
        raise ValueError("cutting must be CutOptions")
    validate_limits(workers, max_state_qubits, max_state_bytes)
    total_state_bytes = max_state_bytes if total_state_bytes is None else total_state_bytes
    if type(total_state_bytes) is not int or total_state_bytes < 1:
        raise ValueError("total_state_bytes must be a positive integer")
    if strategy not in STRATEGIES or optimization not in ("none", "local", "zx"):
        raise ValueError("Invalid strategy or optimization")
    if zx_options is not None and (not isinstance(zx_options, ZXOptions) or optimization != "zx"):
        raise ValueError("ZXOptions requires optimization='zx'")
    if cache is not None and not isinstance(cache, PlanningCache):
        raise ValueError("cache must be PlanningCache")
    if cost_model is not None:
        from .batch_cost import BatchCostModel

        if not isinstance(cost_model, BatchCostModel):
            raise ValueError(
                "Cut/worker plans require a BatchCostModel calibrated on instrument branches"
            )
        cost_model.validate_environment()
        if workers > 1:
            if any(dict(cost_model.environment_items).get(k) != "1" for k in THREAD_KEYS):
                raise ValueError("Parallel calibration requires every native thread setting = 1")
    requests = tuple(requests)
    if len(requests) > cutting.max_requests or any(not isinstance(r, Request) for r in requests):
        raise ValueError("Provide expectation Requests within max_requests")
    # Every configuration-dependent decision is in this whole-plan cache key.
    identities = (
        ("batch", implementation_identity()),
        ("cutting", cutting_identity() if cutting.max_cuts else "no-cut-basis"),
        ("optimizer", optimizer_identity(optimization) if optimization != "none" else "none"),
    )
    options = zx_options or ZXOptions()
    key = (
        "batch-v1",
        _cache_key(requests, strategy),
        cutting,
        optimization,
        options,
        workers,
        max_state_qubits,
        max_state_bytes,
        total_state_bytes,
        identities,
        cost_model,
    )
    cached = cache._get(key) if cache else None
    if cached is not None:
        return replace(
            cached, planning_ns=perf_counter_ns() - start, cache_status="hit", optimization_ns=0
        )
    # Resolve even identity-only requests so unbound inputs cannot bypass validation.
    resolved = {r.program: r.program.resolved_operations() for r in requests}

    def resolve(program):
        """Return a program's bound operations, resolving each program once."""
        if program not in resolved:
            resolved[program] = program.resolved_operations()
        return resolved[program]

    constants = tuple((i, 1.0) for i, r in enumerate(requests) if not r.observable.factors)
    active = tuple(i for i, r in enumerate(requests) if r.observable.factors)
    variants = [("none", requests)]
    records, rewrite_ns = (), 0
    if optimization != "none":
        local, zx, records, rewrite_ns = rewrite_programs(requests, optimization, options)
        if local != requests:
            variants.append(("local", local))
        if optimization == "zx" and zx != local and zx != requests:
            variants.append(("zx", zx))
    if cutting.search == "fixed":
        variants = variants[-1:]  # Explicit fixed rewrite-then-cut baseline.
    # Structural hints may cross angle changes only if the equality partition
    # of concrete preparations is unchanged. Record both erased shape and the
    # canonical class of equal bound operations for each request.
    topology = []
    for name, batch in variants:
        program_shapes, bound_programs, order, outputs = {}, {}, [], []
        for r in batch:
            operations = resolve(r.program)
            shape = _erased_program_key(r.program.n_qubits, operations)
            if shape not in program_shapes:
                program_shapes[shape] = len(order)
                order.append(shape)
            bound_key = (r.program.n_qubits, operations)
            if bound_key not in bound_programs:
                bound_programs[bound_key] = len(bound_programs)
            outputs.append(
                (
                    program_shapes[shape],
                    bound_programs[bound_key],
                    tuple(r.observable.factors),
                )
            )
        topology.append((name, tuple(order), tuple(outputs)))
    topology_key = _freeze(tuple(topology))
    # Accept structural hints only on exact topology/options match and for the
    # mainstream auto/separate path with active requests. Rejections are
    # recorded once analysis_stats exists below, for hit-rate measurement.
    hinted = None
    hints_rejected = False
    if _structural_hints is not None and not isinstance(_structural_hints, dict):
        raise ValueError("structural hints must be a template dict")
    if isinstance(_structural_hints, dict) and strategy in ("auto", "separate") and active:
        try:
            guards_match = (
                _freeze(_structural_hints["topology"]) == topology_key
                and _structural_hints.get("strategy") == strategy
                and _structural_hints.get("optimization") == optimization
                and _structural_hints.get("search") == cutting.search
                and _freeze(_structural_hints.get("cutting")) == _freeze(asdict(cutting))
                and _structural_hints.get("workers") == workers
                and _structural_hints.get("max_state_qubits") == max_state_qubits
                and _structural_hints.get("max_state_bytes") == max_state_bytes
                and _structural_hints.get("total_state_bytes") == total_state_bytes
                and _structural_hints.get("cost_profile")
                == (cost_model.profile_id if cost_model is not None else None)
            )
        except (KeyError, TypeError, ValueError):
            guards_match = False
        if guards_match:
            hinted = _structural_hints
        else:
            hints_rejected = True
    # Memory constraints become a fragment-width bound before compiling branches.
    payload_width = min(max_state_bytes, total_state_bytes).bit_length() - 1 - 4
    width = min(cutting.max_fragment_qubits, max_state_qubits, payload_width)
    if active and width < 1:
        raise NoFeasiblePlan("No single-qubit fragment fits the state-payload budget")
    effective = replace(cutting, max_fragment_qubits=max(1, width))
    deadline = perf_counter_ns() + int(cutting.soft_timeout_ms * 1e6)
    compiled, rejections = {}, []
    preparations, formulas, priced = {}, {}, {}
    analysis_stats = {"recipe_builds": 0, "recipe_hits": 0, "candidate_hits": 0}
    if hints_rejected:
        analysis_stats["structural_miss"] = 1
    states_count, exhausted = 0, False
    beam_exhausted = False

    def group_choices(variant, indices, opts=None, pinned=None):
        """Enumerate feasible groupings for one candidate variant.

        With ``pinned=(cuts, parts)``, compile exactly that cut option instead
        of enumerating proposals (structural-hint replay).
        """
        nonlocal exhausted
        effective_opts = opts or effective
        name, batch = variants[variant]
        p = batch[indices[0]].program
        gates = resolve(p)
        group_key = (
            name,
            p.n_qubits,
            gates,
            tuple((i, batch[i].observable) for i in indices),
            strategy,
            effective_opts.max_cuts,
            effective_opts.max_fragment_qubits,
        )
        if group_key in compiled:
            return compiled[group_key]
        support = set(q for i in indices for q, _ in batch[i].observable.factors)
        wires, cone = (
            (tuple(range(p.n_qubits)), gates)
            if strategy == "full"
            else causal_cone(gates, Pauli(tuple((q, "Z") for q in sorted(support))))
        )
        task = Task(wires, cone, tuple((i, batch[i].observable) for i in indices))
        choices = []
        if pinned is not None:
            pinned_cuts = tuple(pinned[0])
            pinned_parts = tuple(tuple(p) for p in pinned[1])
            options_iter = [(pinned_cuts, pinned_parts)]
        else:
            options_iter = cut_choices(task, effective_opts, rejections)
        for cuts, parts in options_iter:
            # Preserve the cheap zero-cut fallback even when search expires.
            recipe_key = (wires, cone, cuts, parts)
            if cuts and recipe_key not in preparations and perf_counter_ns() >= deadline:
                exhausted = True
                break
            try:
                if recipe_key not in preparations:
                    preparations[recipe_key] = compile_recipes(
                        task, cuts, parts, effective_opts
                    )
                    formulas[recipe_key] = {}
                    analysis_stats["recipe_builds"] += 1
                else:
                    analysis_stats["recipe_hits"] += 1
                choices.append(
                    compile_group(
                        task,
                        cuts,
                        parts,
                        effective_opts,
                        name,
                        recipes=preparations[recipe_key],
                        formula_cache=formulas[recipe_key],
                    )
                )
            except CutLimitError as exc:
                rejections.append(str(exc))
        compiled[group_key] = tuple(choices)
        return compiled[group_key]

    def candidate(groups, label):
        """Assemble and price one batch candidate from selected groups."""
        groups = tuple(groups)
        if groups not in priced:
            priced[groups] = price(
                assemble(groups, effective, label), workers, total_state_bytes, cost_model
            )
        else:
            analysis_stats["candidate_hits"] += 1
        return replace(priced[groups], strategy=label)

    def complete_candidate(plan):
        """Whether a candidate reconstructs every nonconstant request."""
        return {r.output for r in plan.reconstruction} == set(active)

    def best_group(choices):
        """Return the least costly feasible grouping from a choice sequence."""
        ranked = []
        for g in choices:
            try:
                ranked.append((candidate((g,), "local-choice").score, g))
            except CutLimitError as exc:
                rejections.append(str(exc))
        return min(ranked, key=lambda pair: pair[0])[1] if ranked else None

    # Construct per-variant grouping options. Independent mode restricts groups to
    # singletons; joint mode includes identical cones, bounded overlapping unions and
    # whole-program groups. Explicit full/grouped strategies retain their contract.
    groups_by_variant, fixed_partitions = [], []
    variant_cone_groups = []
    for vi, (_, batch) in enumerate(variants):
        by_program = {}
        for i in active:
            p = batch[i].program
            by_program.setdefault((p.n_qubits, resolve(p)), []).append(i)
        definitions, fixed_groups = set(), []
        cone_groups = []
        for (_, ops), indices in by_program.items():
            cones = {i: causal_cone(ops, batch[i].observable) for i in indices}
            identical = {}
            for i in indices:
                identical.setdefault(cones[i], []).append(i)
            cone_groups.append(([set(key[0]) for key in identical], tuple(indices)))
            fixed_groups.extend(
                [tuple(indices)]
                if strategy == "full"
                else [(i,) for i in indices]
                if strategy == "separate"
                else [tuple(g) for g in identical.values()]
            )
            if strategy == "full":
                definitions.add(tuple(indices))
            elif strategy == "grouped" or cutting.search == "fixed":
                definitions.update(tuple(g) for g in identical.values())
            else:
                definitions.update((i,) for i in indices)
                if strategy == "auto" and cutting.search == "joint":
                    definitions.add(tuple(indices))
                    definitions.update(tuple(g) for g in identical.values())
                    for i in indices:
                        neighbours = sorted(
                            (j for j in indices if j > i and set(cones[i][0]) & set(cones[j][0])),
                            key=lambda j: (-len(set(cones[i][0]) & set(cones[j][0])), j),
                        )
                        for size in range(1, min(len(neighbours), cutting.max_group_size - 1) + 1):
                            definitions.add(tuple(sorted((i, *neighbours[:size]))))
        groups_by_variant.append(sorted(definitions, key=lambda g: (g[0], len(g), g)))
        fixed_partitions.append(fixed_groups)
        variant_cone_groups.append(cone_groups)
    # Establish the fixed rewrite-then-cut incumbent before broader search can
    # spend its budget. It shares preparation/formula caches with the rest.
    complete = []
    use_hints = hinted is not None
    if use_hints:
        try:
            variant_index = {name: vi for vi, (name, _) in enumerate(variants)}
            hint_groups = []
            for role, label in (
                ("baseline", "independent" if cutting.search == "joint" else cutting.search),
                ("selected", hinted["selected_label"]),
            ):
                groups = []
                for element in hinted[role]:
                    vi = variant_index[element["variant"]]
                    idx = tuple(element["indices"])
                    pinned = (
                        tuple(element["cuts"]),
                        tuple(tuple(p) for p in element["parts"]),
                    )
                    choices = group_choices(vi, idx, pinned=pinned)
                    if not choices:
                        raise _HintStale(f"no compiled choice for {element}")
                    groups.append(choices[0])
                plan = candidate(groups, label)
                if not complete_candidate(plan):
                    raise _HintStale("template does not cover every active output")
                hint_groups.append(plan)
            complete.extend(hint_groups)
            analysis_stats["structural_hit"] = 1
        except (KeyError, TypeError, ValueError, _HintStale) as exc:
            rejections.append(f"structural hints stale: {exc}")
            analysis_stats["structural_miss"] = 1
            use_hints = False
            complete = []
    if cutting.search == "joint" and not use_hints:
        fixed = []
        for indices in fixed_partitions[-1]:
            selected = best_group(group_choices(len(variants) - 1, indices))
            if selected is None:
                fixed = None
                break
            fixed.append(selected)
        if fixed is not None:
            try:
                plan = candidate(fixed, "fixed-fallback")
                if complete_candidate(plan):
                    complete.append(plan)
            except CutLimitError as exc:
                rejections.append(str(exc))
    # Independent search then deduplication is a complete retained baseline. For
    # constrained grouping, use the fixed grouping of that variant instead.
    # Hinted runs already built baseline and selected candidates above.
    baseline_groups = []
    if use_hints:
        baseline_groups = None
    elif strategy in ("auto", "separate") and cutting.search != "fixed":
        for i in active:
            choices = [g for vi in range(len(variants)) for g in group_choices(vi, (i,))]
            selected = best_group(choices)
            if selected is None:
                baseline_groups = None
                break
            baseline_groups.append(selected)
    else:
        for vi, definitions in enumerate(groups_by_variant):
            fixed = []
            for group in definitions:
                selected = best_group(group_choices(vi, group))
                if selected is None:
                    fixed = None
                    break
                fixed.append(selected)
            if fixed is not None:
                try:
                    complete.append(candidate(fixed, cutting.search))
                except CutLimitError as exc:
                    rejections.append(str(exc))
        baseline_groups = None
    if baseline_groups is not None:
        try:
            complete.append(
                candidate(
                    baseline_groups, "independent" if cutting.search == "joint" else cutting.search
                )
            )
        except CutLimitError as exc:
            rejections.append(str(exc))
    # Disjoint fast path: when distinct identical-cone groups are pairwise
    # wire-disjoint, they share no recipes by construction, so the joint beam
    # can only re-derive singletons (already complete via the baseline) plus
    # whole-program unions. Evaluate unions with zero cuts only; a union
    # needing cuts on disjoint wires is skipped as a documented heuristic
    # (measured, not proven: see disjoint_batch64).
    disjoint_fastpath = (
        cutting.search == "joint"
        and len(active) >= 1
        and all(
            a.isdisjoint(b)
            for variant_groups in variant_cone_groups
            for wire_sets, _ in variant_groups
            for idx, a in enumerate(wire_sets)
            for b in wire_sets[idx + 1 :]
        )
    )
    if disjoint_fastpath and not use_hints:
        analysis_stats["disjoint_fastpath"] = 1
        uncut = replace(effective, max_cuts=0)
        for vi, (_, batch) in enumerate(variants):
            unions = []
            for _, indices in variant_cone_groups[vi]:
                try:
                    union = [
                        g for g in group_choices(vi, tuple(indices), uncut) if not g.cut_indices
                    ]
                    if union:
                        unions.append(union[0])
                except CutLimitError as exc:
                    rejections.append(f"disjoint fastpath skipped union: {exc}")
            if unions:
                plan = candidate(tuple(unions), "joint-union")
                if complete_candidate(plan):
                    complete.append(plan)
    economic_state_cap = cutting.max_search_states
    # Cost-model v2 input: split incumbent construction from marginal beam
    # exploration. Marginal search cost is planning_ns minus this timestamp.
    analysis_stats["incumbent_build_ns"] = perf_counter_ns() - start
    if complete and cost_model is not None and cutting.expected_collections is not None:
        # Bound additional exploration by an upper bound on all execution cost
        # that could possibly be saved across the declared number of collections.
        allowance_ns = int(min(c.score for c in complete) * cutting.expected_collections)
        # Deterministic instrument: express the allowance as extra beam
        # expansions instead of wall-clock time, so identical inputs explore
        # identically and seed plan caches deterministically. Each expansion is
        # charged a nominal planning cost (order of measured per-expansion cost
        # on reference hardware; the Weeks 3-5 cost model calibrates it
        # properly), and exploration is additionally capped structurally per
        # expected collection. Near-zero predicted savings therefore stop
        # exploration instead of spending a fixed structural budget. The soft
        # wall-clock timeout remains only as a safety backstop.
        expansion_cost = PLANNING_NS_PER_EXPANSION
        if cost_model.planning_ns_per_expansion is not None:
            expansion_cost = cost_model.planning_ns_per_expansion
        economic_state_cap = min(
            cutting.max_search_states,
            cutting.expected_collections * cutting.beam_width * max(1, len(active)),
            int(allowance_ns // expansion_cost),
        )
        analysis_stats["economic_allowance_ns"] = allowance_ns
        analysis_stats["economic_state_cap"] = economic_state_cap
    if cutting.search == "joint" and not disjoint_fastpath and not use_hints:
        frontier = [()]
        for anchor in active:
            expanded = []
            for groups in frontier:
                covered = {i for g in groups for i in g.outputs}
                if anchor in covered:
                    expanded.append(groups)
                    continue
                for vi, definitions in enumerate(groups_by_variant):
                    for indices in definitions:
                        if states_count >= economic_state_cap or perf_counter_ns() >= deadline:
                            beam_exhausted = exhausted = True
                            break
                        if indices[0] != anchor or covered.intersection(indices):
                            continue
                        for group in group_choices(vi, indices):
                            if states_count >= economic_state_cap or perf_counter_ns() >= deadline:
                                beam_exhausted = exhausted = True
                                break
                            states_count += 1
                            expanded.append(groups + (group,))
                        if exhausted:
                            break
                    if exhausted:
                        break
                if exhausted:
                    break
            ranked = []
            for groups in expanded:
                try:
                    scored = candidate(groups, "joint")
                    ranked.append((scored.score, groups))
                    if complete_candidate(scored):
                        complete.append(scored)
                except CutLimitError as exc:
                    rejections.append(str(exc))
            if exhausted or not ranked:
                break
            frontier = [
                groups
                for _, groups in sorted(ranked, key=lambda pair: pair[0])[: cutting.beam_width]
            ]
    complete = [plan for plan in complete if complete_candidate(plan)]
    if not complete:
        detail = "; ".join(dict.fromkeys(rejections))
        raise NoFeasiblePlan(
            "No executable cut plan found within fragment/cut/branch/search limits"
            + (": " + detail if detail else "")
        )
    # Deduplicate complete physical candidates without changing their result ordering.
    unique = {}
    for c in complete:
        unique.setdefault((c.tasks, c.reconstruction), c)
    alternatives = tuple(sorted(unique.values(), key=lambda c: (c.score, len(c.tasks))))
    selected = alternatives[0]
    if any(t.width > max_state_qubits or t.state_bytes > max_state_bytes for t in selected.tasks):
        raise NoFeasiblePlan("Selected experiment exceeds state limits")

    def partition_of(candidate):
        """Extract replayable (variant, indices, cuts, parts) per group."""
        elements = []
        for group in candidate.groups:
            elements.append(
                {
                    "variant": group.optimization,
                    "indices": sorted({formula.output for formula in group.formulas}),
                    "cuts": list(group.cut_indices),
                    "parts": [list(part) for part in group.components],
                }
            )
        return elements

    baseline_template = next(
        (c for c in alternatives if c.label == "independent"), selected
    )
    structural_template = json.dumps(
        {
            "topology": topology_key,
            "strategy": strategy,
            "optimization": optimization,
            "search": cutting.search,
            "cutting": asdict(cutting),
            "workers": workers,
            "max_state_qubits": max_state_qubits,
            "max_state_bytes": max_state_bytes,
            "total_state_bytes": total_state_bytes,
            "cost_profile": cost_model.profile_id if cost_model is not None else None,
            "baseline": partition_of(baseline_template),
            "selected": partition_of(selected),
            "selected_label": selected.label,
        },
        allow_nan=False,
    )
    plan = BatchPlan(
        selected,
        alternatives,
        len(requests),
        constants,
        max_state_qubits,
        max_state_bytes,
        workers,
        total_state_bytes,
        cutting,
        perf_counter_ns() - start,
        states_count,
        exhausted or any("budget exhausted" in r for r in rejections),
        cost_model,
        "miss" if cache else "disabled",
        rewrite_ns,
        rewrite_ns,
        records,
        identities,
        tuple(dict.fromkeys(rejections)),
        tuple(analysis_stats.items()),
        beam_exhausted,
        structural_template,
    )
    if cache:
        cache._put(key, plan)
    return plan
