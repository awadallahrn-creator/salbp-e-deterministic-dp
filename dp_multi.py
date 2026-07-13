"""
dp_multi.py — Deterministic multi-order extension of the DP heuristic.

Single-order DP fixes ONE canonical topological order. This extension
generates a SMALL, FIXED set of DIVERSE topological orders using different
DETERMINISTIC tie-break priorities, decodes each with the optimal min--max
partition, and keeps the best. No randomness is introduced, so the method
remains fully reproducible: the same input always yields the same output.

Tie-break priorities (all deterministic) used when several tasks are
simultaneously "ready" in Kahn's algorithm:
  R1  smallest index            (the original canonical order)
  R2  largest processing time   (heavy-first)
  R3  smallest processing time  (light-first)
  R4  most immediate successors (open up the graph early)
  R5  largest successor-subtree total time (critical-weight first)
  R6  largest index             (reverse canonical)

The extension is denoted DP-k, where k is the number of orders tried.
"""

from __future__ import annotations
from typing import List, Dict, Callable, Tuple
import heapq
from core import SALBPInstance
from evaluate import Solution
from baselines import decode_order


def _subtree_weight(inst: SALBPInstance) -> Dict[int, int]:
    """Total processing time of a task plus all its (transitive) successors.
    Computed in reverse topological order."""
    # reverse-topo via Kahn on reversed graph
    outdeg = {i: len(inst.succs[i]) for i in range(1, inst.n + 1)}
    ready = [i for i in range(1, inst.n + 1) if outdeg[i] == 0]
    w = {i: inst.time_of(i) for i in range(1, inst.n + 1)}
    order = []
    rl = list(ready)
    while rl:
        u = rl.pop()
        order.append(u)
        for p in inst.preds[u]:
            outdeg[p] -= 1
            if outdeg[p] == 0:
                rl.append(p)
    for u in order:
        for s in inst.succs[u]:
            w[u] += w[s]
    return w


def _ordered_kahn(inst: SALBPInstance, key: Callable[[int], tuple]) -> List[int]:
    """Deterministic topological order; among ready tasks pick the one with the
    smallest 'key' value. key must be a total order to keep determinism."""
    indeg = {i: len(inst.preds[i]) for i in range(1, inst.n + 1)}
    ready = [(key(i), i) for i in range(1, inst.n + 1) if indeg[i] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        _, u = heapq.heappop(ready)
        order.append(u)
        for v in inst.succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, (key(v), v))
    return order


def deterministic_orders(inst: SALBPInstance) -> Dict[str, List[int]]:
    """Return the fixed set of diverse deterministic topological orders."""
    sub = _subtree_weight(inst)
    rules: Dict[str, Callable[[int], tuple]] = {
        "R1_smallest_index": lambda i: (i,),
        "R2_heavy_first":     lambda i: (-inst.time_of(i), i),
        "R3_light_first":     lambda i: (inst.time_of(i), i),
        "R4_most_succ":       lambda i: (-len(inst.succs[i]), i),
        "R5_crit_weight":     lambda i: (-sub[i], i),
        "R6_largest_index":   lambda i: (-i,),
    }
    return {name: _ordered_kahn(inst, k) for name, k in rules.items()}


def solve_dpk_fixed_m(inst: SALBPInstance, m: int) -> Tuple[Solution, str]:
    """DP-k for a fixed station count: best min--max partition over the fixed
    set of deterministic orders. Returns (best_solution, winning_rule)."""
    best: Solution | None = None
    best_rule = ""
    for name, order in deterministic_orders(inst).items():
        sol = decode_order(inst, order, m)
        if best is None or sol.efficiency > best.efficiency:
            best, best_rule = sol, name
    return best, best_rule


def solve_dpk_salbpE(inst: SALBPInstance, m_min: int, m_max: int) -> Tuple[Solution, int, str]:
    best: Solution | None = None
    best_m, best_rule = m_min, ""
    for m in range(m_min, m_max + 1):
        sol, rule = solve_dpk_fixed_m(inst, m)
        if best is None or sol.efficiency > best.efficiency:
            best, best_m, best_rule = sol, m, rule
    return best, best_m, best_rule


if __name__ == "__main__":
    from core import parse_in2
    import os
    base = "/home/claude/data/set2/precedence graphs"
    print(f"{'instance':10s} {'m':>3s} {'DP1':>7s} {'DP-k':>7s} {'gain':>6s} {'winner':>16s}")
    for name in ["BUXEY", "SAWYER30", "KILBRID", "TONGE70", "WEE-MAG", "MUKHERJE"]:
        p = os.path.join(base, name + ".IN2")
        if not os.path.exists(p):
            continue
        inst = parse_in2(p)
        m = max(2, round(inst.n / 7))
        # single-order DP = R1 only
        dp1 = decode_order(inst, deterministic_orders(inst)["R1_smallest_index"], m)
        dpk, rule = solve_dpk_fixed_m(inst, m)
        gain = 100 * (dpk.efficiency - dp1.efficiency)
        print(f"{name:10s} {m:3d} {dp1.efficiency:7.4f} {dpk.efficiency:7.4f} "
              f"{gain:5.2f}% {rule:>16s} feas={dpk.is_precedence_feasible()}")
