"""
baselines.py — Greedy, GA, ACO baselines for SALBP-E.

Design (fair and honest comparison):
  * All methods operate on the space of PRECEDENCE-FEASIBLE task orders.
  * A SHARED decoder maps any feasible order + station count m to a solution
    via the optimal min--max contiguous partition (the same operator the DP
    heuristic uses). This isolates the single factor under study: the value of
    *searching over orders* versus using one canonical topological order.

  - DP heuristic : ONE deterministic topological order -> decoder.   (in dp_heuristic.py)
  - Greedy       : canonical order + simple first-fit packing at target
                   cycle ceil(sum_t/m) (textbook constructive; intentionally
                   does NOT optimize the cut -> a genuinely weaker reference).
  - GA           : evolves feasible orders (OX crossover, swap mutation),
                   fitness = efficiency via the shared decoder. Stochastic.
  - ACO          : ants build feasible orders guided by pheromone + heuristic,
                   decoded by the shared decoder. Stochastic.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import random
import math
from core import SALBPInstance, topological_order
from evaluate import Solution
from dp_heuristic import _minmax_partition_binsearch, _cuts_to_solution


# ----------------------------------------------------------------------
# Shared decoder
# ----------------------------------------------------------------------
def decode_order(inst: SALBPInstance, order: List[int], m: int) -> Solution:
    """Optimal min--max partition of a fixed feasible order into m stations."""
    times_ordered = [inst.time_of(t) for t in order]
    _, cuts = _minmax_partition_binsearch(times_ordered, m)
    return _cuts_to_solution(inst, order, cuts, m)


def random_topo_order(inst: SALBPInstance, rng: random.Random) -> List[int]:
    """A random precedence-feasible order (randomized Kahn)."""
    indeg = {i: len(inst.preds[i]) for i in range(1, inst.n + 1)}
    ready = [i for i in range(1, inst.n + 1) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        k = rng.randrange(len(ready))
        u = ready.pop(k)
        order.append(u)
        for v in sorted(inst.succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return order


def is_feasible_order(inst: SALBPInstance, order: List[int]) -> bool:
    pos = {t: p for p, t in enumerate(order)}
    return all(pos[i] < pos[j] for (i, j) in inst.arcs)


# ----------------------------------------------------------------------
# Greedy (textbook constructive, deterministic)
# ----------------------------------------------------------------------
def solve_greedy_fixed_m(inst: SALBPInstance, m: int) -> Solution:
    """First-fit packing along the canonical topological order at the naive
    target cycle ceil(sum_t/m). Opens a new station when the target is
    exceeded; if it runs past m stations, the leftover tasks pile into the
    last station (so the realized max load may exceed the target -> weaker E)."""
    order = topological_order(inst)
    target = math.ceil(inst.total_time / m)
    station_of = {}
    s = 1
    load = 0
    for t in order:
        tt = inst.time_of(t)
        if load + tt > target and s < m:
            s += 1
            load = 0
        station_of[t] = s
        load += tt
    return Solution(inst=inst, station_of=station_of, m=m)


# ----------------------------------------------------------------------
# Genetic Algorithm (stochastic)
# ----------------------------------------------------------------------
def _ox_crossover(p1: List[int], p2: List[int], inst: SALBPInstance,
                  rng: random.Random) -> List[int]:
    """Order crossover that REPAIRS to a feasible order: take a slice from p1,
    fill the rest in p2's relative order, then repair via a feasibility pass."""
    n = len(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    taken = set(p1[a:b + 1])
    fill = [x for x in p2 if x not in taken]
    fi = 0
    for k in range(n):
        if child[k] is None:
            child[k] = fill[fi]
            fi += 1
    # repair to feasibility: stable reorder by a topological pass using child as priority
    return _repair_to_topo(inst, child)


def _repair_to_topo(inst: SALBPInstance, priority: List[int]) -> List[int]:
    """Produce a feasible order that respects 'priority' as much as possible:
    repeatedly pick the highest-priority task whose predecessors are all placed."""
    rank = {t: r for r, t in enumerate(priority)}
    indeg = {i: len(inst.preds[i]) for i in range(1, inst.n + 1)}
    ready = [i for i in range(1, inst.n + 1) if indeg[i] == 0]
    order: List[int] = []
    while ready:
        ready.sort(key=lambda t: rank[t])
        u = ready.pop(0)
        order.append(u)
        for v in inst.succs[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
    return order


def _swap_mutation(order: List[int], inst: SALBPInstance,
                   rng: random.Random) -> List[int]:
    o = order[:]
    if len(o) >= 2:
        i, j = rng.sample(range(len(o)), 2)
        o[i], o[j] = o[j], o[i]
    return _repair_to_topo(inst, o)


def solve_ga_fixed_m(inst: SALBPInstance, m: int, rng: random.Random,
                     pop_size: int = 30, generations: int = 60,
                     mutation_rate: float = 0.10,
                     elite: int = 2) -> Solution:
    pop = [random_topo_order(inst, rng) for _ in range(pop_size)]

    def fit(order):
        return decode_order(inst, order, m).efficiency

    scored = sorted(((fit(o), o) for o in pop), key=lambda x: -x[0])
    for _ in range(generations):
        new = [scored[i][1] for i in range(min(elite, len(scored)))]
        while len(new) < pop_size:
            # tournament selection
            cand = rng.sample(scored, min(4, len(scored)))
            p1 = max(cand, key=lambda x: x[0])[1]
            cand = rng.sample(scored, min(4, len(scored)))
            p2 = max(cand, key=lambda x: x[0])[1]
            child = _ox_crossover(p1, p2, inst, rng)
            if rng.random() < mutation_rate:
                child = _swap_mutation(child, inst, rng)
            new.append(child)
        pop = new
        scored = sorted(((fit(o), o) for o in pop), key=lambda x: -x[0])
    best_order = scored[0][1]
    return decode_order(inst, best_order, m)


# ----------------------------------------------------------------------
# Ant Colony Optimization (stochastic)
# ----------------------------------------------------------------------
def solve_aco_fixed_m(inst: SALBPInstance, m: int, rng: random.Random,
                      n_ants: int = 20, iterations: int = 40,
                      alpha: float = 1.0, beta: float = 2.0,
                      rho: float = 0.3, q: float = 1.0) -> Solution:
    n = inst.n
    # pheromone on (position-independent) task desirability; heuristic = task time
    tau = {i: 1.0 for i in range(1, n + 1)}
    eta = {i: inst.time_of(i) for i in range(1, n + 1)}  # prefer heavier tasks early
    best_order = None
    best_E = -1.0

    def construct() -> List[int]:
        indeg = {i: len(inst.preds[i]) for i in range(1, n + 1)}
        ready = [i for i in range(1, n + 1) if indeg[i] == 0]
        order = []
        while ready:
            weights = [(tau[t] ** alpha) * (eta[t] ** beta) for t in ready]
            tot = sum(weights)
            r = rng.random() * tot
            acc = 0.0
            pick = len(ready) - 1
            for idx, w in enumerate(weights):
                acc += w
                if acc >= r:
                    pick = idx
                    break
            u = ready.pop(pick)
            order.append(u)
            for v in inst.succs[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    ready.append(v)
        return order

    for _ in range(iterations):
        iter_best_order, iter_best_E = None, -1.0
        for _a in range(n_ants):
            order = construct()
            E = decode_order(inst, order, m).efficiency
            if E > iter_best_E:
                iter_best_order, iter_best_E = order, E
        # evaporate
        for i in tau:
            tau[i] *= (1 - rho)
        # deposit along iteration-best
        if iter_best_order is not None:
            for t in iter_best_order:
                tau[t] += q * iter_best_E
            if iter_best_E > best_E:
                best_order, best_E = iter_best_order, iter_best_E
    return decode_order(inst, best_order, m)
