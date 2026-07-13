"""
dp_heuristic.py — Deterministic Dynamic-Programming heuristic for SALBP-E.

IDEA (stated honestly as a heuristic, not an exact optimum):
We fix ONE deterministic topological order of the tasks. A station is then a
*contiguous segment* of that order. For a fixed number of stations m, we use
DP to partition the ordered sequence into m contiguous segments so that the
maximum segment load (the realized cycle time) is minimized.

Because every segment is a prefix-contiguous block of a topological order,
any precedence arc i->j has order(i) < order(j), so i lands in the same or an
earlier segment than j. Hence the partition is ALWAYS precedence-feasible.

This is a heuristic for SALBP-E because the optimum may require a task grouping
that is NOT contiguous in this particular order. We therefore MEASURE the
optimality gap against known bounds rather than claiming optimality.

DP recurrence (min-max partition of a sequence into m blocks):
    Let p[0..n] be prefix sums of the ordered task times.
    f(k, s) = minimal achievable "max block load" when the first k tasks are
              split into s contiguous blocks.
    f(k, s) = min over j<k of  max( f(j, s-1), p[k]-p[j] ).
    f(0,0)=0.
Complexity: O(n^2 * m) states-transitions; with the standard "feasibility of a
target cycle is monotone" reformulation we instead BINARY-SEARCH the cycle and
test it greedily in O(n) — giving the optimal min-max contiguous partition in
O(n log(sum_t)). Both are implemented; the binary-search version is the one
used for scalability, and the O(n^2 m) DP table is kept for verification.
"""

from __future__ import annotations
from typing import List, Tuple, Optional
import math
from core import SALBPInstance, topological_order
from evaluate import Solution, lower_bound_cycle


# ----------------------------------------------------------------------
# Min-max contiguous partition via DP table  (verification reference)
# ----------------------------------------------------------------------
def _minmax_partition_dp(prefix: List[int], m: int) -> Tuple[int, List[int]]:
    """Exact min-max partition of a fixed sequence into m contiguous blocks.
    Returns (max_block_load, cut_positions) where cut_positions has length m-1.
    prefix has length n+1, prefix[0]=0."""
    n = len(prefix) - 1
    INF = float("inf")
    # f[s][k] = min max-load splitting first k items into s blocks
    f = [[INF] * (n + 1) for _ in range(m + 1)]
    arg = [[-1] * (n + 1) for _ in range(m + 1)]
    f[0][0] = 0
    for s in range(1, m + 1):
        for k in range(s, n + 1):           # need at least s items for s blocks
            best, best_j = INF, -1
            for j in range(s - 1, k):
                blk = prefix[k] - prefix[j]
                cand = max(f[s - 1][j], blk)
                if cand < best:
                    best, best_j = cand, j
            f[s][k] = best
            arg[s][k] = best_j
    # backtrack cut positions
    cuts: List[int] = []
    k, s = n, m
    while s > 0:
        j = arg[s][k]
        cuts.append(j)
        k, s = j, s - 1
    cuts = sorted(c for c in cuts if 0 < c < n)
    return int(f[m][n]), cuts


# ----------------------------------------------------------------------
# Min-max contiguous partition via binary search  (scalable, O(n log S))
# ----------------------------------------------------------------------
def _feasible_with_cycle(times_ordered: List[int], m: int, c: int) -> Optional[List[int]]:
    """Can we split the ordered sequence into <= m contiguous blocks each with
    load <= c? Greedy. Returns cut positions if yes, else None."""
    cuts: List[int] = []
    blocks = 1
    cur = 0
    for idx, t in enumerate(times_ordered):
        if t > c:
            return None
        if cur + t <= c:
            cur += t
        else:
            blocks += 1
            cuts.append(idx)
            cur = t
            if blocks > m:
                return None
    return cuts


def _minmax_partition_binsearch(times_ordered: List[int], m: int) -> Tuple[int, List[int]]:
    lo = max(max(times_ordered), math.ceil(sum(times_ordered) / m))
    hi = sum(times_ordered)
    best_cuts: List[int] = []
    while lo < hi:
        mid = (lo + hi) // 2
        cuts = _feasible_with_cycle(times_ordered, m, mid)
        if cuts is not None:
            hi = mid
            best_cuts = cuts
        else:
            lo = mid + 1
    best_cuts = _feasible_with_cycle(times_ordered, m, lo) or best_cuts
    return lo, best_cuts


def _cuts_to_solution(inst: SALBPInstance, order: List[int],
                      cuts: List[int], m: int) -> Solution:
    """Turn cut positions (in the ordered sequence) into a Solution.
    Blocks: [0,cut0), [cut0,cut1), ... ; pad empty trailing stations if needed."""
    cuts = sorted(cuts)
    bounds = [0] + cuts + [len(order)]
    station_of = {}
    st = 0
    for b in range(len(bounds) - 1):
        st += 1
        for pos in range(bounds[b], bounds[b + 1]):
            station_of[order[pos]] = st
    used = st
    # If fewer blocks were produced than m, that's fine: efficiency uses the
    # actual m we report. We report the number of NON-EMPTY stations actually
    # needed, but keep m as requested for the E computation comparison.
    return Solution(inst=inst, station_of=station_of, m=m)


def solve_dp_fixed_m(inst: SALBPInstance, m: int,
                     method: str = "binsearch") -> Solution:
    """Deterministic DP heuristic for a FIXED number of stations m."""
    order = topological_order(inst)
    times_ordered = [inst.time_of(t) for t in order]
    if method == "table":
        prefix = [0]
        for t in times_ordered:
            prefix.append(prefix[-1] + t)
        _, cuts = _minmax_partition_dp(prefix, m)
    else:
        _, cuts = _minmax_partition_binsearch(times_ordered, m)
    return _cuts_to_solution(inst, order, cuts, m)


def solve_salbpE(inst: SALBPInstance,
                 m_min: Optional[int] = None,
                 m_max: Optional[int] = None,
                 method: str = "binsearch") -> Tuple[Solution, int]:
    """Full SALBP-E: search over candidate station counts m and return the
    assignment with the best line efficiency E. Deterministic.

    Default m range follows standard practice:
        m_min = ceil(sum_t / (2 * c_LB-ish))  -> we use a simple, safe band:
        m in [ ceil(sum_t / sum_t) .. n ] is too wide; instead we scan a
        practical band around the ideal, see below.
    Returns (best_solution, best_m).
    """
    if m_min is None:
        m_min = 2
    if m_max is None:
        # an assignment into m stations needs c>=t_max, so m up to
        # floor(sum_t / t_max) keeps stations meaningfully loaded; cap at n.
        m_max = max(m_min, min(inst.n, math.floor(inst.total_time / inst.t_max)))
    best: Optional[Solution] = None
    best_m = m_min
    for m in range(m_min, m_max + 1):
        sol = solve_dp_fixed_m(inst, m, method=method)
        if best is None or sol.efficiency > best.efficiency:
            best = sol
            best_m = m
    return best, best_m


if __name__ == "__main__":
    from core import parse_in2
    inst = parse_in2("/home/claude/data/set2/precedence graphs/BUXEY.IN2")
    # verify table vs binsearch agree on the min-max value for a few m
    order = topological_order(inst)
    to = [inst.time_of(t) for t in order]
    prefix = [0]
    for t in to:
        prefix.append(prefix[-1] + t)
    for m in [3, 5, 7, 10]:
        v1, _ = _minmax_partition_dp(prefix, m)
        v2, _ = _minmax_partition_binsearch(to, m)
        print(f"m={m:2d}  DP-table c={v1:4d}   binsearch c={v2:4d}   match={v1==v2}")
