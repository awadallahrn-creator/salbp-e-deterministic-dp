"""
exact2.py — CORRECTED exact SALBP-2 (min cycle time for fixed m) via the
station-oriented DP over precedence-closed subsets.

Bug fixed relative to the first attempt: a station "block" B added to the
current closed set S is valid iff (S ∪ B) is itself precedence-closed, i.e.
every predecessor of every task in B lies in S ∪ B. This correctly allows a
task to enter a station together with its predecessors in the SAME station
(intra-station precedence is permitted in SALBP). The previous version only
considered tasks already available given S, which under-counted feasible
blocks and overestimated the optimum.

State: S = bitmask of scheduled tasks (always precedence-closed).
f(S) = min #stations to schedule exactly S, each load <= c.
f(0)=0 ; answer f(full). Transition: choose non-empty B ⊆ (¬S) with
load(B) <= c and (S∪B) closed; f(S) = 1 + min_B f(S∪B).

Validated against the brute-force oracle in brute.py.
"""
from __future__ import annotations
from typing import Optional, Dict, List
import math
from core import SALBPInstance


def _masks(inst: SALBPInstance):
    n = inst.n
    pred = [0] * n
    for (i, j) in inst.arcs:
        pred[j - 1] |= (1 << (i - 1))
    times = [inst.time_of(i) for i in range(1, n + 1)]
    return pred, times


def min_stations_for_cycle(inst: SALBPInstance, c: int,
                           node_budget: int = 2_000_000) -> Optional[int]:
    """Exact minimum number of stations to schedule all tasks with each
    station load <= c. None if some t_i > c, or budget exceeded."""
    n = inst.n
    pred, times = _masks(inst)
    if any(t > c for t in times):
        return None
    full = (1 << n) - 1
    memo: Dict[int, int] = {}
    nodes = [0]

    def gen_blocks(S: int) -> List[int]:
        """All non-empty B ⊆ ¬S with load(B)<=c and (S∪B) precedence-closed."""
        # candidate tasks: not in S, and whose predecessors are all in S
        # (a task whose preds are NOT all in S can still join if those preds
        #  also join the same block; we handle that by DFS over all not-in-S
        #  tasks and a closedness check at the end).
        cand = [i for i in range(n) if not (S & (1 << i))]
        blocks: List[int] = []

        def dfs(idx: int, load: int, B: int):
            if idx == len(cand):
                if B:
                    SB = S | B
                    # closedness: every task in B has preds within SB
                    ok = True
                    bb = B
                    while bb:
                        low = bb & (-bb)
                        t = low.bit_length() - 1
                        if (pred[t] & SB) != pred[t]:
                            ok = False
                            break
                        bb ^= low
                    if ok:
                        blocks.append(B)
                return
            # exclude
            dfs(idx + 1, load, B)
            # include cand[idx] if load fits
            ti = times[cand[idx]]
            if load + ti <= c:
                dfs(idx + 1, load + ti, B | (1 << cand[idx]))

        dfs(0, 0, 0)
        return blocks

    def f(S: int) -> int:
        if S == full:
            return 0
        if S in memo:
            return memo[S]
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise TimeoutError
        best = math.inf
        for B in gen_blocks(S):
            v = 1 + f(S | B)
            if v < best:
                best = v
        memo[S] = best
        return best

    try:
        import sys
        sys.setrecursionlimit(1 << 20)
        r = f(0)
        return r if r != math.inf else None
    except TimeoutError:
        return None


def exact_min_cycle_fixed_m(inst: SALBPInstance, m: int,
                            node_budget: int = 2_000_000) -> Optional[int]:
    """Smallest cycle c feasible within m stations (exact SALBP-2 optimum)."""
    S = inst.total_time
    lo, hi = max(inst.t_max, math.ceil(S / m)), S
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        ms = min_stations_for_cycle(inst, mid, node_budget)
        if ms is None:
            return None
        if ms <= m:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best


if __name__ == "__main__":
    from core import parse_in2
    from brute import brute_min_cycle_fixed_m
    import os
    base = "/home/claude/data/set2/precedence graphs"
    print("Validating exact2 against brute-force oracle:")
    allok = True
    for name in ["MERTENS", "JAESCHKE", "JACKSON"]:
        inst = parse_in2(os.path.join(base, name + ".IN2"))
        for m in [3, 4, 5]:
            cb = brute_min_cycle_fixed_m(inst, m)
            cd = exact_min_cycle_fixed_m(inst, m)
            ok = (cb == cd)
            allok &= ok
            print(f"  {name:9s} m={m}: brute c*={cb}  dp c*={cd}  {'OK' if ok else 'MISMATCH!!'}")
    print("ALL MATCH" if allok else "THERE ARE MISMATCHES")
