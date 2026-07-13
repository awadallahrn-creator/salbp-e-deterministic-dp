"""
brute.py — Brute-force EXACT SALBP for tiny instances (verification oracle).

For a fixed number of stations m, enumerate every assignment of tasks to
stations 1..m that respects precedence (station(pred) <= station(succ)), and
return the minimum possible bottleneck load c = max_j L_j. This is the exact
SALBP-2 optimum for that m. Exponential (m^n) — use only for n <= ~10.

This function makes NO algorithmic shortcuts, so it is trusted as the oracle
against which the fast subset-DP solver is validated.
"""
from __future__ import annotations
from typing import Optional
from core import SALBPInstance, topological_order


def brute_min_cycle_fixed_m(inst: SALBPInstance, m: int) -> Optional[int]:
    order = topological_order(inst)          # assign in topo order
    times = [inst.time_of(t) for t in order]
    # predecessors expressed as positions in 'order'
    pos = {t: i for i, t in enumerate(order)}
    pred_pos = [[] for _ in range(inst.n)]
    for (i, j) in inst.arcs:
        pred_pos[pos[j]].append(pos[i])

    best = [float("inf")]
    station = [0] * inst.n   # station index (0-based) for each position

    def rec(idx: int, loads: list):
        if idx == inst.n:
            best[0] = min(best[0], max(loads))
            return
        # lower bound prune
        if max(loads) >= best[0]:
            return
        lo = 0
        for pp in pred_pos[idx]:
            lo = max(lo, station[pp])         # must be >= max predecessor station
        for s in range(lo, m):
            station[idx] = s
            loads[s] += times[idx]
            if loads[s] < best[0]:            # prune
                rec(idx + 1, loads)
            loads[s] -= times[idx]
        station[idx] = 0

    rec(0, [0] * m)
    return None if best[0] == float("inf") else int(best[0])


if __name__ == "__main__":
    from core import parse_in2
    import os
    base = "/home/claude/data/set2/precedence graphs"
    for name, ms in [("MERTENS", [3, 4, 5]), ("JAESCHKE", [3, 4, 5]),
                     ("JACKSON", [3, 4, 5])]:
        inst = parse_in2(os.path.join(base, name + ".IN2"))
        for m in ms:
            c = brute_min_cycle_fixed_m(inst, m)
            E = inst.total_time / (m * c) if c else None
            print(f"{name:9s} m={m} c*={c} E*={E:.4f}" if c else f"{name} m={m} infeasible")
