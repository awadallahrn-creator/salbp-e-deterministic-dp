"""
evaluate.py — SALBP-E solution representation, feasibility, and metrics.

A solution assigns each task to a station (1..m). We evaluate it under the
*correct* SALBP-E objective:

    realized cycle time  c   = max_j L_j          (the true bottleneck load)
    line efficiency      E   = sum_t / (m * c)    (maximize, in (0,1])
    smoothness index     SI  = sqrt( sum_j (c - L_j)^2 )   (lower = better balance)
    balance loss         BL  = (m*c - sum_t) / (m*c)       (= 1 - E)

This is the standard type-E formulation (Scholl & Becker 2006; Esmaeilbeigi
et al. 2015): efficiency is maximized by minimizing the maximum station load.
Note this is NOT a constant — it depends entirely on how tasks are grouped.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import math
from core import SALBPInstance


@dataclass
class Solution:
    inst: SALBPInstance
    station_of: Dict[int, int]      # task (1-based) -> station (1-based)
    m: int                          # number of stations used

    def loads(self) -> List[int]:
        L = [0] * self.m
        for task, st in self.station_of.items():
            L[st - 1] += self.inst.time_of(task)
        return L

    @property
    def realized_cycle(self) -> int:
        return max(self.loads())

    @property
    def efficiency(self) -> float:
        c = self.realized_cycle
        if c == 0:
            return 0.0
        return self.inst.total_time / (self.m * c)

    @property
    def smoothness_index(self) -> float:
        c = self.realized_cycle
        return math.sqrt(sum((c - L) ** 2 for L in self.loads()))

    @property
    def avg_idle(self) -> float:
        c = self.realized_cycle
        L = self.loads()
        return sum(c - x for x in L) / self.m

    @property
    def balance_loss(self) -> float:
        return 1.0 - self.efficiency

    def is_precedence_feasible(self) -> bool:
        """Every arc i->j must have station(i) <= station(j)."""
        for (i, j) in self.inst.arcs:
            if self.station_of[i] > self.station_of[j]:
                return False
        return True

    def respects_cycle(self, c: int) -> bool:
        return all(L <= c for L in self.loads())


def lower_bound_cycle(inst: SALBPInstance, m: int) -> int:
    """Trivial but valid lower bound on the optimal cycle time for given m:
       c >= max( t_max , ceil(sum_t / m) ).
    Any feasible assignment to m stations must have max load at least this."""
    return max(inst.t_max, math.ceil(inst.total_time / m))


def upper_bound_efficiency(inst: SALBPInstance, m: int) -> float:
    """Efficiency upper bound for given m, from the cycle lower bound.
       E <= sum_t / (m * c_LB)."""
    c_lb = lower_bound_cycle(inst, m)
    return inst.total_time / (m * c_lb)
