"""
core.py — SALBP instance representation and benchmark file parsers.

Supports the two standard public benchmark formats:
  * Scholl (1993) ".IN2"  — classic named instances
  * Otto et al. (2013) ".alb" — generated instances (SALBPGen)

An instance stores: number of tasks, integer task times, and a list of
DIRECT precedence arcs (i -> j), 1-based as in the source files.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import os


@dataclass
class SALBPInstance:
    name: str
    n: int                              # number of tasks
    times: List[int]                    # times[k] = time of task (k+1), 1-based externally
    arcs: List[Tuple[int, int]]         # direct precedence (i, j), 1-based
    cycle_hint: int | None = None       # cycle time if provided by the file (Otto), else None

    # ---- derived structures (built lazily) ----
    preds: Dict[int, Set[int]] = field(default_factory=dict, repr=False)
    succs: Dict[int, Set[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.preds = {i: set() for i in range(1, self.n + 1)}
        self.succs = {i: set() for i in range(1, self.n + 1)}
        for (i, j) in self.arcs:
            self.succs[i].add(j)
            self.preds[j].add(i)

    @property
    def total_time(self) -> int:
        return sum(self.times)

    @property
    def t_max(self) -> int:
        return max(self.times)

    def time_of(self, task: int) -> int:
        """task is 1-based."""
        return self.times[task - 1]


# ----------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------
def parse_in2(path: str) -> SALBPInstance:
    """Parse a Scholl (1993) .IN2 file.

    Format (per official README):
      line 1            : number n of tasks
      lines 2..n+1      : integer task times (one per line)
      following lines   : direct precedence "i,j"
      last line         : "-1,-1" end mark (optional)
    """
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", errors="replace") as fh:
        raw = [ln.strip() for ln in fh.read().replace("\r", "").split("\n")]
    raw = [ln for ln in raw if ln != ""]

    n = int(raw[0])
    times = [int(raw[1 + k]) for k in range(n)]

    arcs: List[Tuple[int, int]] = []
    for ln in raw[1 + n:]:
        if "," not in ln:
            continue
        a, b = ln.split(",")
        a, b = int(a), int(b)
        if a == -1 and b == -1:
            break
        arcs.append((a, b))

    return SALBPInstance(name=name, n=n, times=times, arcs=arcs, cycle_hint=None)


def parse_alb(path: str) -> SALBPInstance:
    """Parse an Otto et al. (2013) .alb file.

    Tagged sections: <number of tasks>, <cycle time>, <order strength>,
    <task times> (lines "idx time"), <precedence relations> (lines "i,j"),
    <end>.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", errors="replace") as fh:
        lines = [ln.strip() for ln in fh.read().replace("\r", "").split("\n")]

    section = None
    n = None
    cycle = None
    times_map: Dict[int, int] = {}
    arcs: List[Tuple[int, int]] = []

    for ln in lines:
        if ln == "":
            continue
        if ln.startswith("<") and ln.endswith(">"):
            section = ln.lower()
            continue
        if section == "<number of tasks>":
            n = int(ln)
        elif section == "<cycle time>":
            cycle = int(ln)
        elif section == "<task times>":
            parts = ln.split()
            idx, t = int(parts[0]), int(parts[1])
            times_map[idx] = t
        elif section == "<precedence relations>":
            if "," in ln:
                a, b = ln.split(",")
                arcs.append((int(a), int(b)))
        # other sections (order strength, end) ignored

    times = [times_map[k] for k in range(1, n + 1)]
    return SALBPInstance(name=name, n=n, times=times, arcs=arcs, cycle_hint=cycle)


def load_any(path: str) -> SALBPInstance:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".in2":
        return parse_in2(path)
    if ext == ".alb":
        return parse_alb(path)
    raise ValueError(f"Unsupported extension: {ext}")


# ----------------------------------------------------------------------
# Topological order (deterministic: smallest available index first)
# ----------------------------------------------------------------------
def topological_order(inst: SALBPInstance) -> List[int]:
    """Deterministic Kahn topological sort; ties broken by smallest index.
    Returns a list of 1-based task ids. Raises if the graph has a cycle."""
    import heapq
    indeg = {i: len(inst.preds[i]) for i in range(1, inst.n + 1)}
    ready = [i for i in range(1, inst.n + 1) if indeg[i] == 0]
    heapq.heapify(ready)
    order: List[int] = []
    while ready:
        u = heapq.heappop(ready)
        order.append(u)
        for v in sorted(inst.succs[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(ready, v)
    if len(order) != inst.n:
        raise ValueError(f"Precedence graph of {inst.name} is not a DAG")
    return order


if __name__ == "__main__":
    # Quick self-check on a known instance.
    inst = parse_in2("/home/claude/data/set2/precedence graphs/BUXEY.IN2")
    print(f"name={inst.name} n={inst.n} sum_t={inst.total_time} "
          f"t_max={inst.t_max} arcs={len(inst.arcs)}")
    order = topological_order(inst)
    print("topo (first 10):", order[:10])
    print("is permutation of 1..n:", sorted(order) == list(range(1, inst.n + 1)))
