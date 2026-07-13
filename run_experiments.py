"""
run_experiments.py — produces ALL real results for the paper.

Experiment 1 (Quality + Reproducibility): all 25 Scholl instances, several
station counts m. DP and Greedy deterministic; GA and ACO repeated R times
(mean, std, best). Records efficiency E, realized cycle, optimality-gap proxy
vs lower bound, runtime, and reproducibility (std of E).

Experiment 2 (Scalability): Otto instances n in {20,50,100,1000}, fixed m per
size, measure runtime growth of each method (metaheuristics on a fixed budget).
"""

from __future__ import annotations
import os, glob, time, json, random, statistics, csv
from core import parse_in2, parse_alb
from evaluate import lower_bound_cycle
from dp_heuristic import solve_dp_fixed_m
from baselines import solve_greedy_fixed_m, solve_ga_fixed_m, solve_aco_fixed_m

SCHOLL_DIR = "/home/claude/data/set2/precedence graphs"
OTTO_GLOB = {
    20:   "/home/claude/data/set1/*n=20*/*/instance_n=20_*.alb",
    50:   "/home/claude/data/set1/*n=50_x/*/instance_n=50_*.alb",
    100:  "/home/claude/data/set1/*n=100*/*/instance_n=100_*.alb",
    1000: "/home/claude/data/set1/*n=1000*/*/instance_n=1000_*.alb",
}
R_RUNS = 10              # repetitions for stochastic methods
SEED0 = 20250101
OUT = "/home/claude/salbp/results"
os.makedirs(OUT, exist_ok=True)


def candidate_m_values(n: int):
    """A small band of station counts that keep instances non-trivial."""
    if n <= 12:
        return [3, 4]
    if n <= 35:
        return [4, 6, 8]
    if n <= 90:
        return [6, 10, 14]
    return [10, 16, 22]


def eff_gap(inst, sol, m):
    """Optimality-gap proxy (%): (E_UB - E)/E_UB * 100, E_UB from cycle LB."""
    c_lb = lower_bound_cycle(inst, m)
    E_ub = inst.total_time / (m * c_lb)
    return 100.0 * (E_ub - sol.efficiency) / E_ub, E_ub


def run_quality():
    files = sorted(glob.glob(os.path.join(SCHOLL_DIR, "*.IN2")))
    rows = []
    for fp in files:
        inst = parse_in2(fp)
        for m in candidate_m_values(inst.n):
            if m >= inst.n:
                continue
            # Deterministic
            t0 = time.perf_counter(); dp = solve_dp_fixed_m(inst, m); dp_ms = (time.perf_counter()-t0)*1000
            t0 = time.perf_counter(); gr = solve_greedy_fixed_m(inst, m); gr_ms = (time.perf_counter()-t0)*1000
            dp_gap, E_ub = eff_gap(inst, dp, m)
            gr_gap, _ = eff_gap(inst, gr, m)

            # Stochastic: R runs
            ga_E, aco_E, ga_ms, aco_ms = [], [], [], []
            for r in range(R_RUNS):
                rng = random.Random(SEED0 + r)
                t0 = time.perf_counter(); s = solve_ga_fixed_m(inst, m, rng); ga_ms.append((time.perf_counter()-t0)*1000); ga_E.append(s.efficiency)
                rng = random.Random(SEED0 + 1000 + r)
                t0 = time.perf_counter(); s = solve_aco_fixed_m(inst, m, rng); aco_ms.append((time.perf_counter()-t0)*1000); aco_E.append(s.efficiency)

            rows.append({
                "instance": inst.name, "n": inst.n, "m": m,
                "sum_t": inst.total_time, "E_ub": round(E_ub, 4),
                "DP_E": round(dp.efficiency, 4), "DP_gap": round(dp_gap, 2),
                "DP_c": dp.realized_cycle, "DP_ms": round(dp_ms, 3), "DP_std": 0.0,
                "GR_E": round(gr.efficiency, 4), "GR_gap": round(gr_gap, 2),
                "GR_c": gr.realized_cycle, "GR_ms": round(gr_ms, 3), "GR_std": 0.0,
                "GA_E_mean": round(statistics.mean(ga_E), 4),
                "GA_E_std": round(statistics.pstdev(ga_E), 4),
                "GA_E_best": round(max(ga_E), 4),
                "GA_ms": round(statistics.mean(ga_ms), 3),
                "ACO_E_mean": round(statistics.mean(aco_E), 4),
                "ACO_E_std": round(statistics.pstdev(aco_E), 4),
                "ACO_E_best": round(max(aco_E), 4),
                "ACO_ms": round(statistics.mean(aco_ms), 3),
            })
            print(f"{inst.name:10s} n={inst.n:4d} m={m:2d} | "
                  f"DP={dp.efficiency:.3f} GR={gr.efficiency:.3f} "
                  f"GA={statistics.mean(ga_E):.3f}±{statistics.pstdev(ga_E):.3f} "
                  f"ACO={statistics.mean(aco_E):.3f}±{statistics.pstdev(aco_E):.3f}")
    with open(os.path.join(OUT, "quality.json"), "w") as f:
        json.dump(rows, f, indent=2)
    if rows:
        with open(os.path.join(OUT, "quality.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows


def run_scalability():
    rows = []
    for n, pattern in OTTO_GLOB.items():
        files = sorted(glob.glob(pattern))[:5]   # 5 instances per size
        if not files:
            print(f"[warn] no Otto files for n={n} ({pattern})")
            continue
        m = {20: 4, 50: 8, 100: 12, 1000: 40}[n]
        for fp in files:
            inst = parse_alb(fp)
            t0 = time.perf_counter(); dp = solve_dp_fixed_m(inst, m); dp_ms = (time.perf_counter()-t0)*1000
            t0 = time.perf_counter(); gr = solve_greedy_fixed_m(inst, m); gr_ms = (time.perf_counter()-t0)*1000
            # metaheuristics on a fixed modest budget; skip GA/ACO at n=1000 except light budget
            rng = random.Random(SEED0)
            if n <= 100:
                t0 = time.perf_counter(); ga = solve_ga_fixed_m(inst, m, rng, pop_size=20, generations=30); ga_ms = (time.perf_counter()-t0)*1000
                rng = random.Random(SEED0+7)
                t0 = time.perf_counter(); aco = solve_aco_fixed_m(inst, m, rng, n_ants=15, iterations=20); aco_ms = (time.perf_counter()-t0)*1000
            else:
                t0 = time.perf_counter(); ga = solve_ga_fixed_m(inst, m, rng, pop_size=10, generations=10); ga_ms = (time.perf_counter()-t0)*1000
                rng = random.Random(SEED0+7)
                t0 = time.perf_counter(); aco = solve_aco_fixed_m(inst, m, rng, n_ants=8, iterations=8); aco_ms = (time.perf_counter()-t0)*1000
            rows.append({"instance": inst.name, "n": inst.n, "m": m,
                         "DP_ms": round(dp_ms, 2), "GR_ms": round(gr_ms, 2),
                         "GA_ms": round(ga_ms, 2), "ACO_ms": round(aco_ms, 2),
                         "DP_E": round(dp.efficiency, 4)})
            print(f"{inst.name:22s} n={inst.n:4d} DP={dp_ms:8.2f}ms GR={gr_ms:7.2f}ms GA={ga_ms:9.2f}ms ACO={aco_ms:9.2f}ms")
    with open(os.path.join(OUT, "scalability.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


if __name__ == "__main__":
    print("=== EXPERIMENT 1: QUALITY / REPRODUCIBILITY (Scholl) ===")
    q = run_quality()
    print(f"\n[done] {len(q)} (instance,m) configurations\n")
    print("=== EXPERIMENT 2: SCALABILITY (Otto) ===")
    s = run_scalability()
    print(f"\n[done] {len(s)} scalability measurements")
