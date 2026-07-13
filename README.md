# A Deterministic Dynamic-Programming Heuristic for SALBP-E

Reproducibility package for the paper:

> **A Deterministic Dynamic-Programming Heuristic for the Type-E Simple Assembly
> Line Balancing Problem: A Reproducible Efficiency–Runtime Trade-off on Standard
> Benchmarks.**

This repository contains the full implementation, the exact/brute-force
validators, and all raw result files (JSON/CSV) needed to regenerate every table
and figure in the manuscript. The deterministic methods (DP, DP-*k*, Greedy)
reproduce identical results on every run; the stochastic baselines (GA, ACO) use
fixed, documented seeds.

## Repository layout

```
.
├── core.py               # instance parsing (Scholl .IN2 and Otto .alb) and data structures
├── dp_heuristic.py       # single-order deterministic DP heuristic (min–max partition)
├── dp_multi.py           # multi-order extension DP-k (deterministic tie-break rules R1–R6)
├── baselines.py          # Greedy, Genetic Algorithm (GA), Ant Colony Optimization (ACO)
├── evaluate.py           # efficiency, cycle time, and lower-bound helpers
├── exact2.py             # exact station-oriented SALBP-2/E solver (subset DP)
├── brute.py              # brute-force oracle used to validate exact2.py on tiny instances
├── run_experiments.py    # driver: produces quality + scalability result files
├── make_figures.py       # regenerates the figures from the result files
├── requirements.txt      # pinned Python dependencies
├── quality.json          # per-configuration quality results (Scholl set)
├── quality_R30.json      # quality results with R = 30 stochastic repetitions
├── quality.csv           # tabular export of the quality results
├── dpk.json              # DP-k aggregate results
├── exact_gap.json        # certified optimality gaps at fixed m
├── ksens.json            # sensitivity of DP-k to the number of orders k (k = 1..6)
├── ksens12.json          # extended k-sensitivity (up to k = 12)
└── scalability.json      # runtime scalability on Otto instances (n up to 1000)
```

## Requirements

Python 3.12 with the pinned versions in `requirements.txt`:

```bash
pip install -r requirements.txt
```

(`numpy==2.4.4`, `matplotlib==3.10.8`, `scipy==1.17.1`.)

## Benchmark data

The experiments use two standard, publicly available benchmark families, which
are **not redistributed here**:

- **Scholl (1993)** classical set — 25 precedence graphs (`.IN2` files).
- **Otto et al. (2013)** generated set (SALBPGen) — instances with
  n ∈ {20, 50, 100, 1000} (`.alb` files).

Download them from their original sources, then set the data-directory paths at
the top of `run_experiments.py` (`SCHOLL_DIR`, `OTTO_GLOB`) and the output path
`OUT` in both `run_experiments.py` and `make_figures.py` to your local
locations.

## Reproducing the results

The pre-computed result files in this repository already back every number in
the paper. To regenerate them from scratch:

```bash
python3 run_experiments.py    # writes the quality and scalability result files
python3 make_figures.py       # regenerates the figures from those files
```

Notes:
- The deterministic methods (DP, DP-*k*, Greedy) return identical output on every
  execution; no seed is required.
- The stochastic methods (GA, ACO) are run over multiple repetitions with seeds
  set to the run index, as documented in `run_experiments.py`.
- `exact2.py` computes certified SALBP-E optima on the small instances and was
  validated against the exhaustive enumeration in `brute.py`.

## License

Released under the MIT License (see `LICENSE`).

## Citation

If you use this code, please cite the paper (see `CITATION.cff`). The exact
bibliographic details will be finalized upon publication.
