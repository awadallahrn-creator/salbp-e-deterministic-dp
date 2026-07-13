"""make_figures.py — build all paper figures from the REAL result files."""
import json, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/claude/salbp/results"
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

q = json.load(open(f"{OUT}/quality.json"))
s = json.load(open(f"{OUT}/scalability.json"))

# ---- Fig 1: mean efficiency with variability (box-like bar + error) ----
methods = [("Greedy", "GR_E", None), ("DP", "DP_E", None),
           ("GA", "GA_E_mean", "GA_E_std"), ("ACO", "ACO_E_mean", "ACO_E_std")]
means, errs, labels = [], [], []
for name, mk, sk in methods:
    vals = [r[mk] for r in q]
    means.append(statistics.mean(vals))
    # representative variability: mean within-instance std for stochastic, 0 else
    errs.append(statistics.mean([r[sk] for r in q]) if sk else 0.0)
    labels.append(name)
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["#b0b0b0", "#1f6feb", "#2da44e", "#bf8700"]
bars = ax.bar(labels, means, yerr=errs, capsize=5, color=colors, edgecolor="black", linewidth=0.6)
for b, mn in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, mn+0.005, f"{mn:.3f}", ha="center", fontsize=10)
ax.set_ylabel("Mean line efficiency $E$")
ax.set_ylim(0.5, 1.02)
ax.set_title("Mean efficiency across 70 (instance, $m$) configurations")
fig.savefig(f"{OUT}/fig_efficiency.png")
plt.close(fig)

# ---- Fig 2: reproducibility — std of E per method (DP/Greedy = 0) ----
fig, ax = plt.subplots(figsize=(6, 4))
ga_std = [r["GA_E_std"] for r in q]
aco_std = [r["ACO_E_std"] for r in q]
ax.boxplot([[0.0]*len(q), [0.0]*len(q), ga_std, aco_std],
           tick_labels=["Greedy", "DP", "GA", "ACO"], showmeans=True)
ax.set_ylabel("Within-configuration std. of $E$ (10 runs)")
ax.set_title("Solution variability (reproducibility)")
fig.savefig(f"{OUT}/fig_reproducibility.png")
plt.close(fig)

# ---- Fig 3: runtime scalability (log-y) from Otto ----
sizes = sorted(set(r["n"] for r in s))
def avg_ms(n, key): 
    vals=[r[key] for r in s if r["n"]==n]; return statistics.mean(vals)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for key, name, mk in [("GR_ms","Greedy","o"),("DP_ms","DP","s"),
                      ("GA_ms","GA","^"),("ACO_ms","ACO","d")]:
    ax.plot(sizes, [avg_ms(n,key) for n in sizes], marker=mk, label=name)
ax.set_xlabel("Number of tasks $n$")
ax.set_ylabel("Runtime (ms, log scale)")
ax.set_yscale("log")
ax.set_title("Runtime scalability on Otto et al. (2013) instances")
ax.legend()
fig.savefig(f"{OUT}/fig_runtime.png")
plt.close(fig)

# ---- Fig 4: DP vs best-metaheuristic ratio across instances ----
ratios = sorted(r["DP_E"]/max(r["GA_E_best"], r["ACO_E_best"]) for r in q)
fig, ax = plt.subplots(figsize=(6.4, 4))
ax.plot(range(1, len(ratios)+1), ratios, marker=".", linewidth=1)
ax.axhline(1.0, color="green", ls="--", lw=1, label="parity with best metaheuristic")
ax.axhline(0.95, color="orange", ls=":", lw=1, label="95% of best")
ax.set_xlabel("Configuration (sorted)")
ax.set_ylabel("$E_{DP}\\,/\\,E_{best-meta}$")
ax.set_title("DP efficiency relative to best metaheuristic solution")
ax.legend(loc="lower right")
fig.savefig(f"{OUT}/fig_ratio.png")
plt.close(fig)

print("figures written:")
import os
for f in ["fig_efficiency.png","fig_reproducibility.png","fig_runtime.png","fig_ratio.png"]:
    print("  ", f, os.path.getsize(f"{OUT}/{f}"), "bytes")
