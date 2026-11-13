"""Family-level paired benefit with 95% bootstrap intervals across environments (R1 figure)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
FAM = ["temporal", "routing", "probability", "assignment", "optimization"]
ENVS = ["A", "B", "C", "D", "E_test"]
LAB = {"A": "A", "B": "B", "C": "C", "D": "D", "E_test": "E test"}
COL = {"shift_main": "#2a78d6", "shift_qwen": "#eb6834"}
NAME = {"shift_main": "DeepSeek-V3.2 → Qwen3.5-397B", "shift_qwen": "Qwen3.5-35B → Qwen3.5-397B"}
MARK = {"shift_main": "o", "shift_qwen": "s"}

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.labelcolor": "#333", "xtick.color": "#555", "ytick.color": "#555"})
fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.35), sharey=True)
for ax, fam in zip(axes, FAM):
    ax.axhline(15, color="#999", lw=0.8, ls="--", zorder=0)
    ax.axhline(0, color="#ccc", lw=0.6, zorder=0)
    for j, key in enumerate(["shift_main", "shift_qwen"]):
        cells = {c["env"]: c for c in R[key]["cells"] if c["family"] == fam}
        xs = [i + (-0.13 if j == 0 else 0.13) for i in range(len(ENVS))]
        ys = [100 * cells[e]["tau"] for e in ENVS]
        lo = [100 * (cells[e]["tau"] - cells[e]["lo"]) for e in ENVS]
        hi = [100 * (cells[e]["hi"] - cells[e]["tau"]) for e in ENVS]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=MARK[key], ms=3.6, color=COL[key], ecolor=COL[key],
                    elinewidth=1.2, capsize=0, mec="white", mew=0.5, label=NAME[key])
    ax.set_title(fam.capitalize(), fontsize=8.5, color="#222")
    ax.set_xticks(range(len(ENVS)))
    ax.set_xticklabels([LAB[e] for e in ENVS], fontsize=7)
    ax.grid(axis="y", color="#eee", lw=0.6)
    ax.set_ylim(-12, 60)
axes[0].set_ylabel("Paired benefit τ (pp)")
axes[0].text(-0.45, 17, "γ = 15 pp", fontsize=6.5, color="#666", ha="left")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out / "benefit_shift_ci.pdf")
fig.savefig(out / "benefit_shift_ci.png", dpi=220)
print("ok")
