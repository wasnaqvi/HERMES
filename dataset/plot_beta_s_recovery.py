"""
plot_beta_s_recovery.py — Two-panel β_s recovery figure for the scatter-threshold experiment.

Left:  Median posterior β_s vs intrinsic scatter σ  (attenuation bias)
Right: Median |β_s / σ_{β_s}| vs σ                (significance)

Reads: hermes_scatter_threshold_results.csv
Saves: figures/scatter_threshold_beta_s_recovery.pdf
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
CSV  = HERE / "hermes_scatter_threshold_results.csv"
OUT  = HERE / "figures" / "scatter_threshold_beta_s_recovery.pdf"

# ── HERMES style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
})

# ── N-based color palette ────────────────────────────────────────────────
N_COLORS = {
    80:  "#e74c3c",   # red
    150: "#e67e22",   # orange
    250: "#2ecc71",   # green
    400: "#3498db",   # blue
    600: "#8e44ad",   # purple
}

# ── load and filter ──────────────────────────────────────────────────────
df = pd.read_csv(CSV)
df3 = df[df["model"] == "3D Model"].copy()

# Compute significance: |β_s / σ_{β_s}|
df3["beta_s_sig"] = np.abs(df3["beta_s_mean"] / df3["beta_s_sd"])

# Group by (sigma, N): take median across surveys/seeds/classes
grp = df3.groupby(["sigma", "N"]).agg(
    beta_s_med=("beta_s_mean", "median"),
    sig_med=("beta_s_sig", "median"),
).reset_index()

N_vals = sorted(grp["N"].unique())

# ── plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=False)

for N in N_vals:
    sub = grp[grp["N"] == N].sort_values("sigma")
    c = N_COLORS.get(N, "gray")

    # LEFT: median β_s vs σ
    axes[0].plot(sub["sigma"], sub["beta_s_med"],
                 "o-", color=c, ms=5, lw=1.5, label=f"$N={N}$")

    # RIGHT: median |β_s / σ_{β_s}| vs σ
    axes[1].plot(sub["sigma"], sub["sig_med"],
                 "o-", color=c, ms=5, lw=1.5, label=f"$N={N}$")

# LEFT panel
axes[0].axhline(1.0, color="black", ls="--", lw=0.8, alpha=0.6)
axes[0].set_xlabel(r"Intrinsic scatter $\sigma$ (dex)")
axes[0].set_ylabel(r"Median posterior $\beta_s$")
axes[0].set_title(r"$\beta_s$ Recovery")
axes[0].legend(fontsize=9, frameon=False)

# RIGHT panel
axes[1].axhline(2.0, color="black", ls="--", lw=0.8, alpha=0.6,
                label=r"$2\sigma$ threshold")
axes[1].set_xlabel(r"Intrinsic scatter $\sigma$ (dex)")
axes[1].set_ylabel(r"Median $|\beta_s\,/\,\sigma_{\beta_s}|$")
axes[1].set_title(r"$\beta_s$ Significance")
axes[1].legend(fontsize=9, frameon=False)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
