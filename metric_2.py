import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

ppc = pd.read_parquet("results/ppc_residuals_by_survey.parquet")

z1 = ppc["z_1d"].to_numpy(float)
z3 = ppc["z_3d"].to_numpy(float)
z1 = z1[np.isfinite(z1)]
z3 = z3[np.isfinite(z3)]

x = np.linspace(-4, 4, 400)

fig, ax = plt.subplots(figsize=(6.5, 4.0))

ax.hist(z1, bins=60, density=True, alpha=0.45, label="1D model")
ax.hist(z3, bins=60, density=True, alpha=0.45, label="3D model")

ax.plot(x, norm.pdf(x), "k--", lw=2, label=r"$\mathcal{N}(0,1)$")

ax.set_xlabel("PPC z-score")
ax.set_ylabel("Probability density")
ax.set_title("Pooled posterior predictive residuals")
ax.legend(frameon=False)

ax.axvline(0, color="k", lw=1)
ax.set_xlim(-4, 4)

plt.tight_layout()
plt.savefig("results/plots_paper/z_hist_pooled_1D_vs_3D.pdf")
plt.close()

ppc["N_bin"] = pd.cut(
    ppc["N"],
    bins=[0, 40, 80, 150, 300],
    labels=["small", "medium", "large", "very large"]
)

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)

for ax, (label, d) in zip(axes.flat, ppc.groupby("N_bin")):
    z1 = d["z_1d"].dropna().to_numpy()
    z3 = d["z_3d"].dropna().to_numpy()

    ax.hist(z1, bins=40, density=True, alpha=0.45, label="1D")
    ax.hist(z3, bins=40, density=True, alpha=0.45, label="3D")
    ax.plot(x, norm.pdf(x), "k--", lw=1)

    ax.set_title(f"N = {label}")
    ax.axvline(0, color="k", lw=0.8)

axes[0, 0].legend(frameon=False)
fig.suptitle("Posterior predictive residuals by survey size", y=0.98)

plt.tight_layout()
plt.savefig("results/plots_paper/z_hist_by_N.pdf")
plt.close()

from scipy.stats import probplot

fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

probplot(z1, dist="norm", plot=axes[0])
axes[0].set_title("1D model")

probplot(z3, dist="norm", plot=axes[1])
axes[1].set_title("3D model")


plt.tight_layout()
plt.savefig("results/plots_paper/z_qq_1D_vs_3D.pdf")
plt.close()

dz2 = z1**2 - z3**2

fig, ax = plt.subplots(figsize=(6.5, 4.0))

ax.hist(dz2, bins=60, alpha=0.7)
ax.axvline(0, color="k", lw=1)

ax.set_xlabel(r"$z_{\mathrm{1D}}^2 - z_{\mathrm{3D}}^2$")
ax.set_ylabel("Count")
ax.set_title("Per-point improvement in squared residual")

plt.tight_layout()
plt.savefig("results/plots_paper/delta_z2_hist.pdf")
plt.close()

print("Fraction improved by 3D:", np.mean(dz2 > 0))
