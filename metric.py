import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# paths (edit if needed)
# =========================
PPC_PATHS = [
    Path("results/ppc_residuals_by_survey.parquet"),
    Path("/mnt/data/ppc_residuals_by_survey.parquet"),
]
COMP_PATHS = [
    Path("results/hermes_model_comparison.csv"),
    Path("/mnt/data/hermes_model_comparison.csv"),
    Path("results/hermes_met_bivariate.csv"),
    Path("/mnt/data/hermes_met_bivariate.csv"),
]

ppc_path = next((p for p in PPC_PATHS if p.exists()), None)
if ppc_path is None:
    raise FileNotFoundError(f"Could not find PPC parquet. Tried: {PPC_PATHS}")

comp_path = next((p for p in COMP_PATHS if p.exists()), None)
if comp_path is None:
    raise FileNotFoundError(f"Could not find leverage CSV. Tried: {COMP_PATHS}")

outdir = Path("results/plots_paper")
outdir.mkdir(parents=True, exist_ok=True)

# =========================
# helpers
# =========================
def finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]

def pick_leverage_column(df):
    # prefer L_logM, else any column starting with L_
    if "L_logM" in df.columns:
        return "L_logM"
    Lcols = [c for c in df.columns if c.startswith("L_")]
    if Lcols:
        # deterministic: choose shortest name (often L_logM) then alphabetical
        Lcols = sorted(Lcols, key=lambda s: (len(s), s))
        return Lcols[0]
    return None

# =========================
# load
# =========================
ppc = pd.read_parquet(ppc_path)
comp = pd.read_csv(comp_path)

if "survey_id" not in ppc.columns:
    raise ValueError(f"PPC file missing 'survey_id'. Columns: {list(ppc.columns)}")
if "survey_id" not in comp.columns:
    raise ValueError(f"Comparison CSV missing 'survey_id'. Columns: {list(comp.columns)}")

# keep only needed columns from comp: survey_id + leverage columns (and N/class if present)
lev_col = pick_leverage_column(comp)
if lev_col is None:
    raise ValueError(
        "Could not find any leverage column in comparison CSV. "
        "Expected 'L_logM' or something starting with 'L_'. "
        f"Columns: {list(comp.columns)}"
    )

keep = ["survey_id", lev_col]
for extra in ["N", "class_label"]:
    if extra in comp.columns:
        keep.append(extra)
comp_small = comp[keep].copy()

# merge leverage into PPC table (per-point)
ppc2 = ppc.merge(comp_small, on="survey_id", how="left")

if ppc2[lev_col].isna().all():
    raise ValueError(
        f"After merge, leverage column '{lev_col}' is all NaN. "
        "This usually means survey_id values don’t match between files."
    )

# =========================
# compute Δz² per point
# =========================
need = {"z_1d", "z_3d"}
missing = need - set(ppc2.columns)
if missing:
    raise ValueError(f"PPC parquet missing columns {missing}. Columns: {list(ppc2.columns)}")

m = np.isfinite(ppc2["z_1d"]) & np.isfinite(ppc2["z_3d"]) & np.isfinite(ppc2[lev_col])
ppc2 = ppc2.loc[m].copy()

ppc2["dz2"] = ppc2["z_1d"].to_numpy(float)**2 - ppc2["z_3d"].to_numpy(float)**2

dz2 = ppc2["dz2"].to_numpy(float)
L   = ppc2[lev_col].to_numpy(float)

print("Loaded PPC:", ppc_path, "rows:", len(ppc))
print("Loaded leverage table:", comp_path, "rows:", len(comp))
print("Merged rows used (finite):", len(ppc2))
print("Using leverage column:", lev_col)
print("Fraction improved by 3D (Δz²>0):", float(np.mean(dz2 > 0)))

# =========================
# style
# =========================
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

def clean(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=4, width=1)

# =========================
# PLOT 1: Δz² histogram
# =========================
fig, ax = plt.subplots(figsize=(6.6, 4.0))
clean(ax)

ax.hist(dz2, bins=70, alpha=0.85)
ax.axvline(0, color="k", lw=1)

ax.set_xlabel(r"$\Delta z^2 = z_{\mathrm{1D}}^2 - z_{\mathrm{3D}}^2$")
ax.set_ylabel("Count")
ax.set_title("Per-point improvement in squared PPC residual")

ax.text(
    0.02, 0.95,
    f"Fraction improved by 3D: {np.mean(dz2>0):.2f}",
    transform=ax.transAxes,
    va="top"
)

fig.tight_layout()
fig.savefig(outdir / "delta_z2_hist.pdf", bbox_inches="tight")
plt.close(fig)

# =========================
# PLOT 2: Δz² vs leverage (scatter + binned median + 68%)
# =========================
fig, ax = plt.subplots(figsize=(7.2, 4.6))
clean(ax)

ax.scatter(L, dz2, s=10, alpha=0.20)

# bin by leverage quantiles
qs = np.quantile(L, np.linspace(0, 1, 9))
qs = np.unique(qs)
idx = np.digitize(L, qs[1:-1], right=True)

Lc, med, lo, hi = [], [], [], []
for i in range(len(qs) - 1):
    sel = idx == i
    if sel.sum() < 50:
        continue
    Li = L[sel]
    zi = dz2[sel]
    Lc.append(float(np.mean(Li)))
    med.append(float(np.median(zi)))
    lo.append(float(np.quantile(zi, 0.16)))
    hi.append(float(np.quantile(zi, 0.84)))

Lc = np.array(Lc); med=np.array(med); lo=np.array(lo); hi=np.array(hi)

ax.plot(Lc, med, lw=2, color="k", label="binned median")
ax.fill_between(Lc, lo, hi, alpha=0.25, color="k", label="binned 68%")

ax.axhline(0, color="k", lw=1, ls="--")
ax.set_xlabel(f"Leverage ({lev_col})")
ax.set_ylabel(r"$\Delta z^2$")
ax.set_title("Does 3D become useful at high leverage?")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(outdir / "delta_z2_vs_leverage.pdf", bbox_inches="tight")
plt.close(fig)

# =========================
# PLOT 3: Δz² histograms split by leverage quartile
# =========================
ppc2["L_bin"] = pd.qcut(ppc2[lev_col], q=4, labels=["low", "mid-low", "mid-high", "high"])

fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
axes = axes.flat

for ax, (lab, d) in zip(axes, ppc2.groupby("L_bin", observed=True)):
    clean(ax)
    vals = d["dz2"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    ax.hist(vals, bins=45, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_title(f"Leverage: {lab} (n={len(vals)})")

axes[2].set_xlabel(r"$\Delta z^2$")
axes[3].set_xlabel(r"$\Delta z^2$")
axes[0].set_ylabel("Count")
axes[2].set_ylabel("Count")
fig.suptitle("Per-point Δz² split by leverage quartile", y=0.98)

fig.tight_layout()
fig.savefig(outdir / "delta_z2_hist_by_leverage.pdf", bbox_inches="tight")
plt.close(fig)

print("Wrote:", outdir / "delta_z2_hist.pdf")
print("Wrote:", outdir / "delta_z2_vs_leverage.pdf")
print("Wrote:", outdir / "delta_z2_hist_by_leverage.pdf")
