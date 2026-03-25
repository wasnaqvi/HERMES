"""
HERMES@AOC.py — Sci-comm style plots for the Ariel Observation Coordination talk.

Produces:
  1. Ariel MCS introductory scatter (mass vs radius, coloured by stellar [Fe/H])
  2. Injected Welbanks 2D trend (log X_{H2O} vs log M_p) with intrinsic scatter
  3. Animated GIF: 2D Welbanks plot smoothly rotates into 3D as stellar
     metallicity is introduced as the third HERMES dimension
  4. ESM vs TSM scatter (coloured by Rp, annotated key planets)
  5. Sky Aitoff — colour = TSM, size = ESM
  6. Sky Aitoff — colour = ESM, size = TSM
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import imageio.v2 as imageio
import tempfile, os
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
ARIEL_CSV = HERE / "Ariel_MCS_Known_2025-08-18.csv"
SYNTH_CSV = HERE / "hermes_synthetic_data_0.2.0.csv"
OUT_DIR   = HERE / "figures_aoc"
OUT_DIR.mkdir(exist_ok=True)

# ── injected trend parameters (Welbanks 2019) ────────────────────────────
BETA_P_TRUE   = -1.09      # mass slope
INTERCEPT     = -0.95      # intercept
BETA_S_TRUE   =  1.0       # stellar metallicity coefficient
SIGMA_SCATTER =  0.53      # injected intrinsic scatter (dex)

# ── colour palette (sci-comm friendly) ───────────────────────────────────
C_TREND   = "#1b4f72"   # deep teal-blue for the Welbanks line
C_SCATTER = "#85c1e9"   # lighter blue for scatter band
C_DATA    = "#e67e22"   # warm orange for observed data
C_TRUTH   = "#2e86c1"   # blue for synthetic truth points
C_STELLAR = "#8e44ad"   # purple accent for stellar metallicity
C_SURFACE = "#aed6f1"   # light blue for 3D surface

# ── global rcParams ──────────────────────────────────────────────────────
def _set_style():
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "serif",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.2,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "xtick.major.size":  5,
        "ytick.major.size":  5,
        "text.usetex":       False,
        "mathtext.fontset":  "cm",
    })


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 1 — Ariel MCS: Planet Mass vs Radius (coloured by stellar [Fe/H])
# ═══════════════════════════════════════════════════════════════════════════
def plot_ariel_mcs_intro():
    """Mass-radius diagram of the Ariel MCS, coloured by stellar [Fe/H]."""
    _set_style()
    df = pd.read_csv(ARIEL_CSV)

    mass   = pd.to_numeric(df["Planet Mass [Mjup]"], errors="coerce")
    radius = pd.to_numeric(df["Planet Radius [Rjup]"], errors="coerce")
    feh    = pd.to_numeric(df["Star Metallicity"],     errors="coerce")

    good = mass.notna() & radius.notna() & feh.notna() & (mass > 0) & (radius > 0)
    mass, radius, feh = mass[good].values, radius[good].values, feh[good].values

    log_mass = np.log10(mass)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    vmin, vmax = np.percentile(feh, [5, 95])
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.coolwarm

    sc = ax.scatter(
        log_mass, radius,
        c=feh, cmap=cmap, norm=norm,
        s=30, alpha=0.80, edgecolors="k", linewidths=0.3, zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(r"Stellar [Fe/H]", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    ax.set_xlabel(r"$\log_{10}\,(M_{\rm p}\,/\,M_{\rm Jup})$",fontsize=14)
    ax.set_ylabel(r"$R_{\rm p}\;[R_{\rm Jup}]$", fontsize=14)
    # ax.set_title("Ariel Mission Candidate Sample", fontweight="bold", pad=12)

    ax.text(
        0.98, 0.97,
        f"{len(mass)} planets",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=12, fontstyle="italic", color="0.35",
    )

    ax.minorticks_on()
    fig.tight_layout()
    out = OUT_DIR / "ariel_mcs_mass_radius.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[1] Saved  {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 2 — Injected Welbanks trend (2D) with intrinsic scatter
# ═══════════════════════════════════════════════════════════════════════════
def plot_welbanks_2d():
    """2D mass-metallicity trend: Welbanks line + scatter band + data."""
    _set_style()
    df = pd.read_csv(SYNTH_CSV)

    logM     = df["logM"].to_numpy(float)
    logXH2O  = df["log(X_H2O)"].to_numpy(float)
    err_lo   = df["uncertainty_lower"].to_numpy(float)
    err_hi   = df["uncertainty_upper"].to_numpy(float)
    star_met = df["Star Metallicity"].to_numpy(float)

    # truth = Welbanks line for each planet (includes stellar met)
    truth = BETA_P_TRUE * logM + BETA_S_TRUE * star_met + INTERCEPT

    # sorted mass axis for smooth line
    m_sort = np.linspace(logM.min() - 0.15, logM.max() + 0.15, 300)
    # pure Welbanks line (no stellar met — the 1D projection)
    y_line = BETA_P_TRUE * m_sort + INTERCEPT

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # scatter band (1-sigma intrinsic scatter)
    ax.fill_between(
        m_sort, y_line - SIGMA_SCATTER, y_line + SIGMA_SCATTER,
        color=C_SCATTER, alpha=0.30, label=rf"$\pm\,\sigma_{{\rm int}}={SIGMA_SCATTER}$ dex",
        zorder=1,
    )

    # observed data
    ax.errorbar(
        logM, logXH2O,
        yerr=[np.abs(err_lo), np.abs(err_hi)],
        fmt="o", ms=4, color=C_DATA, alpha=0.55,
        elinewidth=0.6, capsize=0, zorder=2,
        label="Observed (truth + noise + scatter)",
    )

    # truth points
    ax.scatter(
        logM, truth,
        s=22, facecolors="none", edgecolors=C_TRUTH,
        linewidths=0.9, zorder=4, label='Synthetic "Truth"',
    )

    # Welbanks trend line
    ax.plot(
        m_sort, y_line,
        color=C_TREND, lw=2.8, zorder=5,
        label=(
            r"Welbanks fit: $\log\,X_{\rm H_2O}\!/H"
            r"= -1.09\;\log_{10}\,(M_{\rm p}/M_{\rm Jup})\;-\;0.95$"
        ),
    )

    ax.set_xlabel(r"$\log_{10}\,(M_{\rm p}\,/\,M_{\rm Jup})$",fontsize=14)
    ax.set_ylabel(r"$\log\,X_{\rm H_2O}/H$",fontsize=14)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.85)
    ax.minorticks_on()

    fig.tight_layout()
    out = OUT_DIR / "welbanks_2d_trend.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[2] Saved  {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 3 — Animated GIF: 2D -> 3D transition  (adding stellar metallicity)
# ═══════════════════════════════════════════════════════════════════════════
def make_2d_to_3d_gif(n_frames: int = 120, fps: int = 24):
    """
    Create a smooth animated GIF that begins as a flat 2D
    mass-metallicity plot, then rotates to reveal the stellar-
    metallicity axis, forming the full 3D HERMES MetModel view.
    """
    _set_style()
    plt.rcParams["axes.spines.top"]   = True
    plt.rcParams["axes.spines.right"] = True

    df = pd.read_csv(SYNTH_CSV)
    logM     = df["logM"].to_numpy(float)
    logXH2O  = df["log(X_H2O)"].to_numpy(float)
    star_met = df["Star Metallicity"].to_numpy(float)

    # truth surface grid
    m_grid = np.linspace(logM.min() - 0.1, logM.max() + 0.1, 40)
    s_grid = np.linspace(star_met.min() - 0.05, star_met.max() + 0.05, 40)
    M, S   = np.meshgrid(m_grid, s_grid)
    Y_surf = BETA_P_TRUE * M + BETA_S_TRUE * S + INTERCEPT

    # truth per planet
    truth = BETA_P_TRUE * logM + BETA_S_TRUE * star_met + INTERCEPT

    # axis limits (fixed for all frames)
    xlim = (logM.min() - 0.25, logM.max() + 0.25)
    ylim = (star_met.min() - 0.15, star_met.max() + 0.15)
    zlim = (logXH2O.min() - 0.4, logXH2O.max() + 0.4)

    # ── animation schedule ──
    # Phase 1 (0-19%):   hold flat 2D view with scatter band
    # Phase 2 (20-82%):  rotate to 3D, fade in stellar axis + surface
    # Phase 3 (83-100%): hold final 3D view
    phase1_end = int(n_frames * 0.20)
    phase2_end = int(n_frames * 0.83)

    # view angles: start nearly edge-on to the stellar-met axis
    az_start, az_end   = -90, -50
    el_start, el_end   =  15,  28

    tmp_dir = tempfile.mkdtemp(prefix="hermes_gif_")
    frame_paths = []

    for i in range(n_frames):
        if i < phase1_end:
            t = 0.0
        elif i < phase2_end:
            t = (i - phase1_end) / (phase2_end - phase1_end)
        else:
            t = 1.0

        # smooth ease-in-out via cosine
        t_smooth = 0.5 * (1 - np.cos(np.pi * t))

        azim = az_start + (az_end - az_start) * t_smooth
        elev = el_start + (el_end - el_start) * t_smooth
        surf_alpha = 0.25 * t_smooth

        fig = plt.figure(figsize=(10, 7))
        ax  = fig.add_subplot(111, projection="3d")

        # collapse stellar-met spread to zero at t=0 (flat 2D look)
        s_mean = star_met.mean()
        s_display = s_mean + (star_met - s_mean) * t_smooth

        # ── scatter band (fades out as surface fades in) ──
        m_line = np.linspace(xlim[0] + 0.05, xlim[1] - 0.05, 200)
        y_line = BETA_P_TRUE * m_line + INTERCEPT
        s_line = np.full_like(m_line, s_mean)
        band_alpha = 0.30 * (1 - t_smooth)
        if band_alpha > 0.005:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            verts_upper = list(zip(m_line, s_line, y_line + SIGMA_SCATTER))
            verts_lower = list(zip(m_line[::-1], s_line[::-1],
                                   (y_line + SIGMA_SCATTER * (-1))[::-1]))
            poly = Poly3DCollection([verts_upper + verts_lower],
                                    alpha=band_alpha, facecolor=C_SCATTER,
                                    edgecolor="none", zorder=1)
            ax.add_collection3d(poly)

        # ── observed data (orange) ──
        ax.scatter(
            logM, s_display, logXH2O,
            c=C_DATA, s=14, alpha=0.50, zorder=2, depthshade=True,
        )

        # ── truth points (blue hollow) ──
        ax.scatter(
            logM, s_display, truth,
            facecolors="none", edgecolors=C_TRUTH,
            s=20, linewidths=0.8, alpha=0.65, zorder=4, depthshade=True,
        )

        # ── Welbanks trend line (always visible) ──
        ax.plot(m_line, s_line, y_line, color=C_TREND, lw=2.8, zorder=6)

        # ── 3D trend surface (fades in) ──
        if surf_alpha > 0.008:
            ax.plot_surface(
                M, S, Y_surf,
                alpha=surf_alpha, color=C_SURFACE,
                edgecolor=(0.6, 0.6, 0.6, surf_alpha * 0.3),
                linewidth=0.3, zorder=1,
            )

        # ── axes ──
        ax.set_xlabel(r"$\log_{10}(M_{\rm p}/M_{\rm Jup})$",
                       fontsize=13, labelpad=12)
        ax.set_zlabel(r"$\log\,X_{\rm H_2O}/H$",
                       fontsize=13, labelpad=12)

        # stellar-met label + ticks fade in
        if t_smooth > 0.08:
            lbl_alpha = min(1.0, t_smooth * 1.8)
            ax.set_ylabel(r"[Fe/H]$_\star$",
                          fontsize=13, labelpad=12, alpha=lbl_alpha)
            ax.yaxis.set_tick_params(labelsize=9)
            for lbl in ax.get_yticklabels():
                lbl.set_alpha(lbl_alpha)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)

        # title
        if t_smooth < 0.03:
            title = r"Mass$-$Metallicity (Welbanks 2019)"
        elif t_smooth > 0.97:
            title = r"HERMES: Mass + [Fe/H]$_\star$ $\rightarrow$ Planetary Met"
        else:
            pct = int(t_smooth * 100)
            title = f"Introducing Stellar Metallicity...  ({pct}%)"
        ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="z", labelsize=10)

        # subtle pane styling
        for pane, fc in [(ax.xaxis.pane, 0.97),
                         (ax.yaxis.pane, 0.94),
                         (ax.zaxis.pane, 0.97)]:
            pane.set_facecolor((fc, fc, fc, 0.4))
            pane.set_edgecolor((0.82, 0.82, 0.82, 1.0))

        fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
        fpath = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        fig.savefig(fpath, dpi=140, facecolor="white", edgecolor="none")
        frame_paths.append(fpath)
        plt.close(fig)

    # ── assemble GIF via PIL for correct timing ──
    from PIL import Image as PILImage

    pil_frames = [PILImage.open(fp).convert("RGBA") for fp in frame_paths]

    # hold the last frame for 1.5 s
    n_hold = int(1.5 * fps)
    pil_frames.extend([pil_frames[-1].copy()] * n_hold)

    frame_duration = int(1000 / fps)            # ms per frame
    last_duration  = int(1500 / n_hold) if n_hold else frame_duration
    durations = [frame_duration] * n_frames + [frame_duration] * n_hold

    out = OUT_DIR / "hermes_2d_to_3d.gif"
    pil_frames[0].save(
        str(out),
        save_all=True,
        append_images=pil_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    # clean up temp pngs
    for fp in frame_paths:
        os.remove(fp)
    os.rmdir(tmp_dir)

    print(f"[3] Saved  {out}  ({len(pil_frames)} frames @ {fps} fps)")


# ═══════════════════════════════════════════════════════════════════════════
#  PHYSICS HELPERS — TSM & ESM  (Kempton et al. 2018)
# ═══════════════════════════════════════════════════════════════════════════
_H  = 6.62607015e-34       # Planck constant  [J s]
_C  = 299792458.0           # speed of light   [m/s]
_KB = 1.380649e-23          # Boltzmann const  [J/K]
_R_SUN_M  = 6.957e8         # solar radius     [m]
_AU_M     = 1.495978707e11  # AU               [m]
_R_SUN_OVER_R_JUP = 9.73116


def _planck_ratio(T_num, T_den, lam_um=7.5):
    """B_lambda(T_num) / B_lambda(T_den) at lam_um microns."""
    lam_m = lam_um * 1e-6
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        x_num = (_H * _C) / (lam_m * _KB * np.asarray(T_num, float))
        x_den = (_H * _C) / (lam_m * _KB * np.asarray(T_den, float))
    return np.expm1(x_den) / np.expm1(x_num)


def _compute_teq_geom(df: pd.DataFrame) -> np.ndarray:
    """Teq = T_star * sqrt(R_star / (2 a))  [A=0, full redistribution]."""
    tstar = pd.to_numeric(df["Star Temperature [K]"], errors="coerce").values
    rs    = pd.to_numeric(df["Star Radius [Rs]"],     errors="coerce").values
    a_au  = pd.to_numeric(df["Planet Semi-major Axis [au]"], errors="coerce").values
    teq = tstar * np.sqrt(rs * _R_SUN_M / (2.0 * a_au * _AU_M))
    teq[~np.isfinite(teq) | (teq <= 0)] = np.nan
    return teq


def _compute_tsm(df: pd.DataFrame, teq: np.ndarray) -> np.ndarray:
    """TSM = scale * Rp^3 * Teq / (Mp * Rs^2) * 10^(-mJ/5)."""
    rp = pd.to_numeric(df["Planet Radius [Re]"],  errors="coerce").values
    mp = pd.to_numeric(df["Planet Mass [Me]"],     errors="coerce").values
    rs = pd.to_numeric(df["Star Radius [Rs]"],     errors="coerce").values
    mj = pd.to_numeric(df["Star J Mag"],           errors="coerce").values
    scale = np.full_like(rp, np.nan)
    scale[(rp > 0) & (rp < 1.5)]   = 0.190
    scale[(rp >= 1.5) & (rp < 2.75)] = 1.26
    scale[(rp >= 2.75) & (rp < 4.0)] = 1.28
    scale[(rp >= 4.0) & (rp < 10.0)] = 1.15
    tsm = scale * (rp**3 * teq) / (mp * rs**2) * 10**(-mj / 5.0)
    tsm[~np.isfinite(tsm) | (tsm <= 0)] = np.nan
    return tsm


def _compute_esm(df: pd.DataFrame, teq_geom: np.ndarray) -> np.ndarray:
    """ESM = 4.29e6 * B_7.5(Tday)/B_7.5(T*) * (Rp/Rs)^2 * 10^(-mK/5)."""
    rp_rj  = pd.to_numeric(df["Planet Radius [Rjup]"],  errors="coerce").values
    rs_rs  = pd.to_numeric(df["Star Radius [Rs]"],       errors="coerce").values
    tstar  = pd.to_numeric(df["Star Temperature [K]"],   errors="coerce").values
    mk     = pd.to_numeric(df["Star Ks Mag"],            errors="coerce").values
    t_ecl  = pd.to_numeric(df["Planet Eclipse Temperature [K]"], errors="coerce").values
    tday = np.where(np.isfinite(t_ecl), t_ecl, teq_geom)
    tday = np.where(np.isfinite(tday), tday,
                    pd.to_numeric(df["Planet Temperature [K]"], errors="coerce").values)
    rs_rj = rs_rs * _R_SUN_OVER_R_JUP
    bratio = _planck_ratio(tday, tstar, 7.5)
    esm = 4.29e6 * bratio * (rp_rj / rs_rj)**2 * 10**(-mk / 5.0)
    esm[~np.isfinite(esm) | (esm <= 0)] = np.nan
    return esm


def _load_ariel_with_metrics():
    """Load 2025 Ariel MCS and attach TSM, ESM columns."""
    df = pd.read_csv(ARIEL_CSV)
    teq = _compute_teq_geom(df)
    df["Teq_geom"]  = teq
    df["TSM"] = _compute_tsm(df, teq)
    df["ESM"] = _compute_esm(df, teq)
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 4 — ESM vs TSM scatter  (coloured by planet radius)
# ═══════════════════════════════════════════════════════════════════════════
# planets to label on the scatter
_LABEL_PLANETS = [
    "55Cnce", "GJ1214b", "WASP-107b", "WASP-39b",
    "GJ436b", "HD149026b", "HAT-P-11b", "K2-18b",
    "TOI-270d", "HD97658b", "GJ3470b",
]

def plot_esm_vs_tsm():
    """Log-log ESM vs TSM scatter, coloured by Rp, with Kempton thresholds."""
    _set_style()
    df = _load_ariel_with_metrics()

    tsm = df["TSM"].values
    esm = df["ESM"].values
    rp  = pd.to_numeric(df["Planet Radius [Re]"], errors="coerce").values
    names = df["Planet Name"].astype(str).values

    ok = np.isfinite(tsm) & (tsm > 0) & np.isfinite(esm) & (esm > 0)
    tsm, esm, rp, names = tsm[ok], esm[ok], rp[ok], names[ok]

    x = np.log10(tsm)
    y = np.log10(esm)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # colorbar by planet radius
    vmin, vmax = np.nanpercentile(rp, [2, 98])
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sc = ax.scatter(
        x, y, c=rp, cmap="magma_r", norm=norm,
        s=38, alpha=0.82, edgecolors="k", linewidths=0.35, zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.88)
    cbar.set_label(r"$R_{\rm p}\;[R_\oplus]$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)

    # Kempton recommended thresholds
    ax.axvline(np.log10(10),  color="0.55", ls="--", lw=1.0, zorder=1)
    ax.axhline(np.log10(7.5), color="0.55", ls="--", lw=1.0, zorder=1)
    ax.text(np.log10(10) + 0.06, ax.get_ylim()[0] + 0.08 if ax.get_ylim()[0] < -2 else y.min() - 0.15,
            "TSM = 10", fontsize=9, color="0.45", rotation=90, va="bottom")
    ax.text(x.min() + 0.04, np.log10(7.5) + 0.06,
            "ESM = 7.5", fontsize=9, color="0.45", va="bottom")

    # annotate key planets
    for lbl in _LABEL_PLANETS:
        idx = np.where(names == lbl)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        ax.annotate(
            lbl.replace("b", " b").replace("d", " d").replace("e", " e"),
            (x[i], y[i]),
            fontsize=8, fontweight="bold", color="0.15",
            xytext=(6, 6), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=5,
        )

    ax.set_xlabel(r"$\log_{10}\,(\mathrm{TSM})$")
    ax.set_ylabel(r"$\log_{10}\,(\mathrm{ESM})$")
    ax.set_title(f"Ariel MCS 2025: ESM vs TSM  (N = {ok.sum()})",
                  fontweight="bold", pad=12)
    ax.set_xlim(0,3)
    ax.minorticks_on()

    fig.tight_layout()
    out = OUT_DIR / "ariel_ESM_vs_TSM.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[4] Saved  {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  PLOT 5 & 6 — Sky Aitoff maps  (TSM/ESM colour + size)
# ═══════════════════════════════════════════════════════════════════════════
def _sky_aitoff_pretty(
    df: pd.DataFrame,
    color_col: str,
    size_col: str,
    out_name: str,
    label_n: int = 15,
    title: str = "",
):
    """Aitoff sky map with log-scaled colour + size encoding."""
    _set_style()
    plt.rcParams["axes.spines.top"]   = True
    plt.rcParams["axes.spines.right"] = True

    ra  = pd.to_numeric(df["Star RA"],  errors="coerce").values
    dec = pd.to_numeric(df["Star Dec"], errors="coerce").values
    names = df["Planet Name"].astype(str).values
    cv  = pd.to_numeric(df[color_col], errors="coerce").values
    sv  = pd.to_numeric(df[size_col],  errors="coerce").values

    ok = (np.isfinite(ra) & np.isfinite(dec)
          & np.isfinite(cv) & (cv > 0)
          & np.isfinite(sv) & (sv > 0))
    ra, dec, names = ra[ok], dec[ok], names[ok]
    cv, sv = cv[ok], sv[ok]

    lon = np.deg2rad(((ra + 180) % 360) - 180)
    lat = np.deg2rad(dec)

    c_log = np.log10(cv)
    s_log = np.log10(sv)

    # robust size mapping
    s_lo, s_hi = np.nanpercentile(s_log, [2, 98])
    s_norm = np.clip((s_log - s_lo) / (s_hi - s_lo + 1e-12), 0, 1)
    sizes = 14 + 100 * s_norm

    c_lo, c_hi = np.nanpercentile(c_log, [1, 99])

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection="aitoff")
    ax.set_facecolor((0.98, 0.98, 0.99))
    ax.grid(True, linewidth=0.5, alpha=0.30)

    sc = ax.scatter(
        lon, lat, s=sizes, c=c_log,
        cmap="viridis", vmin=c_lo, vmax=c_hi,
        alpha=0.82, edgecolors="k", linewidths=0.2, zorder=3,
    )

    # RA ticks in hours
    xticks = np.array([-150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150])
    ax.set_xticks(np.deg2rad(xticks))
    ax.set_xticklabels([f"{int(d/15)}h" if d else "0h" for d in xticks],
                        fontsize=10)

    # label the top-scoring planets (log-product of both metrics)
    score = np.log10(cv) + np.log10(sv)
    top_idx = np.argsort(score)[::-1][:min(label_n, len(score))]
    for i in top_idx:
        ax.annotate(
            names[i],
            (lon[i], lat[i]),
            fontsize=7.5, fontweight="bold",
            xytext=(4, 4), textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.70),
        )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal",
                         pad=0.08, fraction=0.045, aspect=45)
    cbar.set_label(rf"$\log_{{10}}$({color_col})", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.text(0.01, -0.09,
            rf"marker size $\propto$ $\log_{{10}}$({size_col})",
            transform=ax.transAxes, fontsize=10, color="0.35")

    n_shown = ok.sum()
    ax.text(0.99, -0.09, f"N = {n_shown}",
            transform=ax.transAxes, fontsize=10, color="0.35",
            ha="right")

    fig.tight_layout()
    out = OUT_DIR / out_name
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_sky_tsm_esm():
    """Sky map: colour = TSM, size = ESM."""
    df = _load_ariel_with_metrics()
    out = _sky_aitoff_pretty(
        df, "TSM", "ESM",
        out_name="ariel_sky_TSM_color_ESM_size.pdf",
        label_n=15,
        title="Ariel MCS 2025: Sky Map (colour = TSM, size = ESM)",
    )
    print(f"[5] Saved  {out}")


def plot_sky_esm_tsm():
    """Sky map: colour = ESM, size = TSM."""
    df = _load_ariel_with_metrics()
    out = _sky_aitoff_pretty(
        df, "ESM", "TSM",
        out_name="ariel_sky_ESM_color_TSM_size.pdf",
        label_n=15,
        title="Ariel MCS 2025: Sky Map (colour = ESM, size = TSM)",
    )
    print(f"[6] Saved  {out}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  HERMES@AOC — generating sci-comm plots")
    print("=" * 60)

    # confirm the sky Aitoff plot exists
    aitoff = HERE / "ariel_targets_sky_aitoff_pretty.pdf"
    if aitoff.exists():
        print(f"[OK] Sky Aitoff plot exists: {aitoff}")
    else:
        print(f"[!!] Sky Aitoff plot NOT found: {aitoff}")

    plot_ariel_mcs_intro()
    plot_welbanks_2d()
    make_2d_to_3d_gif()
    plot_esm_vs_tsm()
    plot_sky_tsm_esm()
    plot_sky_esm_tsm()

    print("=" * 60)
    print("  Done.  All outputs in:", OUT_DIR)
    print("=" * 60)
