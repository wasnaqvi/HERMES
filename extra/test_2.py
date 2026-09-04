import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_rgb

from src.data import HermesData
from src.Survey import SurveySampler

# Per-class colours (single source of truth for fills + legend).
# S2 uses a clean gold so the S1-and-S2 overlap reads as yellow, not muddy tan.
CLASS_COLORS = {"S1": "blue", "S2": "gold", "S3": "green", "S4": "red"}
FILL_ALPHA = 0.35


def _tint(color, alpha=FILL_ALPHA):
    """Pre-blend `color` over a white background at `alpha`.

    Gives the same soft translucent tint as a single alpha layer, but as an
    OPAQUE colour. Drawing the nested classes with these tints (opaque, layered)
    keeps the airy look while avoiding the muddy blending of overlapping
    semi-transparent layers -- so each mass range shows one clean colour that
    matches its legend swatch (e.g. S2 reads as clean yellow, not tan).
    """
    r, g, b = to_rgb(color)
    return (alpha * r + (1 - alpha), alpha * g + (1 - alpha), alpha * b + (1 - alpha))


# Opaque pastel tints used for both fills and legend swatches.
CLASS_TINTS = {k: _tint(v) for k, v in CLASS_COLORS.items()}


def add_nested_S_legend(ax):
    """
    Legend encoding nesting:
      S1: all colours
      S2: gold + green + red
      S3: green + red
      S4: red only
    """
    c = CLASS_TINTS

    def swatch(label):
        # Thin sub-patches read paler than the same colour spread over a wide
        # bar, so give each block a slightly darker edge of its own hue: the
        # fill stays pixel-identical to the histogram, but the swatch reads as
        # the same colour at legend size (matters most for S2's gold).
        r, g, b = c[label]
        return Patch(facecolor=c[label],
                     edgecolor=(r * 0.72, g * 0.72, b * 0.72),
                     linewidth=0.8)

    handles = [
        tuple(swatch(k) for k in ("S1", "S2", "S3", "S4")),
        tuple(swatch(k) for k in ("S2", "S3", "S4")),
        tuple(swatch(k) for k in ("S3", "S4")),
        (swatch("S4"),),
    ]
    labels = [
        r"S1 (entire Ariel MCS)",
        r"S2 (logM $\geq$ $M_{25}$)",
        r"S3 (logM $\geq$ $M_{50}$)",
        r"S4 (logM $\geq$ $M_{75}$)",
    ]
    ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
        # S1 packs four sub-patches into one handle; the default handlelength
        # of 2.0 leaves each ~5 px wide, too narrow to read as its own colour.
        handlelength=5.0,
        handleheight=1.4,
        title="Nested mass classes",
        frameon=True,
        fontsize=15,
        title_fontsize=15,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
    )


def plot_logm_nested_counts(sampler: SurveySampler, bins: int = 30):
    df = sampler.hermes.df
    logm_all = df["logM"].to_numpy(float)
    logm_all = logm_all[np.isfinite(logm_all)]

    q25, q50, q75 = np.quantile(logm_all, [0.25, 0.5, 0.75])

    s1 = logm_all
    s2 = logm_all[logm_all >= q25]
    s3 = logm_all[logm_all >= q50]
    s4 = logm_all[logm_all >= q75]

    bin_edges = np.linspace(logm_all.min(), logm_all.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(
        logm_all,
        bins=bin_edges,
        histtype="step",
        linewidth=2,
        color="black",
        label=f"Entire Ariel MCS (n={len(logm_all)})",
        zorder=5,
    )
    # Opaque pre-blended tints, layered smallest-class-on-top: each mass range
    # shows one clean colour (nested classes share identical counts in overlap
    # bins, so no height information is lost).
    ax.hist(s1, bins=bin_edges, color=CLASS_TINTS["S1"], edgecolor="none", zorder=1)
    ax.hist(s2, bins=bin_edges, color=CLASS_TINTS["S2"], edgecolor="none", zorder=2)
    ax.hist(s3, bins=bin_edges, color=CLASS_TINTS["S3"], edgecolor="none", zorder=3)
    ax.hist(s4, bins=bin_edges, color=CLASS_TINTS["S4"], edgecolor="none", zorder=4)

    ax.set_xlabel(r"$\log\!\left(\frac{M}{M_\mathrm{J}}\right)$", fontsize=21)
    ax.set_ylabel("Count", fontsize=22)

    # --- Secondary top axis: log(M / M_Earth) ---
    LOG_MJ_OVER_ME = np.log10(317.828)

    ax_top = ax.secondary_xaxis(
        "top",
        functions=(
            lambda logMJ: logMJ + LOG_MJ_OVER_ME,
            lambda logME: logME - LOG_MJ_OVER_ME,
        ),
    )
    ax_top.set_xlabel(
        r"$\log\!\left(\frac{M}{M_\oplus}\right)$", fontsize=21, labelpad=10
    )
    ax.tick_params(axis="both", which="major", labelsize=18, length=7, width=1.5)
    ax.tick_params(axis="both", which="minor", labelsize=16, length=4, width=1.2)

    ax_top.tick_params(axis="x", which="major", labelsize=18, length=7, width=1.5)
    ax_top.tick_params(axis="x", which="minor", labelsize=16, length=4, width=1.2)

    add_nested_S_legend(ax)
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    import os

    hermes = HermesData.from_csv("dataset/hermes_synthetic_data_0.6.0.csv")
    sampler = SurveySampler(hermes, rng_seed=42)

    fig, ax = plot_logm_nested_counts(sampler, bins=30)

    desktop = os.path.expanduser("~/Desktop")
    for ext in ("pdf", "svg"):
        out = os.path.join(desktop, f"Ariel_logM_nested_counts.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print(f"[saved] {out}")
