import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple

from src.data import HermesData
from src.Survey import SurveySampler


def add_nested_S_legend(ax):
    """
    Legend encoding nesting:
      S1: all colours
      S2: orange + green + red
      S3: green + red
      S4: red only
    """
    handles = [
        (Patch(facecolor="blue", edgecolor="none", alpha=0.35),
         Patch(facecolor="orange", edgecolor="none", alpha=0.35),
         Patch(facecolor="green", edgecolor="none", alpha=0.35),
         Patch(facecolor="red", edgecolor="none", alpha=0.35)),
        (Patch(facecolor="orange", edgecolor="none", alpha=0.35),
         Patch(facecolor="green", edgecolor="none", alpha=0.35),
         Patch(facecolor="red", edgecolor="none", alpha=0.35)),
        (Patch(facecolor="green", edgecolor="none", alpha=0.35),
         Patch(facecolor="red", edgecolor="none", alpha=0.35)),
        (Patch(facecolor="red", edgecolor="none", alpha=0.35),),
    ]

    labels = [
        "S1 (entire Ariel MCS)",
        "S2 (logM ≥ q25)",
        "S3 (logM ≥ q50)",
        "S4 (logM ≥ q75)",
    ]

    ax.legend(
        handles,
        labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title="Nested mass classes",
        frameon=True,
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

    ax.hist(s1, bins=bin_edges, alpha=0.35, color="blue",   edgecolor="none", zorder=3)
    ax.hist(s2, bins=bin_edges, alpha=0.35, color="orange", edgecolor="none", zorder=3)
    ax.hist(s3, bins=bin_edges, alpha=0.35, color="green",  edgecolor="none", zorder=3)
    ax.hist(s4, bins=bin_edges, alpha=0.35, color="red",    edgecolor="none", zorder=3)

    ax.set_xlabel(r"$\log\!\left(\frac{M}{M_\mathrm{J}}\right)$")
    ax.set_ylabel("Count")
    ax.set_title("Ariel MCS Nested Mass Classes (Counts)")

    # custom legend (replaces ax.legend())
    add_nested_S_legend(ax)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    hermes = HermesData.from_csv("dataset/hermes_synthetic_data.csv")
    sampler = SurveySampler(hermes, rng_seed=42)

    fig, ax = plot_logm_nested_counts(sampler, bins=30)
    fig.savefig("results/Ariel_logM_nested_counts.pdf", bbox_inches="tight")
    plt.show()
