import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from src.data import HermesData
from src.Survey import SurveySampler


def plot_logM_with_classes(
    sampler: SurveySampler,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
):
    """
    Overlaid histogram of logM for:
      - parent sample (all hermes planets) as black outline
      - nested classes S1–S4 as coloured filled histograms

    Colours:
      S1: blue
      S2: orange
      S3: green
      S4: red
    """
    # parent / original sample
    global_logm = sampler.hermes.df["logM"].to_numpy(float)
    global_logm = global_logm[np.isfinite(global_logm)]

    subset_order = ["S1", "S2", "S3", "S4"]
    colors = {"S1": "blue", "S2": "orange", "S3": "green", "S4": "red"}

    mass_classes = sampler.mass_classes
    data_arrays = {}
    for label in subset_order:
        if label in mass_classes:
            arr = mass_classes[label]["logM"].to_numpy(float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                data_arrays[label] = arr

    # bins based on the parent sample
    xmin = np.min(global_logm)
    xmax = np.max(global_logm)
    bin_edges = np.linspace(xmin, xmax, bins)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # --- parent distribution: black outline so it stays visible ---
    ax.hist(
        global_logm,
        bins=bin_edges,
        density=True,
        histtype="step",
        linewidth=2.0,
        color="black",
        label=f"Entire Ariel MCS (n={len(global_logm)})",
        zorder=5,
    )

    # --- overlaid nested classes as filled coloured histograms ---
    for label in subset_order:
        if label not in data_arrays:
            continue
        arr = data_arrays[label]
        ax.hist(
            arr,
            bins=bin_edges,
            density=True,
            alpha=0.5,
            histtype="stepfilled",
            edgecolor="none",
            label=f"{label} (n={len(arr)})",
            color=colors[label],
            zorder=3,
        )

    ax.set_xlabel(r"$\log\!\left(\frac{M}{M_\mathrm{J}}\right)$")
    ax.set_ylabel("Density")
    ax.set_title("Ariel MCS and Nested Survey Selection")
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # use HermesData wrapper, not a raw DataFrame
    hermes = HermesData.from_csv("dataset/hermes_synthetic_data.csv")
    sampler = SurveySampler(hermes, rng_seed=42)

    fig, ax = plot_logM_with_classes(sampler, bins=30)
    plt.savefig("results/Ariel_logM_with_classes.pdf", bbox_inches="tight")
    plt.show()
