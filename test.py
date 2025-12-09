# comparison.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def _ensure_dir(path: str | Path) -> Path:
    """Create parent directories for a file path and return the Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_waic_comparison(
    df: pd.DataFrame,
    model_key_1: str,
    model_key_2: str,
    out_path: str | Path,
) -> None:
    """
    Scatter plot of ΔWAIC = WAIC(model2) − WAIC(model1) vs leverage L(logM),
    colour-coded by survey size N.

    Negative ΔWAIC -> model2 preferred; positive -> model1 preferred.
    """
    out_path = _ensure_dir(out_path)

    if not {f"{model_key_1}_waic", f"{model_key_2}_waic"}.issubset(df.columns):
        print("WAIC columns not found in comparison DataFrame; skipping WAIC plot.")
        return

    waic1 = df[f"{model_key_1}_waic"].to_numpy(float)
    waic2 = df[f"{model_key_2}_waic"].to_numpy(float)
    d_waic = waic2 - waic1
    L = df["L_logM"].to_numpy(float)
    N = df["N"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    sc = ax.scatter(L, d_waic, c=N, s=35, alpha=0.9)
    ax.axhline(0.0, linestyle="--", linewidth=1.1)

    ax.set_xlabel(r"$L(\log M)$")
    ax.set_ylabel(r"$\Delta \mathrm{WAIC}$")
    ax.set_title(
        rf"$\Delta \mathrm{{WAIC}} = \mathrm{{WAIC}}({model_key_2})"
        rf" - \mathrm{{WAIC}}({model_key_1})$"
    )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$N$ (survey size)")

    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_uncertainty_ratio_vs_L(
    df: pd.DataFrame,
    param_sd_1d: str,
    param_sd_nd: str,
    out_path: str | Path,
    label_1d: str,
    label_nd: str,
) -> None:
    """
    Plot SD_nd / SD_1d vs leverage, colour-coded by N.

    Example:
        plot_uncertainty_ratio_vs_L(
            comp_df,
            param_sd_1d="lin1d_beta_sd",
            param_sd_nd="met_beta_m_sd",
            out_path=...,
            label_1d=r"\\beta_{\\mathrm{1D}}",
            label_nd=r"\\beta_m",
        )
    """
    out_path = _ensure_dir(out_path)

    if not {param_sd_1d, param_sd_nd}.issubset(df.columns):
        print(
            f"Columns '{param_sd_1d}' and/or '{param_sd_nd}' not found; "
            "skipping this uncertainty-ratio plot."
        )
        return

    s1 = df[param_sd_1d].to_numpy(float)
    s2 = df[param_sd_nd].to_numpy(float)
    ratio = s2 / s1
    L = df["L_logM"].to_numpy(float)
    N = df["N"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    sc = ax.scatter(L, ratio, c=N, s=35, alpha=0.9)

    ax.axhline(1.0, linestyle="--", linewidth=1.1)
    ax.set_xlabel(r"$L(\log M)$")
    ax.set_ylabel(
        rf"$\mathrm{{SD}}({label_nd}) / \mathrm{{SD}}({label_1d})$"
    )
    ax.set_title("Relative parameter uncertainty vs leverage")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$N$ (survey size)")

    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------
# main: *no sampling*, just read existing comparison CSV
# ---------------------------------------------------------------------
def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    plots_dir = results_dir / "plots_test"

    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    comp_path = results_dir / "hermes_model_comparison.csv"
    if not comp_path.exists():
        raise FileNotFoundError(
            f"Could not find '{comp_path}'.\n"
            "This script now expects an existing comparison table with WAIC and "
            "parameter summaries. Run your full modelling pipeline once to "
            "create hermes_model_comparison.csv, then re-run this script."
        )

    comp_df = pd.read_csv(comp_path)

    # basic sanity checks
    required_cols = {"survey_id", "class_label", "N", "L_logM"}
    missing = required_cols - set(comp_df.columns)
    if missing:
        raise ValueError(
            f"hermes_model_comparison.csv is missing columns: {missing}"
        )

    # 1) ΔWAIC: does MetModel beat the 1D Model?
    plot_waic_comparison(
        comp_df,
        model_key_1="lin1d",
        model_key_2="met",
        out_path=plots_dir / "waic_lin1d_vs_met.pdf",
    )

    # 2) Relative uncertainty on β(logM): SD(β_m) / SD(β_1D)
    plot_uncertainty_ratio_vs_L(
        comp_df,
        param_sd_1d="lin1d_beta_sd",
        param_sd_nd="met_beta_m_sd",
        out_path=plots_dir / "uncertainty_ratio_beta_mass.pdf",
        label_1d=r"\beta_{\mathrm{1D}}",
        label_nd=r"\beta_m",
    )

    # 3) Relative uncertainty on intrinsic scatter: SD(σ_p) / SD(σ_1D)
    plot_uncertainty_ratio_vs_L(
        comp_df,
        param_sd_1d="lin1d_sigma_sd",
        param_sd_nd="met_sigma_p_sd",
        out_path=plots_dir / "uncertainty_ratio_sigma.pdf",
        label_1d=r"\sigma_{\mathrm{1D}}",
        label_nd=r"\sigma_p",
    )

    print("Comparison plots written to:", plots_dir)


if __name__ == "__main__":
    main()
