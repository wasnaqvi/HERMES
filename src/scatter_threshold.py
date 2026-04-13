from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import arviz as az
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .Model import ModelConfig, _fit_met_survey_numpyro, _met_model, _met_model_no_stellar
from .Survey import SurveySampler
from .data import HermesData


@dataclass(frozen=True, slots=True)
class ScatterThresholdConfig:
    scatter_grid: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0)
    n_grid: tuple[int, ...] = (80, 150, 250, 400, 600)
    n_reps: int = 3
    mcmc_seeds: tuple[int, ...] = (321, 42)
    draws: int = 600
    tune: int = 600
    target_accept: float = 0.90
    num_chains: int = 4
    survey_seed: int = 42
    beta_s_true: float = 1.0
    fixed_noise_seed: int = 42
    threshold_fraction: float = 0.80


def _extract_summary(idata: az.InferenceData) -> dict[str, float]:
    row: dict[str, float] = {}
    for param in ["alpha_p", "beta_p", "beta_s", "epsilon"]:
        if param not in idata.posterior:
            continue
        samples = np.asarray(idata.posterior[param]).reshape(-1)
        row[f"{param}_mean"] = float(samples.mean())
        row[f"{param}_sd"] = float(samples.std(ddof=1)) if samples.size > 1 else 0.0
        lo, hi = np.quantile(samples, [0.16, 0.84])
        row[f"{param}_hdi16"] = float(lo)
        row[f"{param}_hdi84"] = float(hi)
    return row


def make_synthetic_catalog(
    base_df: pd.DataFrame,
    sigma: float,
    *,
    beta_s_true: float = 1.0,
    fixed_noise_seed: int = 42,
) -> pd.DataFrame:
    df = base_df.copy()
    base_logm = df["logM"].to_numpy(float)
    base_stellar = df["Star Metallicity"].to_numpy(float)
    z_fixed = np.random.default_rng(fixed_noise_seed).normal(0.0, 1.0, size=len(df))
    df["log(X_H2O)"] = -1.09 * base_logm + beta_s_true * base_stellar - 0.95 + float(sigma) * z_fixed
    return df


def run_scatter_threshold(
    base_df: pd.DataFrame,
    *,
    cfg: ScatterThresholdConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or ScatterThresholdConfig()
    model_cfg = ModelConfig(
        draws=cfg.draws,
        tune=cfg.tune,
        target_accept=cfg.target_accept,
        num_chains=cfg.num_chains,
        compute_log_lik=True,
    )

    scatter_rows: list[dict] = []
    n_surveys = len(cfg.n_grid) * 4 * cfg.n_reps
    total_fits = len(cfg.scatter_grid) * n_surveys * 2 * len(cfg.mcmc_seeds)
    fit_count = 0
    t0 = time.time()

    for sig_idx, sigma in enumerate(cfg.scatter_grid):
        syn_df = make_synthetic_catalog(
            base_df,
            sigma,
            beta_s_true=cfg.beta_s_true,
            fixed_noise_seed=cfg.fixed_noise_seed,
        )
        sampler = SurveySampler(HermesData(syn_df), rng_seed=cfg.survey_seed)
        surveys = sampler.sample_grid(cfg.n_grid, n_reps_per_combo=cfg.n_reps)

        for model_fn, model_name in [(_met_model, "3D Model"), (_met_model_no_stellar, "2D Model")]:
            for mseed in cfg.mcmc_seeds:
                rng = np.random.default_rng(mseed)

                for survey in surveys:
                    fit_count += 1
                    if fit_count % 100 == 0 or fit_count == total_fits:
                        elapsed = time.time() - t0
                        rate = fit_count / elapsed if elapsed > 0 else 0.0
                        eta = (total_fits - fit_count) / rate if rate > 0 else 0.0
                        print(
                            f"  [{fit_count}/{total_fits}] sigma={sigma:.1f} {model_name} "
                            f"({elapsed/60:.1f}m elapsed, ~{eta/60:.0f}m left)",
                            flush=True,
                        )

                    rs = int(rng.integers(0, 2**32 - 1))
                    df_sv = survey.df
                    idata = _fit_met_survey_numpyro(
                        df_sv["logM"].to_numpy(),
                        df_sv["Star Metallicity"].to_numpy(),
                        df_sv["log(X_H2O)"].to_numpy(),
                        df_sv["uncertainty_lower"].to_numpy(),
                        df_sv["uncertainty_upper"].to_numpy(),
                        df_sv["Star Metallicity Error Lower"].to_numpy(),
                        df_sv["Star Metallicity Error Upper"].to_numpy(),
                        cfg=model_cfg,
                        random_seed=rs,
                        model_fn=model_fn,
                    )

                    row = _extract_summary(idata)
                    row.update(
                        {
                            "sigma": float(sigma),
                            "model": model_name,
                            "seed": int(mseed),
                            "survey_id": survey.survey_id,
                            "class_label": survey.class_label,
                            "N": survey.n,
                            "L_mass": survey.leverage(
                                "logM",
                                err_lower_col="logM_err_lower",
                                err_upper_col="logM_err_upper",
                            ),
                            "L_stellar": survey.leverage(
                                "Star Metallicity",
                                err_lower_col="Star Metallicity Error Lower",
                                err_upper_col="Star Metallicity Error Upper",
                            ),
                        }
                    )

                    try:
                        waic = az.waic(idata)
                        row["waic"] = float(waic.elpd_waic)
                        row["waic_se"] = float(waic.se)
                    except Exception:
                        row["waic"] = np.nan
                        row["waic_se"] = np.nan

                    scatter_rows.append(row)

        print(f"  Finished sigma={sigma:.1f} ({sig_idx + 1}/{len(cfg.scatter_grid)})")

    return pd.DataFrame(scatter_rows)


def summarize_scatter_threshold(
    df_scatter: pd.DataFrame,
    *,
    threshold_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    scat_piv = df_scatter.pivot_table(
        index=["sigma", "survey_id", "seed", "class_label", "N", "L_mass", "L_stellar"],
        columns="model",
        values="waic",
    ).reset_index()
    scat_piv["delta_stellar"] = scat_piv["3D Model"] - scat_piv["2D Model"]

    det_summary = (
        scat_piv.groupby(["sigma", "N"])
        .agg(
            n_surveys=("delta_stellar", "count"),
            frac_detect=("delta_stellar", lambda x: (x > 0).mean()),
            median_delta=("delta_stellar", "median"),
            mean_delta=("delta_stellar", "mean"),
        )
        .reset_index()
    )

    det_by_sigma = (
        scat_piv.groupby("sigma")
        .agg(
            frac_detect=("delta_stellar", lambda x: (x > 0).mean()),
            median_delta=("delta_stellar", "median"),
            mean_delta=("delta_stellar", "mean"),
        )
        .reset_index()
    )

    frac_arr = det_by_sigma["frac_detect"].to_numpy(float)
    sig_arr = det_by_sigma["sigma"].to_numpy(float)
    threshold_sigma = np.nan
    for i in range(len(frac_arr) - 1):
        if frac_arr[i] >= threshold_fraction and frac_arr[i + 1] < threshold_fraction:
            threshold_sigma = sig_arr[i] + (
                (threshold_fraction - frac_arr[i]) / (frac_arr[i + 1] - frac_arr[i])
            ) * (sig_arr[i + 1] - sig_arr[i])
            break
    if np.isnan(threshold_sigma):
        if (frac_arr >= threshold_fraction).all():
            threshold_sigma = sig_arr[-1]
        elif (frac_arr < threshold_fraction).all():
            threshold_sigma = sig_arr[0]

    return det_summary, det_by_sigma, float(threshold_sigma)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_scatter_threshold_results(
    det_summary: pd.DataFrame,
    det_by_sigma: pd.DataFrame,
    *,
    threshold_sigma: float,
    scatter_n_grid: tuple[int, ...] | list[int],
    scatter_grid: tuple[float, ...] | list[float],
    plots_dir: Path,
) -> None:
    n_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(scatter_n_grid)))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, n0 in enumerate(scatter_n_grid):
        sub = det_summary[det_summary["N"] == n0].sort_values("sigma")
        ax.plot(sub["sigma"], sub["frac_detect"], "o-", color=n_colors[i], ms=5, lw=1.8, label=f"N={n0}")
    ax.plot(
        det_by_sigma["sigma"],
        det_by_sigma["frac_detect"],
        "s--",
        color="black",
        ms=6,
        lw=2.2,
        label="All N (aggregate)",
        zorder=5,
    )
    if np.isfinite(threshold_sigma):
        ax.axvline(threshold_sigma, color="firebrick", ls="--", lw=1.2, alpha=0.5)
        ax.axvspan(scatter_grid[0], threshold_sigma, alpha=0.06, color="green", zorder=0)
        ax.axvspan(threshold_sigma, scatter_grid[-1], alpha=0.06, color="red", zorder=0)
    ax.set_xlabel(r"Intrinsic scatter $\sigma$ (dex)", fontsize=12)
    ax.set_ylabel("Fraction at which 3D Model differentiates \n between stellar and planetary metallicity\n", fontsize=11)
    ax.set_title("At what intrinsic scatter does the 3D model\nlose stellar metallicity detection?", fontsize=13)
    ax.set_xlim(scatter_grid[0] - 0.05, scatter_grid[-1] + 0.05)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="lower left")
    ax.minorticks_on()
    fig.tight_layout()
    _savefig(plots_dir / "scatter_threshold_detection_curve.pdf")

    heat_piv = det_summary.pivot_table(index="N", columns="sigma", values="frac_detect").sort_index(ascending=False)
    nrows, ncols = heat_piv.shape

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.cm.YlGn
    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.imshow(heat_piv.values, aspect="auto", cmap=cmap, norm=norm)
    for i in range(nrows):
        for j in range(ncols):
            val = heat_piv.values[i, j]
            if np.isfinite(val):
                r, g, b, _ = cmap(norm(val))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "white" if luminance < 0.52 else "black"
                outline = "black" if txt_color == "white" else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.0%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color=txt_color,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground=outline)],
                )
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"{s:.1f}" for s in heat_piv.columns], fontsize=10)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([str(int(n)) for n in heat_piv.index], fontsize=10)
    ax.set_xlabel(r"Intrinsic scatter $\epsilon$ (dex)", fontsize=14)
    ax.set_ylabel("Sample size N", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Recovery fraction", fontsize=14)
    fig.tight_layout()
    _savefig(plots_dir / "scatter_threshold_heatmap_no_title.pdf")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=1)
    im = ax.imshow(heat_piv.values, aspect="auto", cmap=cmap, norm=norm)
    for i in range(nrows):
        for j in range(ncols):
            val = heat_piv.values[i, j]
            if np.isfinite(val):
                r, g, b, _ = cmap(norm(val))
                luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
                txt_color = "white" if luminance < 0.5 else "black"
                ax.text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=10, fontweight="bold", color=txt_color)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"{s:.1f}" for s in heat_piv.columns], fontsize=10)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels([str(int(n)) for n in heat_piv.index], fontsize=10)
    ax.set_xlabel(r"Intrinsic scatter $\sigma$ (dex)", fontsize=13)
    ax.set_ylabel("Sample size N", fontsize=13)
    ax.set_title("Stellar metallicity detection rate by scatter and sample size", fontsize=14)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Detection fraction", fontsize=12)
    fig.tight_layout()
    _savefig(plots_dir / "scatter_threshold_heatmap.pdf")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, n0 in enumerate(scatter_n_grid):
        sub = det_summary[det_summary["N"] == n0].sort_values("sigma")
        ax.plot(sub["sigma"], sub["median_delta"], "o-", color=n_colors[i], ms=5, lw=1.6, label=f"N={n0}")
    ax.plot(
        det_by_sigma["sigma"],
        det_by_sigma["median_delta"],
        "s--",
        color="black",
        ms=6,
        lw=2.0,
        label="All N (aggregate)",
        zorder=5,
    )
    ax.axhline(0, color="k", ls="-", lw=1.0, alpha=0.6)
    ax.fill_between(
        det_by_sigma["sigma"],
        0,
        ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -5,
        alpha=0.04,
        color="red",
        zorder=0,
    )
    ax.set_xlabel(r"Intrinsic scatter $\epsilon$ (dex)", fontsize=14)
    ax.set_ylabel(r"$\Delta$elpd$_{\mathrm{WAIC}}$", fontsize=14)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax.minorticks_on()
    fig.tight_layout()
    _savefig(plots_dir / "scatter_threshold_delta_waic.pdf")


def print_scatter_threshold_summary(
    det_summary: pd.DataFrame,
    det_by_sigma: pd.DataFrame,
    *,
    threshold_sigma: float,
    scatter_grid: tuple[float, ...] | list[float],
    threshold_fraction: float,
) -> None:
    print(f"\n=== Detection threshold ({threshold_fraction:.0%} detection rate): sigma ~ {threshold_sigma:.2f} dex ===\n")
    print("Detection fraction by sigma (aggregated across N):")
    print(det_by_sigma.to_string(index=False))
    print("\nDetection summary by (sigma, N):")
    for sig in scatter_grid:
        sub = det_summary[det_summary["sigma"] == sig]
        print(f"\n  sigma = {sig}:")
        print(sub[["N", "frac_detect", "median_delta"]].to_string(index=False))
