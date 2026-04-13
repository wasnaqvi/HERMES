from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .Model import ModelConfig, _fit_met_survey_numpyro, _met_model
from .plots import _pl_fit_unc, _powerlaw_band, add_legend, scatter_fits
from .scatter_threshold import (
    ScatterThresholdConfig,
    plot_scatter_threshold_results,
    print_scatter_threshold_summary,
    run_scatter_threshold,
    summarize_scatter_threshold,
)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
        }
    )


def extract_summary(idata: az.InferenceData) -> dict[str, float]:
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


def fit_one(survey: Any, model_fn: Any, rs: int, cfg: ModelConfig) -> az.InferenceData:
    df = survey.df
    return _fit_met_survey_numpyro(
        df["logM"].to_numpy(),
        df["Star Metallicity"].to_numpy(),
        df["log(X_H2O)"].to_numpy(),
        df["uncertainty_lower"].to_numpy(),
        df["uncertainty_upper"].to_numpy(),
        df["Star Metallicity Error Lower"].to_numpy(),
        df["Star Metallicity Error Upper"].to_numpy(),
        cfg=cfg,
        random_seed=rs,
        model_fn=model_fn,
    )


def run_experiments(
    surveys: list,
    out_dir: Path,
    *,
    draws: int,
    tune: int,
    target_accept: float,
    num_chains: int,
    compute_waic: bool,
    mcmc_seeds: list[int],
    model_variants: list[tuple[Any, str]],
) -> pd.DataFrame:
    cfg = ModelConfig(
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        num_chains=num_chains,
        compute_log_lik=compute_waic,
    )

    all_rows: list[dict[str, Any]] = []
    total = len(surveys) * len(model_variants) * len(mcmc_seeds)
    count = 0
    t0 = time.time()

    for model_fn, model_name in model_variants:
        print(f"\n--- {model_name} ---")
        for mseed in mcmc_seeds:
            rng = np.random.default_rng(mseed)
            for survey in surveys:
                count += 1
                if count % 50 == 0 or count == total:
                    elapsed = time.time() - t0
                    rate = count / elapsed if elapsed > 0 else 1.0
                    eta = (total - count) / rate
                    print(
                        f"  [{count}/{total}] {model_name} seed={mseed} "
                        f"({elapsed/60:.1f}m, ~{eta/60:.0f}m left)",
                        flush=True,
                    )

                idata = fit_one(survey, model_fn, int(rng.integers(0, 2**32 - 1)), cfg)
                row = extract_summary(idata)
                row.update(
                    {
                        "model": model_name,
                        "seed": mseed,
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

                if compute_waic:
                    try:
                        waic = az.waic(idata)
                        row["waic"] = float(waic.elpd_waic)
                        row["waic_se"] = float(waic.se)
                    except Exception:
                        row["waic"] = np.nan
                        row["waic_se"] = np.nan

                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "hermes_extended_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}  ({len(df)} rows)")
    return df


def fit_oracle(
    hermes_df: pd.DataFrame,
    *,
    draws: int,
    tune: int,
    target_accept: float,
    num_chains: int,
) -> dict[str, float]:
    print("Fitting oracle on full catalog ...")
    cfg = ModelConfig(
        draws=draws * 2,
        tune=tune * 2,
        target_accept=target_accept,
        num_chains=num_chains,
        compute_log_lik=False,
    )
    idata = _fit_met_survey_numpyro(
        hermes_df["logM"].to_numpy(),
        hermes_df["Star Metallicity"].to_numpy(),
        hermes_df["log(X_H2O)"].to_numpy(),
        hermes_df["uncertainty_lower"].to_numpy(),
        hermes_df["uncertainty_upper"].to_numpy(),
        hermes_df["Star Metallicity Error Lower"].to_numpy(),
        hermes_df["Star Metallicity Error Upper"].to_numpy(),
        cfg=cfg,
        random_seed=0,
    )
    ref = {
        param: float(np.asarray(idata.posterior[param]).mean())
        for param in ["alpha_p", "beta_p", "beta_s", "epsilon"]
    }
    print("Oracle:", ref)
    return ref


def add_zscores(df: pd.DataFrame, ref: dict[str, float]) -> pd.DataFrame:
    for param in ["alpha_p", "beta_p", "beta_s", "epsilon"]:
        mean_col, sd_col = f"{param}_mean", f"{param}_sd"
        if mean_col in df.columns and sd_col in df.columns:
            df[f"z_{param}"] = (df[mean_col] - ref[param]) / df[sd_col].clip(lower=1e-10)
    return df


def print_zscore_summary(df_primary: pd.DataFrame, z_params: list[str]) -> None:
    print("\n=== Per-parameter z-score summary (across all surveys) ===")
    rows = []
    for param in z_params:
        values = df_primary[f"z_{param}"].dropna()
        rows.append(
            {
                "parameter": param,
                "mean(z)": f"{values.mean():.3f}",
                "SD(z)": f"{values.std():.3f}",
                "median(z)": f"{values.median():.3f}",
                "|z|<1 (%)": f"{(np.abs(values) < 1).mean() * 100:.1f}",
                "|z|<2 (%)": f"{(np.abs(values) < 2).mean() * 100:.1f}",
                "max|z|": f"{np.abs(values).max():.2f}",
            }
        )
    print(pd.DataFrame(rows).set_index("parameter").to_string())
    print("\nCalibration target: mean(z)~0, SD(z)~1, |z|<1 ~ 68%")


def print_top5(df_primary: pd.DataFrame, z_params: list[str]) -> None:
    cols = ["survey_id", "class_label", "N", "L_mass", "L_stellar"]
    for param in z_params:
        z_col = f"z_{param}"
        if z_col not in df_primary.columns:
            continue
        top = (
            df_primary.assign(_az=lambda d: d[z_col].abs())
            .nsmallest(5, "_az")[cols + [z_col]]
            .reset_index(drop=True)
        )
        print(f"\n--- Top 5 surveys for {param} (smallest |z|) ---")
        print(top.to_string(index=False))


def print_waic(df_results: pd.DataFrame) -> None:
    if "waic" not in df_results.columns or df_results["waic"].isna().all():
        print("WAIC not available.")
        return
    piv = df_results.pivot_table(
        index=["survey_id", "seed", "class_label", "N", "L_mass", "L_stellar"],
        columns="model",
        values="waic",
    ).reset_index()
    if "3D Model" in piv.columns and "2D Model" in piv.columns:
        piv["delta_3D_vs_2D"] = piv["3D Model"] - piv["2D Model"]
        print("\n=== WAIC: delta_elpd(3D - 2D), median by class ===")
        print(piv.groupby("class_label")["delta_3D_vs_2D"].median().round(3).to_string())
    print("\n=== Mean WAIC by model ===")
    print(df_results.groupby("model")["waic"].mean().round(3).to_string())


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_sigma_vs_leverage(
    df_pl: pd.DataFrame,
    plots_dir: Path,
    *,
    cls_ord: list[str],
) -> None:
    param_sets = [
        [("beta_p_sd", r"$\sigma_{\beta_p}$")],
        [("beta_s_sd", r"$\sigma_{\beta_s}$")],
        [("alpha_p_sd", r"$\sigma_{\alpha_p}$"), ("epsilon_sd", r"$\sigma_{\varepsilon}$")],
    ]
    fnames = ["sigma_beta_p_vs_L", "beta_s_sd_vs_leverage", "sigma_alpha_p_vs_L"]

    for n0 in sorted(df_pl["N"].unique()):
        sub = df_pl[df_pl["N"] == n0]
        labels = sub["class_label"].to_numpy(str)
        if len(sub) < 3:
            continue

        for params, fname in zip(param_sets[:2], fnames[:2]):
            sd_col, ylabel = params[0]
            if sd_col not in sub.columns:
                continue
            fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
            fig.suptitle(rf"$N={n0}$", fontsize=12)
            scatter_fits(axes[0], sub["L_mass"].values, sub[sd_col].values, labels, ylabel, r"$L_{\mathrm{mass}}$")
            scatter_fits(axes[1], sub["L_stellar"].values, sub[sd_col].values, labels, ylabel, r"$L_{\mathrm{stellar}}$")
            add_legend(axes[0], sub)
            fig.tight_layout()
            savefig(plots_dir / f"{fname}_N{n0}.pdf")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        fig.suptitle(rf"$N={n0}$", fontsize=12)
        if "alpha_p_sd" in sub.columns:
            scatter_fits(axes[0], sub["L_mass"].values, sub["alpha_p_sd"].values, labels, r"$\sigma_{\alpha_p}$", r"$L_{\mathrm{mass}}$")
        if "epsilon_sd" in sub.columns:
            scatter_fits(axes[1], sub["L_mass"].values, sub["epsilon_sd"].values, labels, r"$\sigma_{\varepsilon}$", r"$L_{\mathrm{mass}}$")
        add_legend(axes[0], sub)
        fig.tight_layout()
        savefig(plots_dir / f"sigma_alpha_p_vs_L_N{n0}.pdf")


def plot_sigma_vs_N(
    df_pn: pd.DataFrame,
    plots_dir: Path,
    *,
    cls_ord: list[str],
    cls_clr: dict[str, str],
) -> None:
    sd_params = [
        ("alpha_p_sd", r"$\sigma_{\alpha_p}$", r"\sigma_\alpha"),
        ("beta_p_sd", r"$\sigma_{\beta_p}$", r"\sigma_{\beta_p}"),
        ("beta_s_sd", r"$\sigma_{\beta_s}$", r"\sigma_{\beta_s}"),
        ("epsilon_sd", r"$\sigma_{\varepsilon}$", r"\sigma_\varepsilon"),
    ]
    sd_params = [(c, l, t) for c, l, t in sd_params if c in df_pn.columns]
    if not sd_params:
        return

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes_flat = axes.flatten()

    for ax, (sd_col, ylabel, tex_sym) in zip(axes_flat, sd_params):
        for cls in cls_ord:
            csub = df_pn[df_pn["class_label"] == cls]
            if csub.empty:
                continue
            ax.scatter(csub["N"].values, csub[sd_col].values, s=20, alpha=0.7, color=cls_clr.get(cls, "k"), label=cls)

        x_all = df_pn["N"].values.astype(float)
        y_all = df_pn[sd_col].values.astype(float)
        mask = np.isfinite(x_all) & np.isfinite(y_all) & (x_all > 0) & (y_all > 0)
        if mask.sum() >= 2:
            ng = np.linspace(x_all[mask].min() * 0.9, x_all[mask].max() * 1.05, 200)
            yh, lo, hi = _powerlaw_band(x_all[mask], y_all[mask], ng, prediction=True)
            ax.fill_between(ng, lo, hi, alpha=0.12, color="grey", linewidth=0)
            ax.plot(ng, yh, "k--", lw=1.2, alpha=0.6)
            b_exp, b_err = _pl_fit_unc(x_all[mask], y_all[mask])
            ann = rf"${tex_sym} \propto N^{{{b_exp:.2f}\;^{{+{b_err:.2f}}}_{{-{b_err:.2f}}}}}$"
            ax.text(0.05, 0.95, ann, transform=ax.transAxes, fontsize=13, va="top")

        ax.set_xlabel(r"$N$", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.minorticks_on()

    add_legend(axes_flat[0], df_pn)
    fig.tight_layout()
    savefig(plots_dir / "uncertainty_vs_N.pdf")

    print("\nMean posterior SD by class and N:")
    for sd_col, ylabel, _ in sd_params:
        print(f"\n{ylabel}:")
        tab = df_pn.pivot_table(index="class_label", columns="N", values=sd_col, aggfunc="mean")
        print(tab.round(4).to_string())


def plot_zscores(
    df_z: pd.DataFrame,
    z_params: list[str],
    plots_dir: Path,
    *,
    cls_ord: list[str],
    cls_clr: dict[str, str],
) -> None:
    z_labels = {
        "alpha_p": r"$z(\alpha_p)$",
        "beta_p": r"$z(\beta_p)$",
        "beta_s": r"$z(\beta_s)$",
        "epsilon": r"$z(\varepsilon)$",
    }
    xgrid = np.linspace(-4, 4, 200)
    gauss = np.exp(-0.5 * xgrid**2) / np.sqrt(2 * np.pi)

    n_par = len(z_params)
    fig, axes = plt.subplots(1, n_par, figsize=(3.8 * n_par, 3.5))
    if n_par == 1:
        axes = [axes]
    for ax, param in zip(axes, z_params):
        vals = df_z[f"z_{param}"].dropna()
        ax.hist(vals, bins=25, density=True, alpha=0.6, edgecolor="k", lw=0.5)
        ax.plot(xgrid, gauss, "r--", lw=1.2, label="N(0,1)")
        ax.set_xlabel(z_labels.get(param, f"z({param})"))
        ax.set_ylabel("density")
        ax.legend(fontsize=9, frameon=False)
        ax.minorticks_on()
    fig.tight_layout()
    savefig(plots_dir / "z_histograms.pdf")

    for param in z_params:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        for ax, lcol, llab in [
            (axes[0], "L_mass", r"$L_{\mathrm{mass}}$"),
            (axes[1], "L_stellar", r"$L_{\mathrm{stellar}}$"),
        ]:
            for cls in cls_ord:
                cm = df_z["class_label"] == cls
                if not cm.any():
                    continue
                ax.scatter(df_z.loc[cm, lcol], df_z.loc[cm, f"z_{param}"], s=18, alpha=0.7, color=cls_clr.get(cls, "k"), label=cls)
            xlims = ax.get_xlim()
            ax.fill_between(xlims, -1, 1, alpha=0.10, color="cornflowerblue", zorder=0)
            ax.set_xlim(xlims)
            ax.axhline(0, color="grey", ls="--", lw=0.8)
            ax.axhline(1, color="grey", ls=":", lw=0.6)
            ax.axhline(-1, color="grey", ls=":", lw=0.6)
            ax.set_xlabel(llab)
            ax.set_ylabel(z_labels.get(param, f"z({param})"))
            ax.minorticks_on()
        add_legend(axes[0], df_z)
        fig.tight_layout()
        savefig(plots_dir / f"z_{param}_vs_leverage.pdf")


def plot_multiseed(
    df_results: pd.DataFrame,
    plots_dir: Path,
    *,
    primary: str,
    mcmc_seeds: list[int],
    cls_ord: list[str],
    cls_clr: dict[str, str],
) -> None:
    if len(mcmc_seeds) < 2:
        print("Only 1 seed — skipping convergence check.")
        return

    df_div = df_results[df_results["model"] == primary].copy()
    grp = (
        df_div.groupby("survey_id")
        .agg(
            N=("N", "first"),
            class_label=("class_label", "first"),
            L_mass=("L_mass", "first"),
            L_stellar=("L_stellar", "first"),
            bp_spread=("beta_p_mean", lambda x: x.max() - x.min()),
            bp_sd_mean=("beta_p_sd", "mean"),
            bs_spread=("beta_s_mean", lambda x: x.max() - x.min()),
            bs_sd_mean=("beta_s_sd", "mean"),
        )
        .reset_index()
    )
    grp["bp_ratio"] = grp["bp_spread"] / grp["bp_sd_mean"].clip(lower=1e-10)
    grp["bs_ratio"] = grp["bs_spread"] / grp["bs_sd_mean"].clip(lower=1e-10)

    labels = grp["class_label"].to_numpy(str)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(f"MCMC convergence: seed-range / posterior SD ({len(mcmc_seeds)} seeds)", fontsize=11)
    for ax, col, pname, lcol, llab in [
        (axes[0], "bp_ratio", r"$\beta_p$", "L_mass", r"$L_{\mathrm{mass}}$"),
        (axes[1], "bs_ratio", r"$\beta_s$", "L_stellar", r"$L_{\mathrm{stellar}}$"),
    ]:
        for cls in cls_ord:
            cm = labels == cls
            if not cm.any():
                continue
            ax.scatter(grp.loc[cm, lcol], grp.loc[cm, col], s=18, alpha=0.7, color=cls_clr.get(cls, "k"), label=cls)
        ax.axhline(0.1, color="green", ls="--", lw=0.8, alpha=0.6)
        ax.axhline(0.5, color="red", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel(llab)
        ax.set_ylabel(f"Seed range / posterior SD ({pname})")
        ax.set_title(f"{pname}: < 0.1 = well converged")
        ax.minorticks_on()
    add_legend(axes[0], grp)
    fig.tight_layout()
    savefig(plots_dir / "mcmc_convergence.pdf")


def plot_model_comparison(
    df_results: pd.DataFrame,
    plots_dir: Path,
    *,
    seed0: int,
) -> None:
    model_names = sorted(df_results["model"].unique())
    n_mod = len(model_names)

    for n0 in sorted(df_results["N"].unique()):
        sub = df_results[(df_results["N"] == n0) & (df_results["seed"] == seed0)]
        if len(sub) < 3:
            continue

        for lcol, llab, fname_sfx in [
            ("L_mass", r"$L_{\mathrm{mass}}$", "L_mass"),
            ("L_stellar", r"$L_{\mathrm{stellar}}$", "L_stellar"),
        ]:
            fig, axes = plt.subplots(1, n_mod, figsize=(4.2 * n_mod, 4), sharey=True)
            if n_mod == 1:
                axes = [axes]
            fig.suptitle(rf"$N={n0}$", fontsize=12)
            for ax, model_name in zip(axes, model_names):
                msub = sub[sub["model"] == model_name]
                if msub.empty:
                    ax.set_title(model_name)
                    continue
                scatter_fits(ax, msub[lcol].values, msub["beta_p_sd"].values, msub["class_label"].to_numpy(str), r"$\sigma_{\beta_p}$", llab)
                ax.set_title(model_name, fontsize=10)
            add_legend(axes[0], sub)
            fig.tight_layout()
            savefig(plots_dir / f"beta_p_sd_by_model_{fname_sfx}_N{n0}.pdf")


def plot_epsilon_vs_L(df_sc: pd.DataFrame, plots_dir: Path) -> None:
    if "epsilon_mean" not in df_sc.columns:
        return
    for n0 in sorted(df_sc["N"].unique()):
        sub = df_sc[df_sc["N"] == n0]
        if len(sub) < 3:
            continue
        labels = sub["class_label"].to_numpy(str)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
        fig.suptitle(rf"$N={n0}$", fontsize=12)
        scatter_fits(axes[0], sub["L_mass"].values, sub["epsilon_mean"].values, labels, r"$\hat{\varepsilon}$", r"$L_{\mathrm{mass}}$")
        scatter_fits(axes[1], sub["L_stellar"].values, sub["epsilon_mean"].values, labels, r"$\hat{\varepsilon}$", r"$L_{\mathrm{stellar}}$")
        add_legend(axes[0], sub)
        fig.tight_layout()
        savefig(plots_dir / f"epsilon_vs_L_N{n0}.pdf")


def plot_best_survey_fits(
    df_results: pd.DataFrame,
    surveys: list,
    plots_dir: Path,
    *,
    primary: str,
    mcmc_seed0: int,
    draws: int,
    tune: int,
    target_accept: float,
    num_chains: int,
    cls_clr: dict[str, str],
) -> None:
    mask = (df_results["model"] == primary) & (df_results["seed"] == mcmc_seed0)
    df_sel = df_results[mask].copy()
    df_sel["abs_z_beta_p"] = df_sel["z_beta_p"].abs()
    best_rows = df_sel.loc[df_sel.groupby("class_label")["abs_z_beta_p"].idxmin()]
    survey_lookup = {sv.survey_id: sv for sv in surveys}

    print("\nBest survey per class (smallest |z_beta_p|):")
    for _, row in best_rows.iterrows():
        print(f"  {row['class_label']}: Survey {int(row['survey_id'])}, N={int(row['N'])}, z(beta_p)={row['z_beta_p']:+.3f}")

    cfg = ModelConfig(
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        num_chains=num_chains,
        compute_log_lik=False,
    )

    for _, row in best_rows.iterrows():
        sv = survey_lookup[int(row["survey_id"])]
        df_sv = sv.df
        x_m = df_sv["logM"].to_numpy(float)
        x_s = df_sv["Star Metallicity"].to_numpy(float)
        y = df_sv["log(X_H2O)"].to_numpy(float)
        el_p = df_sv["uncertainty_lower"].to_numpy(float)
        eh_p = df_sv["uncertainty_upper"].to_numpy(float)
        yerr = np.clip(0.5 * (np.abs(el_p) + np.abs(eh_p)), 1e-6, None)
        el_s = df_sv["Star Metallicity Error Lower"].to_numpy(float)
        eh_s = df_sv["Star Metallicity Error Upper"].to_numpy(float)
        xerr_s = np.clip(0.5 * (np.abs(el_s) + np.abs(eh_s)), 1e-6, None)

        idata_best = fit_one(sv, _met_model, mcmc_seed0, cfg)
        a = np.asarray(idata_best.posterior["alpha_p"]).reshape(-1)
        bp = np.asarray(idata_best.posterior["beta_p"]).reshape(-1)
        bs = np.asarray(idata_best.posterior["beta_s"]).reshape(-1)

        x_m_c = x_m - float(np.mean(x_m))
        x_s_c = x_s - float(np.mean(x_s))
        m_grid = np.linspace(x_m_c.min(), x_m_c.max(), 250)
        s_grid = np.linspace(x_s_c.min(), x_s_c.max(), 250)
        mu_m = a[:, None] + bp[:, None] * m_grid[None, :]
        mu_s = a[:, None] + bs[:, None] * s_grid[None, :]

        color = cls_clr.get(sv.class_label, "C0")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
        fig.suptitle(rf"{sv.class_label}  |  N={sv.n}  |  " rf"$z(\beta_p)={row['z_beta_p']:+.3f}$", fontsize=12)

        ax = axes[0]
        ax.errorbar(x_m_c, y, yerr=yerr, fmt="o", ms=4, alpha=0.85, capsize=2, lw=0.9, color=color)
        ax.plot(m_grid, np.median(mu_m, axis=0), "k-", lw=1.5, label="Posterior median")
        ax.fill_between(m_grid, np.quantile(mu_m, 0.16, axis=0), np.quantile(mu_m, 0.84, axis=0), alpha=0.25, color="grey", label="68% CI")
        ax.set_xlabel(r"$\log M/M_J$ (centred)", fontsize=14)
        ax.set_ylabel(r"$\log(X_{\mathrm{H_2O}})$", fontsize=14)
        ax.legend(fontsize=10, frameon=False)
        ax.minorticks_on()

        ax = axes[1]
        ax.errorbar(x_s_c, y, yerr=yerr, xerr=xerr_s, fmt="o", ms=4, alpha=0.85, capsize=2, lw=0.9, color=color)
        ax.plot(s_grid, np.median(mu_s, axis=0), "k-", lw=1.5, label="Posterior median")
        ax.fill_between(s_grid, np.quantile(mu_s, 0.16, axis=0), np.quantile(mu_s, 0.84, axis=0), alpha=0.25, color="grey", label="68% CI")
        ax.set_xlabel(r"Stellar metallicity (centred, dex)", fontsize=14)
        ax.set_ylabel(r"$\log(X_{\mathrm{H_2O}})$", fontsize=14)
        ax.legend(fontsize=10, frameon=False)
        ax.minorticks_on()

        savefig(plots_dir / f"best_survey_{sv.class_label}.pdf")

    print("\n" + "=" * 70)
    print("PLANET NAMES IN BEST SURVEYS")
    for _, row in best_rows.iterrows():
        sv = survey_lookup[int(row["survey_id"])]
        print(f"\n--- {sv.class_label}: Survey {int(row['survey_id'])} (N={int(row['N'])}, z(beta_p)={row['z_beta_p']:+.3f}) ---")
        names = sv.df["Planet Name"].tolist() if "Planet Name" in sv.df.columns else sv.planet_names
        for nm in sorted(set(names)):
            print(f"  {nm}")


def run_scatter_threshold_pipeline(
    hermes_df: pd.DataFrame,
    results_dir: Path,
    plots_dir: Path,
    *,
    scatter_cfg: ScatterThresholdConfig,
) -> None:
    csv_path = results_dir / "hermes_scatter_threshold_results.csv"
    if csv_path.exists():
        print(f"\nLoading cached scatter-threshold results from {csv_path}")
        df_scatter = pd.read_csv(csv_path)
    else:
        print(
            "\nScatter experiment: "
            f"{len(scatter_cfg.scatter_grid)} scatter values x "
            f"{len(scatter_cfg.n_grid) * 4 * scatter_cfg.n_reps} surveys x "
            f"2 models x {len(scatter_cfg.mcmc_seeds)} seeds"
        )
        df_scatter = run_scatter_threshold(hermes_df, cfg=scatter_cfg)
        df_scatter.to_csv(csv_path, index=False)
        print(f"Saved → {csv_path}  ({len(df_scatter)} rows)")

    det_summary, det_by_sigma, threshold_sigma = summarize_scatter_threshold(
        df_scatter,
        threshold_fraction=scatter_cfg.threshold_fraction,
    )
    print_scatter_threshold_summary(
        det_summary,
        det_by_sigma,
        threshold_sigma=threshold_sigma,
        scatter_grid=scatter_cfg.scatter_grid,
        threshold_fraction=scatter_cfg.threshold_fraction,
    )
    plot_scatter_threshold_results(
        det_summary,
        det_by_sigma,
        threshold_sigma=threshold_sigma,
        scatter_n_grid=scatter_cfg.n_grid,
        scatter_grid=scatter_cfg.scatter_grid,
        plots_dir=plots_dir,
    )
