# comparison.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from src.data import HermesData
from src.Survey import SurveySampler, Survey
from src.Model import Model, MetModel


def _ensure_dir(path: str | Path) -> Path:
    """Create parent directories and return Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_models_on_surveys(
    surveys: List[Survey],
    models: Dict[str, object],
    seed: int = 123,
) -> pd.DataFrame:
    """
    Fit each model in `models` on every survey and collect:

      - basic survey info:
          survey_id, class_label, N, L_met, L_logM

      - per-model WAIC / ELPD-WAIC:
          <key>_elpd_waic, <key>_p_waic,
          <key>_elpd_waic_se (if available),
          <key>_waic, <key>_waic_se

      - per-model parameter summaries, using that model's own
        summarize_single(...) and prefixing all keys with <key>_.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for survey in surveys:
        base = {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": survey.leverage(col="log(X_H2O)"),
            "L_logM": survey.leverage(col="logM"),
        }

        row = dict(base)

        for key, model in models.items():
            rs = int(rng.integers(0, 2**32 - 1))

            # --- fit this model on this survey ---
            idata = model.fit_survey(survey, random_seed=rs)

            # --- WAIC (ArviZ >= 0.15 returns ELPDData) ---
            try:
                waic_res = az.waic(idata)

                # These *do* exist in your version
                elpd_waic = float(waic_res.elpd_waic)
                p_waic = float(waic_res.p_waic)

                # SE attribute name is version-dependent; grab it if present
                elpd_se_attr = None
                for cand in ("elpd_waic_se", "se", "elpd_se"):
                    if hasattr(waic_res, cand):
                        elpd_se_attr = cand
                        break

                if elpd_se_attr is not None:
                    elpd_waic_se = float(getattr(waic_res, elpd_se_attr))
                else:
                    elpd_waic_se = np.nan

                # Convert to deviance-style WAIC:
                #   WAIC = -2 * elpd_waic
                waic_val = -2.0 * elpd_waic
                waic_se = np.nan if np.isnan(elpd_waic_se) else 2.0 * elpd_waic_se

                row[f"{key}_elpd_waic"] = elpd_waic
                row[f"{key}_p_waic"] = p_waic
                row[f"{key}_elpd_waic_se"] = elpd_waic_se
                row[f"{key}_waic"] = waic_val
                row[f"{key}_waic_se"] = waic_se

            except Exception as e:
                print(f"WAIC failed for model '{key}' on survey {survey.survey_id}: {e}")
                row[f"{key}_elpd_waic"] = np.nan
                row[f"{key}_p_waic"] = np.nan
                row[f"{key}_elpd_waic_se"] = np.nan
                row[f"{key}_waic"] = np.nan
                row[f"{key}_waic_se"] = np.nan

            # --- model-specific parameter summaries ---
            summary_dict = model.summarize_single(survey, idata)
            for sk, sv in summary_dict.items():
                if sk in base:
                    continue
                row[f"{key}_{sk}"] = sv

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("survey_id")
        .reset_index(drop=True)
    )


def plot_waic_comparison(
    df: pd.DataFrame,
    model_key_1: str,
    model_key_2: str,
    out_path: str | Path,
) -> None:
    """
    Scatter of ΔWAIC = WAIC(model2) - WAIC(model1) vs leverage L_logM,
    colour-coded by N. Negative ΔWAIC => model2 preferred.
    """
    out_path = _ensure_dir(out_path)

    waic1 = df[f"{model_key_1}_waic"].to_numpy(float)
    waic2 = df[f"{model_key_2}_waic"].to_numpy(float)
    d_waic = waic2 - waic1
    L = df["L_logM"].to_numpy(float)
    N = df["N"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sc = ax.scatter(L, d_waic, c=N, s=30, alpha=0.9)
    ax.axhline(0.0, linestyle="--", linewidth=1.0)

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
    label_1d: str = "1D",
    label_nd: str = "ND",
) -> None:
    """
    Plot the ratio of posterior SDs (multi-predictor vs 1D) vs leverage.
    """
    out_path = _ensure_dir(out_path)

    s1 = df[param_sd_1d].to_numpy(float)
    s2 = df[param_sd_nd].to_numpy(float)
    ratio = s2 / s1
    L = df["L_logM"].to_numpy(float)
    N = df["N"].to_numpy(int)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    sc = ax.scatter(L, ratio, c=N, s=30, alpha=0.9)

    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(r"$L(\log M)$")
    ax.set_ylabel(
        rf"$\mathrm{{SD}}({label_nd})"
        rf"/\mathrm{{SD}}({label_1d})$"
    )
    ax.set_title("Relative parameter uncertainty vs leverage")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$N$ (survey size)")

    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    data_path = base_dir / "dataset" / "hermes_synthetic_data_0.2.0.csv"
    results_dir = base_dir / "results"
    plots_dir = results_dir / "plots"

    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) load data
    hermes = HermesData.from_csv(str(data_path))

    # 2) build surveys
    sampler = SurveySampler(hermes, rng_seed=42)
    N_grid = [10, 20, 40, 80, 100]
    surveys = sampler.sample_grid(N_grid, n_reps_per_combo=5)

    # 3) models
    lin1d = Model(draws=1000, tune=1000, target_accept=0.9)
    met   = MetModel(draws=1000, tune=1000, target_accept=0.9)

    models = {"lin1d": lin1d, "met": met}

    # 4) run both models on all surveys
    comp_df = run_models_on_surveys(surveys, models, seed=123)

    comp_csv = results_dir / "hermes_model_comparison.csv"
    comp_df.to_csv(comp_csv, index=False)
    print("Wrote model comparison table to:", comp_csv)

    # 5) WAIC comparison
    if "lin1d_waic" in comp_df.columns and "met_waic" in comp_df.columns:
        plot_waic_comparison(
            comp_df,
            model_key_1="lin1d",
            model_key_2="met",
            out_path=plots_dir / "waic_lin1d_vs_met.pdf",
        )

    # 6) example uncertainty ratio plots (adjust names if needed)
    if "lin1d_beta_sd" in comp_df.columns and "met_beta_m_sd" in comp_df.columns:
        plot_uncertainty_ratio_vs_L(
            comp_df,
            param_sd_1d="lin1d_beta_sd",
            param_sd_nd="met_beta_m_sd",
            out_path=plots_dir / "uncertainty_ratio_beta_mass.pdf",
            label_1d=r"\beta_{\mathrm{1D}}",
            label_nd=r"\beta_m",
        )

    if "lin1d_sigma_sd" in comp_df.columns and "met_sigma_p_sd" in comp_df.columns:
        plot_uncertainty_ratio_vs_L(
            comp_df,
            param_sd_1d="lin1d_sigma_sd",
            param_sd_nd="met_sigma_p_sd",
            out_path=plots_dir / "uncertainty_ratio_sigma.pdf",
            label_1d=r"\sigma_{\mathrm{1D}}",
            label_nd=r"\sigma_p",
        )


if __name__ == "__main__":
    main()
