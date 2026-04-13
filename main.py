"""
HERMES main pipeline — mirrors HERMES_Extended_MetModel.ipynb.

Outputs written to results/ and results/plots/.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import jax
import pandas as pd
import numpyro

from src.Model import _met_model, _met_model_no_stellar
from src.Survey import SurveySampler
from src.data import HermesData
from src.scatter_threshold import ScatterThresholdConfig
from src.utils import (
    add_zscores,
    apply_style,
    fit_oracle,
    plot_best_survey_fits,
    plot_epsilon_vs_L,
    plot_model_comparison,
    plot_multiseed,
    plot_sigma_vs_N,
    plot_sigma_vs_leverage,
    plot_zscores,
    print_top5,
    print_waic,
    print_zscore_summary,
    run_experiments,
    run_scatter_threshold_pipeline,
)

warnings.filterwarnings("ignore")
numpyro.set_platform("cpu")

DATA_PATH = Path("dataset/hermes_synthetic_data_0.6.0.csv")
RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

MCMC_SEEDS = [321, 42, 7]
SURVEY_SEED = 42
N_GRID = [50, 80, 150, 200, 250, 350, 400, 500, 600]
N_REPS = 5
DRAWS = 800
TUNE = 800
TARGET_ACCEPT = 0.90
NUM_CHAINS = 4
COMPUTE_WAIC = True
PRIMARY = "3D Model"
RUN_SCATTER_THRESHOLD = True

SCATTER_CFG = ScatterThresholdConfig(
    scatter_grid=(0.1, 0.2, 0.3, 0.4, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0),
    n_grid=(80, 150, 250, 400, 600),
    n_reps=3,
    mcmc_seeds=(321, 42),
    draws=600,
    tune=600,
    target_accept=TARGET_ACCEPT,
    num_chains=NUM_CHAINS,
    survey_seed=SURVEY_SEED,
    beta_s_true=1.0,
    fixed_noise_seed=42,
    threshold_fraction=0.80,
)

MODEL_VARIANTS = [
    (_met_model, "3D Model"),
    (_met_model_no_stellar, "2D Model"),
]

CLS_ORD = ["S1", "S2", "S3", "S4"]
CLS_CLR = {"S1": "C0", "S2": "C1", "S3": "C2", "S4": "C3"}


def main() -> None:
    print("JAX devices:", jax.devices())
    apply_style()
    RESULTS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    hermes = HermesData.from_csv(str(DATA_PATH))
    hermes_df = hermes.df
    print(f"Loaded {len(hermes_df)} rows from {DATA_PATH}")

    sampler = SurveySampler(hermes, rng_seed=SURVEY_SEED)
    surveys = sampler.sample_grid(N_GRID, n_reps_per_combo=N_REPS)
    print(f"Built {len(surveys)} surveys")

    csv_path = RESULTS_DIR / "hermes_src_results.csv"
    if csv_path.exists():
        print(f"Loading cached results from {csv_path}")
        df_results = pd.read_csv(csv_path)
    else:
        df_results = run_experiments(
            surveys,
            RESULTS_DIR,
            draws=DRAWS,
            tune=TUNE,
            target_accept=TARGET_ACCEPT,
            num_chains=NUM_CHAINS,
            compute_waic=COMPUTE_WAIC,
            mcmc_seeds=MCMC_SEEDS,
            model_variants=MODEL_VARIANTS,
        )

    ref = fit_oracle(
        hermes_df,
        draws=DRAWS,
        tune=TUNE,
        target_accept=TARGET_ACCEPT,
        num_chains=NUM_CHAINS,
    )
    df_results = add_zscores(df_results, ref)
    df_results.to_csv(csv_path, index=False)

    df_primary = df_results[
        (df_results["model"] == PRIMARY) & (df_results["seed"] == MCMC_SEEDS[0])
    ].copy()
    z_params = [
        p for p in ["alpha_p", "beta_p", "beta_s", "epsilon"] if f"z_{p}" in df_primary.columns
    ]

    print(f"\nPrimary model: {PRIMARY}")
    sd_cols = [
        c
        for c in ["alpha_p_sd", "beta_p_sd", "beta_s_sd", "epsilon_sd"]
        if c in df_primary.columns
    ]
    print("\nMean posterior SD by class:")
    print(df_primary.groupby("class_label")[sd_cols].mean().round(4).to_string())

    print_zscore_summary(df_primary, z_params)
    print_top5(df_primary, z_params)
    print_waic(df_results)

    print("\nGenerating plots ...")
    plot_sigma_vs_leverage(df_primary, PLOTS_DIR, cls_ord=CLS_ORD)
    plot_sigma_vs_N(df_primary, PLOTS_DIR, cls_ord=CLS_ORD, cls_clr=CLS_CLR)
    plot_zscores(df_primary, z_params, PLOTS_DIR, cls_ord=CLS_ORD, cls_clr=CLS_CLR)
    plot_multiseed(
        df_results,
        PLOTS_DIR,
        primary=PRIMARY,
        mcmc_seeds=MCMC_SEEDS,
        cls_ord=CLS_ORD,
        cls_clr=CLS_CLR,
    )
    plot_model_comparison(df_results, PLOTS_DIR, seed0=MCMC_SEEDS[0])
    plot_epsilon_vs_L(df_primary, PLOTS_DIR)

    if "z_beta_p" in df_results.columns:
        plot_best_survey_fits(
            df_results,
            surveys,
            PLOTS_DIR,
            primary=PRIMARY,
            mcmc_seed0=MCMC_SEEDS[0],
            draws=DRAWS,
            tune=TUNE,
            target_accept=TARGET_ACCEPT,
            num_chains=NUM_CHAINS,
            cls_clr=CLS_CLR,
        )

    if RUN_SCATTER_THRESHOLD:
        run_scatter_threshold_pipeline(
            hermes_df,
            RESULTS_DIR,
            PLOTS_DIR,
            scatter_cfg=SCATTER_CFG,
        )

    print(f"\nDone. Plots → {PLOTS_DIR}")


if __name__ == "__main__":
    main()
