# main.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.data import HermesData
from src.Survey import SurveySampler
from src.Model import MetModel
from src.plots import (
    make_design_space_N_with_L_contours,
    make_met_fixedN_uncertainty_vs_L_from_df,
    make_met_global_slope_3d_from_df,
    plot_mass_histogram_nested_classes,
     make_met_fixedN_scatter_mean_vs_L_from_df
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # paths
    data_path = base_dir / "dataset" / "hermes_synthetic_data_0.2.0.csv"
    results_dir = base_dir / "results"
    plots_dir = results_dir / "plots_bivariate"
    results_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    hermes = HermesData.from_csv(str(data_path))

    # 2) build surveys (same design as before)
    sampler = SurveySampler(hermes, rng_seed=42)

    N_grid = [30, 40, 50, 90]
    surveys = sampler.sample_grid(N_grid, n_reps_per_combo=5)
    
    make_design_space_N_with_L_contours(
        surveys,
        out_path=plots_dir / "hermes_met_design_space_N_Lcontours.pdf",
    )
    met_model = MetModel(draws=2000, tune=1000, target_accept=0.9)
    met_fit_df = met_model.run_on_surveys(surveys, seed=321)

    met_csv_path = results_dir / "hermes_met_bivariate.csv"
    met_fit_df.to_csv(met_csv_path, index=False)
    print("Wrote metallicity survey summaries to:", met_csv_path)
    
    met_fit_df = pd.read_csv(results_dir / "hermes_met_bivariate.csv")
    # 4) Visualizations: fixed-N leverage panels + 3D slope view
    make_met_fixedN_uncertainty_vs_L_from_df(
        met_fit_df,
        out_dir=plots_dir,
        L_col="L_logM",
    )

    make_met_fixedN_scatter_mean_vs_L_from_df(
    met_fit_df,
    out_dir=plots_dir,
    L_col="L_logM",
)

if __name__ == "__main__":
    main()
