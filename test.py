# main.py
from __future__ import annotations

from pathlib import Path

from src.data import HermesData
from src.Survey import SurveySampler
from src.Model import Model
from src.plots import make_leverage_panels_from_df,make_design_space_N_with_L_contours,make_design_space_N_vs_std,make_fixedN_sigma_vs_L_scatter_from_df
import pandas as pd

base_dir = Path(__file__).resolve().parent
data_path = base_dir / "dataset" / "hermes_synthetic_data.csv"
results_dir = base_dir / "results"
plots_dir = results_dir / "plots"
survey_plot_dir = results_dir / "hermes_survey_fit_plots"

    # make sure output dirs exist
results_dir.mkdir(exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)
survey_plot_dir.mkdir(parents=True, exist_ok=True)

    # 1) load data
hermes = HermesData.from_csv(str(data_path))

    # 2) build surveys
sampler = SurveySampler(hermes, rng_seed=42)
N_grid = [10, 20, 30, 40, 50,60,70]
surveys = sampler.sample_grid(N_grid, n_reps_per_combo=10)
make_design_space_N_with_L_contours(
        surveys,
        out_path=plots_dir / "Survey_design_space_N_Lcontours.pdf",
    )
make_design_space_N_vs_std(surveys,
    out_path=plots_dir / "Survey_design_space_N_std.pdf",
)

fit_df = pd.read_csv(results_dir / "hermes_massclass_fits.csv")
make_fixedN_sigma_vs_L_scatter_from_df(fit_df, L_col="L_logM")
