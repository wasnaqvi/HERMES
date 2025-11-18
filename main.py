# main.py
from __future__ import annotations

from pathlib import Path

from src.data import HermesData
from src.Survey import SurveySampler
from src.Model import Model
from src.plots import make_leverage_panels_from_df,make_design_space_N_with_L_contours,make_fixedN_sigma_vs_L_scatter_from_df
from src.report import HermesReport


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    # paths
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
    N_grid = [10,20, 25, 30, 40, 50,60,65,70,75,80,90,100]
    surveys = sampler.sample_grid(N_grid, n_reps_per_combo=50)
    make_design_space_N_with_L_contours(
        surveys,
        out_path=plots_dir / "hermes_survey_design_space_N_Lcontours.pdf",
    )
    # 3) run the model on all surveys
    model = Model(draws=1000, tune=1000, target_accept=0.9)
    fit_df = model.run_on_surveys(surveys, seed=123)

    csv_path = results_dir / "hermes_massclass_fits.csv"
    fit_df.to_csv(csv_path, index=False)

    
    #per-N σ vs L panels
    make_fixedN_sigma_vs_L_scatter_from_df(
        fit_df,
        L_col="L_logM",
        out_path=plots_dir / "hermes_massclass_fixedN_sigma_vs_L.pdf",
    )
    rep = HermesReport(version="0.1", out_dir="results")
    rep.add_pngs_as_pdf(
    [
        "plots/hermes_massclass_L_alpha.pdf",
        "plots/hermes_massclass_L_beta.pdf",
        "plots/hermes_massclass_L_epsilon.pdf",
        "plots/hermes_massclass_N_alpha.pdf",
        "plots/hermes_massclass_N_beta.pdf",
        "plots/hermes_massclass_N_epsilon.pdf",
    ],
    out_name="HERMES_panels.pdf",
)

    # Optionally add per-survey fit PDF if it exists
    survey_pdf = survey_plot_dir / "hermes_survey_fits_10pp.pdf"
    if survey_pdf.exists():
        rep.add_pdf(survey_pdf)

    final_report = rep.build()
    print("Report written to:", final_report)


if __name__ == "__main__":
    main()
