# HERMES: HiERarchical Modelling for Exoplanet Science

HERMES, or **HiERarchical Modelling for Exoplanet Science**, studies how exoplanet population trends can be recovered from finite, noisy atmospheric samples. The code in this repository builds mock survey draws, fits hierarchical Bayesian models, and asks how survey size, leverage, measurement uncertainty, and intrinsic scatter affect the recovery of population-level relations. The paper gives the full motivation, model interpretation, and scientific conclusions; this repository gives the computational path used to run the analysis.

The main HERMES pipeline works with synthetic HERMES-style catalogs, repeatedly samples survey subsets, fits NumPyro models with JAX-backed NUTS, and writes posterior summaries, model-comparison quantities, and diagnostic figures. The emphasis is on reproducible analysis rather than a packaged library. In practice, the repository lets a reader inspect how the catalog is sampled, how the Bayesian fits are launched, how posterior summaries are cached, and how the diagnostic figures are generated.

## Scientific Scope

HERMES treats individual planets as noisy draws from an underlying population relation. In the main atmospheric-metallicity workflow, the sampled surveys probe how well the model recovers mass-dependent structure in planetary and stellar quantities when the available targets span different regions of parameter space. The code tracks this through survey classes, sample size, leverage, posterior uncertainty, z-score behavior, WAIC comparisons, and scatter-threshold experiments.

The central design question is not only whether a regression can be fit, but whether a finite survey has enough information to recover the population trend with useful precision. HERMES therefore treats sample size and leverage as separate axes of survey quality. A survey with many targets can still be weak if the targets occupy a narrow region of predictor space, while a smaller but better distributed survey can constrain a trend more efficiently. The code quantifies this behavior through repeated draws from the parent catalog and posterior summaries across survey classes.

The atmospheric workflow also keeps measurement uncertainty and intrinsic scatter distinct. Observed catalog values enter the likelihood with their measurement errors, while the model estimates additional population scatter where appropriate. This separation matters because an apparent broad population relation can reflect noisy observations, genuine astrophysical dispersion, or both. HERMES uses the hierarchical Bayesian setup to propagate those uncertainties into the inferred population-level quantities.

The repository also includes `HERMES4Rocky.py`, a related rocky-exoplanet extension that fits complete-pooling and partial-pooling core-mass-fraction regressions across survey measurements. That script uses the same basic NumPyro/JAX machinery but applies it to a different compositional data product.

The README intentionally does not reproduce the full mathematical specification or the interpretation of the recovered parameters. Those belong in the manuscript. The goal here is to make the code navigable and to show enough of the scientific structure that the scripts are understandable.

## Inference Stack

HERMES uses `numpyro` for the probabilistic models and NUTS sampling, with `jax` and `jax.numpy` providing the numerical backend. `arviz` stores and summarizes posterior samples. `pandas`, `numpy`, and `matplotlib` handle catalog manipulation, numerical preprocessing, and figures. The default scripts set NumPyro to CPU execution; accelerator runs depend on the local JAX installation.

NumPyro is used because the project repeatedly fits many related Bayesian models across mock survey realizations. JAX makes those repeated NUTS runs practical and keeps the model code close to array-level scientific Python. ArviZ provides the bridge from sampler output to posterior summaries, intervals, and model-comparison quantities.

Install the core stack in your preferred environment with:

```bash
pip install numpy pandas matplotlib arviz jax numpyro
```

There is currently no pinned environment file, so exact dependency versions should be recorded separately for archival runs.

## Repository Layout

```text
.
|-- main.py                  # Main HERMES synthetic-survey pipeline
|-- HERMES4Rocky.py          # Rocky-exoplanet CMF regression extension
|-- comparison.py            # Additional comparison script
|-- metric.py, metric_2.py   # Metric experiments
|-- test.py, test_2.py       # Local checks and exploratory runs
|-- src/                     # Model, data, survey, plotting, and utility code
|-- dataset/                 # Catalogs, notebooks, generated tables, and figures
`-- results/                 # Pipeline outputs
```

The top-level scripts are the entry points. The reusable machinery lives in `src/`.

## The `src/` Structure

`src/data.py` defines the `HermesData` wrapper used to load and carry catalog tables. `src/Survey.py` defines survey subsets, target tracking, survey classes, leverage calculations, and survey sampling. `src/Model.py` contains the main NumPyro models, the shared NUTS runner, model configuration, posterior summaries, and model variants. `src/utils.py` coordinates repeated fits, cached result tables, z-score summaries, WAIC reporting, and calls into the plotting layer. `src/plots.py` holds the figure routines. `src/scatter_threshold.py` runs the synthetic scatter-threshold experiments. `src/Rocky.py` and `src/RockyUtils.py` support the rocky-exoplanet CMF extension.

The structure is deliberately simple. `main.py` defines the run configuration, `Survey.py` builds the survey objects, `Model.py` owns the probabilistic models and sampler calls, and `utils.py` handles the repetitive analysis work around those fits. This keeps the scientific model code separate from plotting and bookkeeping, while still leaving the pipeline easy to follow from the top-level script.

The central path is:

```text
main.py -> src/data.py -> src/Survey.py -> src/Model.py -> src/utils.py/src/plots.py
```

## Dataset And Notebooks

`dataset/` contains the analysis inputs and companion material. The main scripted pipeline reads `dataset/hermes_synthetic_data_0.7.0.csv` by default. The folder also contains Ariel/planet catalog snapshots, generated CSV summaries, previously generated figures, and notebooks used while developing the data-cleaning and model-analysis workflow.

The notebooks are useful for context but are not the cleanest reproducibility target. `dataset/dataclean.ipynb` records catalog preparation work, including the practical filtering and table-building steps that led to the HERMES-style inputs. `dataset/dataviz.ipynb` contains exploratory visualization and sanity checks. `dataset/HERMES_Extended_MetModel.ipynb` tracks the extended metallicity-model development path and is useful for seeing how the scripted workflow relates to the notebook analysis. Use `main.py` when you want the scripted HERMES run.

Several generated figures and CSV files also live in `dataset/`. They are retained as analysis artifacts, but new scripted outputs are written to `results/` and `results/plots/`.

## Running HERMES

From the repository root:

```bash
python main.py
```

By default, the script reads:

```text
dataset/hermes_synthetic_data_0.7.0.csv
```

and writes:

```text
results/
results/plots/
```

The run builds survey samples, fits the configured model variants, computes posterior summaries, writes result tables, and generates the diagnostic plots. If `results/hermes_src_results_qstellar.csv` already exists, `main.py` loads that cached table instead of refitting the full grid.

The default configuration uses repeated survey draws, multiple MCMC seeds, multiple chains, WAIC computation, and scatter-threshold experiments. A fresh run can therefore take time. For a quick smoke test, reduce `DRAWS`, `TUNE`, `NUM_CHAINS`, `N_REPS`, or set `RUN_SCATTER_THRESHOLD = False` in `main.py`.

The main configuration block in `main.py` controls the survey grid, number of repeated draws, MCMC seeds, sampler settings, injected synthetic relation, and whether the scatter-threshold experiment runs. The defaults are chosen for analysis rather than speed. If you change the configuration, treat existing cached CSV files with care, since a cached result table may no longer correspond to the settings you intend to inspect.

## Running HERMES4Rocky

Run the rocky-exoplanet extension with:

```bash
python HERMES4Rocky.py
```

The script loads the available rocky-exoplanet survey data, builds pooled and partial-pooling datasets, fits both NumPyro models, prints posterior summaries, and writes:

```text
results/rocky_cmf_regression.png
```

You can restrict the surveys:

```bash
python HERMES4Rocky.py --surveys Behmard Brinkman Ross
```

or change the sampler settings:

```bash
python HERMES4Rocky.py --draws 4000 --tune 2000 --target-accept 0.9
```

## What To Expect

A normal run produces CSV result tables in `results/`, terminal summaries of posterior quantities, optional WAIC/model-comparison output, and figures in `results/plots/`. The exact output set depends on the cached files already present and on the switches enabled in the top-level scripts.

The printed output is meant to give a quick read on the active run: available JAX devices, catalog size, number of survey draws, posterior uncertainty summaries, z-score summaries, and model-comparison diagnostics when enabled. The figures then show how uncertainty and recovery behavior vary across sample size, leverage, survey class, and model choice.

This README intentionally stops short of reproducing the paper. The code shows how the HERMES calculations are organized and run; the manuscript gives the scientific argument.
