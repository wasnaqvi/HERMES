# HERMES: Hierarchical Exoplanet Regression and Survey Design

This repository explores how well different survey designs can recover
population–level trends between exoplanet mass and metallicity. It works with
a synthetic catalog (`hermes_synthetic_data.csv`), repeatedly draws mock
“surveys” from that catalog, fits Bayesian regression models, and then studies
how the **posterior uncertainties** scale with survey size and survey
**leverage**.

There are two main models:

- `Model`: a **single–response** mass–metallicity regression for planetary
  water abundance.
- `MetModel`: a **joint planetary–stellar metallicity** model that fits both
  planet and host–star metallicities simultaneously with a shared intrinsic
  covariance.

The key idea is to compare survey strategies at fixed sample size $N$ but
different leverage $L$, and to quantify how leverage improves constraints on
slopes, intercepts, and intrinsic scatter (and, in the joint model, the
planet–star covariance).

---

## 1. Data and survey construction

The starting point is a synthetic catalog wrapped by `HermesData`:

- `dataset/hermes_synthetic_data.csv` contains columns such as  
  - `logM` — planetary mass (log-scale),  
  - `log(X_H2O)` — planetary water abundance,  
  - `uncertainty_lower`, `uncertainty_upper` — asymmetric error bars on
    `log(X_H2O)`,  
  - `Star Metallicity` — host–star metallicity (e.g. [Fe/H]),  
  - `Star Metallicity Lower`, `Star Metallicity Upper` — corresponding
    uncertainties.

`SurveySampler` builds many mock surveys from this parent catalog:

- We choose a grid of target sample sizes $N$ (e.g. 10, 20, 30, …).
- For each $N$ and each mass–class label (S1–S4) we draw multiple surveys.
- Each survey has  
  - a fixed number of targets $N$,  
  - a class label (S1–S4, nested in mass),  
  - and an associated **leverage**
    $$
    L \equiv \sqrt{\sum_i (x_i - \bar x)^2}
    $$
    in mass space (here $x_i = \log M_i$).

The results for each fitted survey are collected into a `pandas.DataFrame`
and written out as CSV files (e.g. `results/hermes_massclass_fits.csv` and
`results/hermes_met_fits.csv`).

---

## 2. Single–response baseline model (`Model`)

For a given survey we observe, for each planet $i$:

- predictor: $x_i = \log M_i$,  
- response: $y_i = \log(X_{\mathrm{H_2O},i})$,  
- asymmetric measurement errors
  $(\sigma_{i,\mathrm{low}}, \sigma_{i,\mathrm{high}})$.

We convert the asymmetric error bars into an effective Gaussian measurement
scatter
$$
\sigma_{i,\mathrm{meas}} = \tfrac12\left(
  \lvert \sigma_{i,\mathrm{low}} \rvert +
  \lvert \sigma_{i,\mathrm{high}} \rvert
\right),
$$
and center the predictor
$$
x_i^{\mathrm{c}} = x_i - \bar x.
$$

### 2.1 PyMC model

The baseline PyMC function `_fit_leverage_survey` implements

- **Priors**
  $$
  \alpha \sim \mathcal{N}\big(\overline{y}, s_y^2 / n\big), \qquad
  \beta  \sim \mathcal{N}\big(0, s_y / s_x\big), \qquad
  \epsilon \sim \mathrm{HalfNormal}(s_y),
  $$
  where $\overline{y}$ and $s_y$ are the sample mean and standard deviation
  of the responses $y_i$, and $s_x$ is the spread of the centered predictor
  $x_i^{\mathrm{c}}$.

- **Likelihood with intrinsic scatter**
  $$
  y_i \sim \mathcal{N}\!\left(
      \alpha + \beta x_i^{\mathrm{c}},
      \sqrt{\sigma_{i,\mathrm{meas}}^2 + \epsilon^2}
  \right).
  $$

Here:

- $\alpha$ is the intercept,  
- $\beta$ is the slope of planetary metallicity vs. mass,  
- $\epsilon$ is an intrinsic scatter term that is added in quadrature with
  the measurement noise.

For each survey we draw posterior samples of $(\alpha, \beta, \epsilon)$
using NUTS (`pm.sample`) and summarize them with ArviZ (`az.summary`), storing
for each parameter its mean, posterior standard deviation, and 68% highest
density interval (HDI).

These summaries are used to produce plots of

- $\sigma_\alpha$ vs. leverage $L$,  
- $\sigma_\beta$ vs. leverage $L$,  
- $\sigma_\epsilon$ vs. leverage $L$,

at fixed $N$. This reproduces the “fixed–$N$ survey performance” plots
for the 1D model.

---

## 3. Joint metallicity model (`MetModel`)

`MetModel` augments the baseline model to simultaneously fit **planetary** and
**stellar** metallicities, sharing an intrinsic 2D covariance structure.

For each planet $i$ in a survey we now have:

- predictor: $x_i = \log M_i$,  
- planetary response: $y_{p,i} = \log(X_{\mathrm{H_2O},i})$,  
- stellar response: $y_{s,i} = \mathrm{[Fe/H]}_{\star,i}$,  
- asymmetric measurement errors
  $(\sigma_{i,\mathrm{low}}^{p}, \sigma_{i,\mathrm{high}}^{p})$ and
  $(\sigma_{i,\mathrm{low}}^{s}, \sigma_{i,\mathrm{high}}^{s})$.

We again define effective measurement scatters
$$
\sigma^{p}_{i,\mathrm{meas}} = \tfrac12\left(
  \lvert \sigma^{p}_{i,\mathrm{low}} \rvert +
  \lvert \sigma^{p}_{i,\mathrm{high}} \rvert
\right), \qquad
\sigma^{s}_{i,\mathrm{meas}} = \tfrac12\left(
  \lvert \sigma^{s}_{i,\mathrm{low}} \rvert +
  \lvert \sigma^{s}_{i,\mathrm{high}} \rvert
\right),
$$
and centered predictor
$$
x_i^{\mathrm{c}} = x_i - \bar x.
$$

### 3.1 Regression planes

We define two parallel regression relations for the **latent** (noise–free)
metallicities:

- Planet regression  
  $$
  \mu_{p,i} = \alpha_p + \beta_p x_i^{\mathrm{c}}.
  $$

- Stellar regression  
  $$
  \mu_{s,i} = \alpha_s + \beta_s x_i^{\mathrm{c}}.
  $$

The pair of slopes $(\beta_p, \beta_s)$ and intercepts
$(\alpha_p, \alpha_s)$ define a “plane” in the 3D space
$(\log M, \log X_{\mathrm{H_2O}}, \mathrm{[Fe/H]}_\star)$.

### 3.2 Intrinsic 2×2 covariance via LKJ

Instead of treating the two regressions as independent, we introduce a shared
intrinsic covariance matrix
$$
\Sigma =
\begin{pmatrix}
  \sigma_p^2 & \rho\,\sigma_p\sigma_s \\
  \rho\,\sigma_p\sigma_s & \sigma_s^2
\end{pmatrix},
$$
with:

- $\sigma_p$ = intrinsic scatter of planetary metallicity at fixed mass,  
- $\sigma_s$ = intrinsic scatter of stellar metallicity at fixed mass,  
- $\rho$ = intrinsic correlation between the two metallicities.

In code, this covariance is parameterized using the LKJ prior

- `pm.LKJCholeskyCov("chol_cov", n=2, eta=2.0, sd_dist=HalfNormal(0.5))`

which returns a Cholesky factor of $\Sigma$, the correlation matrix, and
the vector of marginal standard deviations $(\sigma_p, \sigma_s)$.

### 3.3 Latent metallicities and observed data

Given the regression means and intrinsic covariance, we introduce latent
intrinsic metallicities
$$
\begin{pmatrix}
  z_{p,i} \\
  z_{s,i}
\end{pmatrix}
\sim
\mathcal{N}_2\!\left(
  \begin{pmatrix}
    \mu_{p,i} \\
    \mu_{s,i}
  \end{pmatrix},
  \Sigma
\right).
$$

The observed values are then modeled as
$$
y_{p,i} \sim \mathcal{N}\big(z_{p,i}, \sigma^{p}_{i,\mathrm{meas}}\big),
\qquad
y_{s,i} \sim \mathcal{N}\big(z_{s,i}, \sigma^{s}_{i,\mathrm{meas}}\big).
$$

This yields a coherent generative story:

1. Given mass, draw a pair of intrinsic metallicities from the bivariate normal
   with means on the regression plane and covariance $\Sigma$.  
2. Then add measurement noise independently to each component to produce the
   observed catalog values.

### 3.4 What `MetModel` does per survey

`MetModel.fit_survey` takes one `Survey` object, extracts the relevant columns
from `survey.df`, and calls `_fit_met_survey` (the PyMC model above). For each
survey, we sample the joint posterior of

- regression parameters: $\alpha_p, \beta_p, \alpha_s, \beta_s$,  
- intrinsic scatters: $\sigma_p, \sigma_s$,  
- intrinsic correlation: $\rho$.

We then summarize these parameters (means, standard deviations, HDIs) and
record them in a metallicity results table, e.g.

- `beta_p_mean`, `beta_p_sd`,  
- `beta_s_mean`, `beta_s_sd`,  
- `sigma_p_mean`, `sigma_s_mean`,  
- `rho_mean`, etc.

This table is written to `results/hermes_met_fits.csv` for downstream
analysis and plotting.

---

## 4. Visualization and survey–design diagnostics

Several plotting utilities in `src/plots.py` use the summary DataFrames to
visualize survey performance:

- **Design space:** `make_design_space_N_with_L_contours` shows each mock
  survey as a point in the $(N, \mathrm{std}(\log M))$ plane, coloured by mass class
  label (S1–S4), with overlaid curves of constant leverage
  $L \approx \sqrt{N}\,\mathrm{std}(\log M)$.

- **Fixed–N uncertainty vs. leverage:** functions like
  `make_fixedN_sigma_vs_L_scatter_from_df` build 3–panel figures where, for a
  fixed $N$, we plot $\sigma_\alpha$, $\sigma_\beta$,
  and $\sigma_\epsilon$ vs. $L$, coloured by class label. Each panel
  overlays:
  - a power–law fit $\sigma \approx a L^b$ in log–log space with a
    $1\sigma$ band,  
  - a linear fit in the original $\sigma$–$L$ space.

- **MetModel–specific extensions** (not all shown here) can similarly plot
  how posterior uncertainties in $\beta_p$, $\beta_s$,
  $\sigma_p$, $\sigma_s$, and $\rho$ scale with leverage at fixed
  $N$, and can reconstruct a “slope plane” in the space
  $(L, \beta_p, \beta_s)$.

Together, these diagnostics answer questions like:

- At fixed survey size $N$, how much leverage do we need to constrain the
  mass–metallicity slope to a given precision?  
- How do different mass–class selections (S1–S4) trade off sample size vs.
  leverage?  
- In the joint metallicity model, how well can we constrain the **relationship
  between planetary and stellar metallicity** as we vary survey design?

---

## 5. Running the code

A minimal workflow is:

1. Ensure the synthetic catalog exists at
   `dataset/hermes_synthetic_data.csv`.  
2. From the repository root, run:

   ```bash
   python main.py
