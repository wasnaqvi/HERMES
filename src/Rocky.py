# src/Rocky.py
"""Bayesian CMF regression models for rocky exoplanets (HERMES4Rocky)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist

from .Model import (
    _run_nuts,
    ModelConfig,
    _scalar_stats_from_idata,
    _et68,
    _as_1d_float,
    _finite_mask,
    _safe_sd,
)

# ---------------------------------------------------------------------------
# NumPyro models
# ---------------------------------------------------------------------------

def _rocky_pooled_model(
    *,
    star_cmf_obs: jax.Array,       # (N,) observed star CMF
    star_cmf_err: jax.Array,       # (N,) star CMF uncertainty
    planet_cmf_obs: jax.Array,     # (N,) observed planet CMF (or None for prior predictive)
    planet_cmf_err: jax.Array,     # (N,) planet CMF uncertainty
    alpha_mu: float,
    alpha_sigma: float,
    beta_mu: float,                # 1.0 (expect 1:1)
    beta_sigma: float,
    epsilon_sigma: float,
) -> None:
    """Complete-pooling measurement-error regression: Planet_CMF = alpha + beta * Star_CMF + eps."""
    # Latent true stellar CMF (measurement error model)
    star_cmf_true = numpyro.sample("star_cmf_true", dist.Normal(star_cmf_obs, star_cmf_err))

    # Regression parameters
    alpha   = numpyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
    beta    = numpyro.sample("beta", dist.Normal(beta_mu, beta_sigma))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_sigma))

    # Science equation
    mu = alpha + beta * star_cmf_true
    obs_sigma = jnp.sqrt(planet_cmf_err**2 + epsilon**2)
    numpyro.sample("planet_cmf", dist.Normal(mu, obs_sigma), obs=planet_cmf_obs)


def _rocky_partial_pooling_model(
    *,
    star_cmf_obs: jax.Array,       # (N_obs,)
    star_cmf_err: jax.Array,       # (N_obs,)
    planet_cmf_obs: jax.Array,     # (N_obs,)
    planet_cmf_err: jax.Array,     # (N_obs,)
    planet_idx: jax.Array,         # (N_obs,) int index -> unique planet
    n_planets: int,
    mu_star: float,                # population mean of stellar CMFs
    sigma_star: float,             # population sd of stellar CMFs
    alpha_mu: float,
    alpha_sigma: float,
    beta_mu: float,
    beta_sigma: float,
    epsilon_sigma: float,
) -> None:
    """
    Hierarchical partial-pooling model (following NumPyro BHM pattern).

    Structure:
      Population level:  alpha, beta, epsilon (single regression for all planets)
      Planet level:      star_cmf_true_j, planet_cmf_true_j (latent per planet)
      Observation level: star_cmf_obs_i, planet_cmf_obs_i (noisy per survey)

    Planets observed by multiple surveys share latent true values,
    enabling shrinkage toward the population regression.
    """
    # --- Population-level regression parameters ---
    alpha   = numpyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
    beta    = numpyro.sample("beta", dist.Normal(beta_mu, beta_sigma))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_sigma))

    # --- Planet-level latent true values ---
    with numpyro.plate("planets", n_planets):
        # Latent true stellar CMF per unique planet
        star_cmf_true = numpyro.sample(
            "star_cmf_true",
            dist.Normal(mu_star, sigma_star),
        )
        # Latent true planet CMF: drawn from regression + intrinsic scatter
        # THIS is the key hierarchical step — planet_cmf_true is a latent
        # random variable, not deterministic. Intrinsic scatter (epsilon)
        # is separated from measurement error (planet_cmf_err).
        planet_cmf_true = numpyro.sample(
            "planet_cmf_true",
            dist.Normal(alpha + beta * star_cmf_true, epsilon),
        )

    # --- Observation-level likelihoods ---
    with numpyro.plate("obs", star_cmf_obs.shape[0]):
        # Each survey's stellar CMF measurement
        numpyro.sample(
            "star_cmf_obs_ll",
            dist.Normal(star_cmf_true[planet_idx], star_cmf_err),
            obs=star_cmf_obs,
        )
        # Each survey's planet CMF measurement (pure measurement error)
        numpyro.sample(
            "planet_cmf",
            dist.Normal(planet_cmf_true[planet_idx], planet_cmf_err),
            obs=planet_cmf_obs,
        )


# ---------------------------------------------------------------------------
# Helper: compute data-scaled priors
# ---------------------------------------------------------------------------

def _compute_priors(planet_cmf: np.ndarray, n: int) -> Dict[str, float]:
    """Data-scaled prior hyperparameters following HERMES conventions."""
    y_mean = float(np.mean(planet_cmf))
    y_sd = _safe_sd(planet_cmf, fallback=0.1)
    return {
        "alpha_mu": y_mean,
        "alpha_sigma": max(y_sd / np.sqrt(n), 1e-3),
        "beta_mu": 1.0,           # expect 1:1 star-planet CMF (like MetModel beta_s)
        "beta_sigma": 1.0,
        "epsilon_sigma": max(y_sd, 1e-3),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class RockyModel:
    """Bayesian CMF regression for rocky exoplanets."""

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        self.cfg = cfg

    # ---- Complete pooling ----

    def fit_pooled(self, pooled_df: pd.DataFrame, seed: int = 42) -> az.InferenceData:
        """Fit complete-pooling model. Returns ArviZ InferenceData."""
        star = pooled_df["star_cmf"].to_numpy(dtype=np.float64)
        planet = pooled_df["planet_cmf"].to_numpy(dtype=np.float64)
        star_err = pooled_df["star_cmf_err"].to_numpy(dtype=np.float64)
        planet_err = pooled_df["planet_cmf_err"].to_numpy(dtype=np.float64)

        priors = _compute_priors(planet, len(planet))
        dtype = self.cfg.jax_dtype

        model_kwargs: Dict[str, Any] = dict(
            star_cmf_obs=jnp.asarray(star, dtype=dtype),
            star_cmf_err=jnp.asarray(star_err, dtype=dtype),
            planet_cmf_obs=jnp.asarray(planet, dtype=dtype),
            planet_cmf_err=jnp.asarray(planet_err, dtype=dtype),
            **priors,
        )

        rng_key = jax.random.PRNGKey(seed)
        mcmc, ll = _run_nuts(
            _rocky_pooled_model,
            rng_key,
            draws=self.cfg.draws,
            tune=self.cfg.tune,
            target_accept=self.cfg.target_accept,
            num_chains=self.cfg.num_chains,
            model_kwargs=model_kwargs,
            compute_log_lik=self.cfg.compute_log_lik,
            chain_method=self.cfg.chain_method,
        )
        return az.from_numpyro(mcmc, log_likelihood=ll) if ll is not None else az.from_numpyro(mcmc)

    # ---- Partial pooling ----

    def fit_partial_pooling(self, pp_data: dict, seed: int = 42) -> az.InferenceData:
        """Fit hierarchical partial-pooling model. Returns ArviZ InferenceData."""
        planet_cmf = pp_data["planet_cmf_obs"]
        star_cmf = pp_data["star_cmf_obs"]
        planet_idx = pp_data["planet_idx"]
        n_planets = pp_data["n_planets"]

        priors = _compute_priors(planet_cmf, len(planet_cmf))
        dtype = self.cfg.jax_dtype

        # Population-level star CMF prior (weakly informative from data)
        mu_star = float(np.mean(star_cmf))
        sigma_star = max(_safe_sd(star_cmf, fallback=0.1), 0.01)

        model_kwargs: Dict[str, Any] = dict(
            star_cmf_obs=jnp.asarray(star_cmf, dtype=dtype),
            star_cmf_err=jnp.asarray(pp_data["star_cmf_err"], dtype=dtype),
            planet_cmf_obs=jnp.asarray(planet_cmf, dtype=dtype),
            planet_cmf_err=jnp.asarray(pp_data["planet_cmf_err"], dtype=dtype),
            planet_idx=jnp.asarray(planet_idx, dtype=jnp.int32),
            n_planets=n_planets,
            mu_star=mu_star,
            sigma_star=sigma_star,
            **priors,
        )

        rng_key = jax.random.PRNGKey(seed)
        mcmc, ll = _run_nuts(
            _rocky_partial_pooling_model,
            rng_key,
            draws=self.cfg.draws,
            tune=self.cfg.tune,
            target_accept=self.cfg.target_accept,
            num_chains=self.cfg.num_chains,
            model_kwargs=model_kwargs,
            compute_log_lik=self.cfg.compute_log_lik,
            chain_method=self.cfg.chain_method,
        )
        return az.from_numpyro(mcmc, log_likelihood=ll) if ll is not None else az.from_numpyro(mcmc)

    # ---- Summary ----

    def summarize(
        self,
        idata: az.InferenceData,
        params: Sequence[str] = ("alpha", "beta", "epsilon"),
    ) -> pd.DataFrame:
        """Posterior summary for scalar parameters."""
        rows = []
        for p in params:
            stats = _scalar_stats_from_idata(idata, p)
            rows.append({"param": p, **stats})
        return pd.DataFrame(rows).set_index("param")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SURVEY_COLORS = {
    "Behmard": "#1f77b4",
    "Brinkman": "#ff7f0e",
    "Ross": "#2ca02c",
}


def plot_rocky_cmf(
    pooled_df: pd.DataFrame,
    idata: az.InferenceData,
    *,
    title: str = "",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Planet CMF vs Star CMF scatter colored by survey,
    with posterior regression line and 68% credible band.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Scatter with error bars by survey
    for survey, color in SURVEY_COLORS.items():
        sub = pooled_df[pooled_df["survey"] == survey]
        if sub.empty:
            continue
        ax.errorbar(
            sub["star_cmf"], sub["planet_cmf"],
            xerr=sub["star_cmf_err"], yerr=sub["planet_cmf_err"],
            fmt="o", color=color, label=survey,
            ms=5, alpha=0.8, ecolor=color, elinewidth=0.8, capsize=2,
            markeredgecolor="k", markeredgewidth=0.4,
        )

    # Posterior regression line
    alpha_samples = np.asarray(idata.posterior["alpha"]).ravel()
    beta_samples = np.asarray(idata.posterior["beta"]).ravel()

    x_grid = np.linspace(pooled_df["star_cmf"].min() - 0.02, pooled_df["star_cmf"].max() + 0.02, 100)
    y_lines = alpha_samples[:, None] + beta_samples[:, None] * x_grid[None, :]

    y_mean = y_lines.mean(axis=0)
    y_lo, y_hi = np.quantile(y_lines, [0.16, 0.84], axis=0)

    ax.plot(x_grid, y_mean, "k-", lw=1.5, label="Posterior mean")
    ax.fill_between(x_grid, y_lo, y_hi, color="gray", alpha=0.25, label="68% CI")

    # 1:1 reference line
    ax.plot(x_grid, x_grid, "k--", lw=0.8, alpha=0.4, label="1:1")

    ax.set_xlabel("Star CMF")
    ax.set_ylabel("Planet CMF")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax
