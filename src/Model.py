# src/Model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood

from .Survey import Survey  # relative import


Array1D = npt.NDArray[np.floating]


def _as_1d_float(x: npt.ArrayLike) -> Array1D:
    a = np.asarray(x, dtype=float).ravel()
    return a


def _finite_mask(*arrays: Array1D) -> npt.NDArray[np.bool_]:
    m = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        m &= np.isfinite(a)
    return m


def _safe_ptp(x: Array1D, fallback: float = 1.0) -> float:
    span = float(np.ptp(x))
    return span if np.isfinite(span) and span > 0.0 else float(fallback)


def _safe_sd(x: Array1D, fallback: float = 1.0) -> float:
    if x.size <= 1:
        return float(fallback)
    sd = float(np.std(x, ddof=1))
    return sd if np.isfinite(sd) and sd > 0.0 else float(fallback)


def _run_nuts(
    model_fn,
    rng_key: jax.Array,
    *,
    draws: int,
    tune: int,
    target_accept: float,
    num_chains: int,
    model_kwargs: Mapping[str, Any],
) -> Tuple[MCMC, Dict[str, jax.Array]]:
    """
    Runs NUTS and returns (mcmc, log_lik_dict).
    log_lik_dict is suitable for az.from_numpyro(..., log_likelihood=...).
    """
    kernel = NUTS(model_fn, target_accept_prob=float(target_accept))
    mcmc = MCMC(
        kernel,
        num_warmup=int(tune),
        num_samples=int(draws),
        num_chains=int(num_chains),
        progress_bar=False,
    )
    mcmc.run(rng_key, **model_kwargs)

    # group_by_chain=True yields shape (chains, draws, ...) which ArviZ likes
    posterior = mcmc.get_samples(group_by_chain=True)
    ll = log_likelihood(model_fn, posterior, **model_kwargs)

    return mcmc, ll


# ----------------------------
# 1) Linear + intrinsic scatter
# ----------------------------
def _linear_scatter_model(
    *,
    x_c: jax.Array,          # (n,)
    meas_sigma: jax.Array,   # (n,)
    y_obs: Optional[jax.Array] = None,  # (n,)
    alpha_mu: float,
    alpha_sigma: float,
    beta_sigma: float,
    epsilon_sigma: float,
) -> None:
    alpha = numpyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
    beta = numpyro.sample("beta", dist.Normal(0.0, beta_sigma))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_sigma))

    mu = alpha + beta * x_c
    obs_sigma = jnp.sqrt(meas_sigma**2 + epsilon**2)

    numpyro.sample("y", dist.Normal(mu, obs_sigma), obs=y_obs)


def _fit_leverage_survey_numpyro(
    x: npt.ArrayLike,
    y_obs: npt.ArrayLike,
    y_err_low: npt.ArrayLike,
    y_err_high: npt.ArrayLike,
    *,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 14,
    num_chains: int = 1,
) -> az.InferenceData:
    x_np = _as_1d_float(x)
    y_np = _as_1d_float(y_obs)
    el_np = _as_1d_float(y_err_low)
    eh_np = _as_1d_float(y_err_high)

    m = _finite_mask(x_np, y_np, el_np, eh_np)
    if int(m.sum()) == 0:
        raise ValueError("No finite rows after filtering.")

    x_np, y_np, el_np, eh_np = x_np[m], y_np[m], el_np[m], eh_np[m]

    meas_sigma_np = 0.5 * (np.abs(el_np) + np.abs(eh_np))
    meas_sigma_np = np.clip(meas_sigma_np, 1e-6, None)

    x_mean = float(x_np.mean())
    x_c_np = x_np - x_mean

    span_x = _safe_ptp(x_c_np, fallback=1.0)
    span_y = _safe_ptp(y_np, fallback=1.0)
    y_sd = _safe_sd(y_np, fallback=1.0)

    alpha_mu = float(y_np.mean())
    alpha_sigma = max(float(y_sd / np.sqrt(y_np.size)), 1e-3)
    beta_sigma = max(float(span_y / span_x), 1e-3)
    epsilon_sigma = max(float(y_sd), 1e-3)

    model_kwargs = dict(
        x_c=jnp.asarray(x_c_np),
        meas_sigma=jnp.asarray(meas_sigma_np),
        y_obs=jnp.asarray(y_np),
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
        beta_sigma=beta_sigma,
        epsilon_sigma=epsilon_sigma,
    )

    rng_key = jax.random.PRNGKey(int(random_seed))
    mcmc, ll = _run_nuts(
        _linear_scatter_model,
        rng_key,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        num_chains=num_chains,
        model_kwargs=model_kwargs,
    )

    idata = az.from_numpyro(mcmc, log_likelihood=ll)
    return idata


@dataclass(frozen=True, slots=True)
class ModelConfig:
    draws: int = 2000
    tune: int = 1000
    target_accept: float = 0.9
    num_chains: int = 1


class Model:
    """
    NumPyro/JAX version of the linear + intrinsic scatter model,
    run on Survey objects.
    """

    def __init__(
        self,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.9,
        num_chains: int = 1,
    ) -> None:
        self.cfg = ModelConfig(
            draws=int(draws),
            tune=int(tune),
            target_accept=float(target_accept),
            num_chains=int(num_chains),
        )

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_leverage_survey_numpyro(
            df["logM"].to_numpy(),
            df["log(X_H2O)"].to_numpy(),
            df["uncertainty_lower"].to_numpy(),
            df["uncertainty_upper"].to_numpy(),
            draws=self.cfg.draws,
            tune=self.cfg.tune,
            target_accept=self.cfg.target_accept,
            random_seed=int(random_seed),
            num_chains=self.cfg.num_chains,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> Dict[str, float]:
        summ = az.summary(
            idata,
            var_names=["alpha", "beta", "epsilon"],
            hdi_prob=0.68,
            round_to=None,
        ).rename(index={"epsilon": "sigma"})

        return {
            "survey_id": float(survey.survey_id),
            "class_label": float(survey.class_label) if isinstance(survey.class_label, (int, float, np.number)) else survey.class_label,  # type: ignore[assignment]
            "N": float(survey.n),
            "L_met": float(survey.leverage(col="log(X_H2O)")),
            "L_logM": float(survey.leverage(col="logM")),
            "alpha_mean": float(summ.loc["alpha", "mean"]),
            "alpha_sd": float(summ.loc["alpha", "sd"]),
            "alpha_hdi16": float(summ.loc["alpha", "hdi_16%"]),
            "alpha_hdi84": float(summ.loc["alpha", "hdi_84%"]),
            "beta_mean": float(summ.loc["beta", "mean"]),
            "beta_sd": float(summ.loc["beta", "sd"]),
            "beta_hdi16": float(summ.loc["beta", "hdi_16%"]),
            "beta_hdi84": float(summ.loc["beta", "hdi_84%"]),
            "sigma_mean": float(summ.loc["sigma", "mean"]),
            "sigma_sd": float(summ.loc["sigma", "sd"]),
            "sigma_hdi16": float(summ.loc["sigma", "hdi_16%"]),
            "sigma_hdi84": float(summ.loc["sigma", "hdi_84%"]),
        }

    def run_on_surveys(self, surveys: Sequence[Survey], seed: int = 123) -> pd.DataFrame:
        rng = np.random.default_rng(int(seed))
        rows: List[Dict[str, Any]] = []
        for survey in surveys:
            rs = int(rng.integers(0, 2**32 - 1))
            idata = self.fit_survey(survey, random_seed=rs)
            rows.append(self.summarize_single(survey, idata))
        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)


# -----------------------------------------
# 2) Metallicty model: y on logM and [Fe/H]
# -----------------------------------------
def _met_model(
    *,
    x_m_c: jax.Array,           # (n,)
    x_s_obs: jax.Array,         # (n,)
    sig_meas_p: jax.Array,      # (n,)
    sig_meas_s: jax.Array,      # (n,)
    y_planet: Optional[jax.Array] = None,  # (n,)
    alpha_mu: float,
    alpha_sigma: float,
    beta_m_sigma: float,
    beta_s_sigma: float,
    epsilon_sigma: float,
) -> None:
    # latent true stellar metallicity (measurement model)
    x_s_true = numpyro.sample("x_s_true", dist.Normal(x_s_obs, sig_meas_s))
    x_s_true_c = x_s_true - jnp.mean(x_s_true)

    alpha = numpyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
    beta_m = numpyro.sample("beta_m", dist.Normal(0.0, beta_m_sigma))
    beta_s = numpyro.sample("beta_s", dist.Normal(0.0, beta_s_sigma))

    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_sigma))
    numpyro.deterministic("sigma_p", epsilon)

    mu = alpha + beta_m * x_m_c + beta_s * x_s_true_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon**2)

    numpyro.sample("y_planet", dist.Normal(mu, obs_sigma), obs=y_planet)


def _fit_met_survey_numpyro(
    x_mass: npt.ArrayLike,
    x_star: npt.ArrayLike,
    y_planet: npt.ArrayLike,
    y_planet_err_low: npt.ArrayLike,
    y_planet_err_high: npt.ArrayLike,
    x_star_err_low: npt.ArrayLike,
    x_star_err_high: npt.ArrayLike,
    *,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 14,
    num_chains: int = 1,
) -> az.InferenceData:
    x_m = _as_1d_float(x_mass)
    x_s_obs = _as_1d_float(x_star)
    yp = _as_1d_float(y_planet)

    el_p = _as_1d_float(y_planet_err_low)
    eh_p = _as_1d_float(y_planet_err_high)
    el_s = _as_1d_float(x_star_err_low)
    eh_s = _as_1d_float(x_star_err_high)

    m = _finite_mask(x_m, x_s_obs, yp, el_p, eh_p, el_s, eh_s)
    x_m, x_s_obs, yp, el_p, eh_p, el_s, eh_s = (
        x_m[m], x_s_obs[m], yp[m], el_p[m], eh_p[m], el_s[m], eh_s[m]
    )

    if x_m.size == 0:
        raise ValueError("No finite rows for metallicity model in this survey.")

    sig_meas_p_np = 0.5 * (np.abs(el_p) + np.abs(eh_p))
    sig_meas_s_np = 0.5 * (np.abs(el_s) + np.abs(eh_s))
    sig_meas_p_np = np.clip(sig_meas_p_np, 1e-6, None)
    sig_meas_s_np = np.clip(sig_meas_s_np, 1e-6, None)

    xm_mean = float(x_m.mean())
    xs_mean = float(x_s_obs.mean())

    x_m_c_np = x_m - xm_mean
    x_s_c_obs_np = x_s_obs - xs_mean  # only for prior scaling

    span_xm = _safe_ptp(x_m_c_np, fallback=1.0)
    span_xs = _safe_ptp(x_s_c_obs_np, fallback=1.0)
    span_yp = _safe_ptp(yp, fallback=1.0)
    yp_sd = _safe_sd(yp, fallback=1.0)

    alpha_mu = float(yp.mean())
    alpha_sigma = max(float(yp_sd / np.sqrt(yp.size)), 1e-3)

    # scales should map predictor spans to response span (unit-consistent)
    beta_m_sigma = max(float(span_yp / span_xm), 1e-3)
    beta_s_sigma = max(float(span_yp / span_xs), 1e-3)

    epsilon_sigma = max(float(yp_sd), 1e-3)

    model_kwargs = dict(
        x_m_c=jnp.asarray(x_m_c_np),
        x_s_obs=jnp.asarray(x_s_obs),
        sig_meas_p=jnp.asarray(sig_meas_p_np),
        sig_meas_s=jnp.asarray(sig_meas_s_np),
        y_planet=jnp.asarray(yp),
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
        beta_m_sigma=beta_m_sigma,
        beta_s_sigma=beta_s_sigma,
        epsilon_sigma=epsilon_sigma,
    )

    rng_key = jax.random.PRNGKey(int(random_seed))
    mcmc, ll = _run_nuts(
        _met_model,
        rng_key,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        num_chains=num_chains,
        model_kwargs=model_kwargs,
    )

    idata = az.from_numpyro(mcmc, log_likelihood=ll)
    return idata


class MetModel:
    """
    NumPyro/JAX version of:
      y_p = log(X_H2O) ~ alpha + beta_m*(logM-centered) + beta_s*(Fe/H_true-centered)
    with:
      - heteroskedastic measurement error on y_p and Fe/H,
      - latent Fe/H_true per planet,
      - intrinsic scatter epsilon exposed as sigma_p.
    """

    def __init__(
        self,
        draws: int = 2000,
        tune: int = 1000,
        target_accept: float = 0.9,
        num_chains: int = 1,
    ) -> None:
        self.cfg = ModelConfig(
            draws=int(draws),
            tune=int(tune),
            target_accept=float(target_accept),
            num_chains=int(num_chains),
        )

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_met_survey_numpyro(
            df["logM"].to_numpy(),
            df["Star Metallicity"].to_numpy(),
            df["log(X_H2O)"].to_numpy(),
            df["uncertainty_lower"].to_numpy(),
            df["uncertainty_upper"].to_numpy(),
            df["Star Metallicity Error Lower"].to_numpy(),
            df["Star Metallicity Error Upper"].to_numpy(),
            draws=self.cfg.draws,
            tune=self.cfg.tune,
            target_accept=self.cfg.target_accept,
            random_seed=int(random_seed),
            num_chains=self.cfg.num_chains,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> Dict[str, float]:
        summ = az.summary(
            idata,
            var_names=["alpha", "beta_m", "beta_s", "sigma_p"],
            hdi_prob=0.68,
            round_to=None,
        )

        row: Dict[str, Any] = {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": survey.leverage(col="log(X_H2O)"),
            "L_logM": survey.leverage(col="logM"),
        }

        def add_param(prefix: str, name: str) -> None:
            row[f"{prefix}_mean"] = float(summ.loc[name, "mean"])
            row[f"{prefix}_sd"] = float(summ.loc[name, "sd"])
            row[f"{prefix}_hdi16"] = float(summ.loc[name, "hdi_16%"])
            row[f"{prefix}_hdi84"] = float(summ.loc[name, "hdi_84%"])

        add_param("alpha", "alpha")
        add_param("beta_m", "beta_m")
        add_param("beta_s", "beta_s")
        add_param("sigma_p", "sigma_p")

        # keep return type consistent (float values where applicable)
        return {k: float(v) if isinstance(v, (int, float, np.number)) else v for k, v in row.items()}  # type: ignore[return-value]

    def run_on_surveys(self, surveys: Sequence[Survey], seed: int = 321) -> pd.DataFrame:
        rng = np.random.default_rng(int(seed))
        rows: List[Dict[str, Any]] = []
        for survey in surveys:
            rs = int(rng.integers(0, 2**32 - 1))
            idata = self.fit_survey(survey, random_seed=rs)
            rows.append(self.summarize_single(survey, idata))
        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)
