# src/Model.py
from __future__ import annotations
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union, Literal

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

from .Survey import Survey  

Array1D = npt.NDArray[np.floating]
ModelKind = Literal["lin", "met", "met_no_scatter", "met_no_stellar"]
JaxDType = Union[jnp.float32, jnp.float64]

def _as_1d_float(x: npt.ArrayLike) -> Array1D:
    return np.asarray(x, dtype=float).ravel()


def _finite_mask(*arrays: Array1D) -> npt.NDArray[np.bool_]:
    '''
    Basicall returns a boolean mask to select finite values over
    for array a by
    a[m]
    '''
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


def _et68(x: np.ndarray) -> Tuple[float, float]:
    """
    Equal-tailed 68% interval (fast). (Not HDI!!)
    """
    lo, hi = np.quantile(x, [0.16, 0.84])
    return float(lo), float(hi)


def _scalar_stats_from_idata(idata: az.InferenceData, var: str) -> Dict[str, float]:
    """
    Fast scalar summaries from posterior samples:
      mean, sd, and equal-tailed 68% interval.
    """
    s = np.asarray(idata.posterior[var]).reshape(-1)
    mean = float(s.mean())
    sd = float(s.std(ddof=1)) if s.size > 1 else 0.0
    lo, hi = _et68(s)
    return {"mean": mean, "sd": sd, "hdi16": lo, "hdi84": hi}


# NUTS runner (compute log_likelihood only if you want the WAICs for Model comparison.
def _run_nuts(
    model_fn,
    rng_key: jax.Array,
    *,
    draws: int,
    tune: int,
    target_accept: float,
    num_chains: int,
    model_kwargs: Mapping[str, Any],
    compute_log_lik: bool,
    chain_method: Literal["parallel", "vectorized", "sequential"] = "sequential",
) -> Tuple[MCMC, Optional[Dict[str, jax.Array]]]:
    kernel = NUTS(model_fn, target_accept_prob=float(target_accept))
    mcmc = MCMC(
        kernel,
        num_warmup=int(tune),
        num_samples=int(draws),
        num_chains=int(num_chains),
        chain_method=chain_method,
        progress_bar=False,
    )
    mcmc.run(rng_key, **model_kwargs)

    if not compute_log_lik:
        return mcmc, None

    posterior = mcmc.get_samples(group_by_chain=True)
    ll = log_likelihood(model_fn, posterior, **model_kwargs)
    return mcmc, ll


def _linear_scatter_model(
    *,
    x_c: jax.Array,                 # (N,)
    meas_sigma: jax.Array,          # (N,)
    y_obs: Optional[jax.Array],     # (N,)
    alpha_mu: float,
    alpha_sigma: float,
    beta_sigma: float,
    epsilon_sigma: float,
) -> None:
    alpha = numpyro.sample("alpha", dist.Normal(alpha_mu, alpha_sigma))
    beta = numpyro.sample("beta", dist.Normal(0.0, beta_sigma))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_sigma))

    alpha_b = alpha[..., None]
    beta_b = beta[..., None]
    eps_b = epsilon[..., None]

    mu = alpha_b + beta_b * x_c
    obs_sigma = jnp.sqrt(meas_sigma**2 + eps_b**2)

    numpyro.sample("y", dist.Normal(mu, obs_sigma), obs=y_obs)


def _fit_leverage_survey_numpyro(
    x: npt.ArrayLike,
    y_obs: npt.ArrayLike,
    y_err_low: npt.ArrayLike,
    y_err_high: npt.ArrayLike,
    *,
    cfg: "ModelConfig",
    random_seed: int,
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

    x_c_np = x_np - float(x_np.mean())

    span_x = _safe_ptp(x_c_np, fallback=1.0)
    span_y = _safe_ptp(y_np, fallback=1.0)
    y_sd = _safe_sd(y_np, fallback=1.0)

    alpha_mu = float(y_np.mean())
    alpha_sigma = max(float(y_sd / np.sqrt(y_np.size)), 1e-3)
    beta_sigma = max(float(span_y / span_x), 1e-3)
    epsilon_sigma = max(float(y_sd), 1e-3)

    dtype = cfg.jax_dtype

    model_kwargs: Dict[str, Any] = dict(
        x_c=jnp.asarray(x_c_np, dtype=dtype),
        meas_sigma=jnp.asarray(meas_sigma_np, dtype=dtype),
        y_obs=jnp.asarray(y_np, dtype=dtype),
        alpha_mu=alpha_mu,
        alpha_sigma=alpha_sigma,
        beta_sigma=beta_sigma,
        epsilon_sigma=epsilon_sigma,
    )

    rng_key = jax.random.PRNGKey(int(random_seed))
    mcmc, ll = _run_nuts(
        _linear_scatter_model,
        rng_key,
        draws=cfg.draws,
        tune=cfg.tune,
        target_accept=cfg.target_accept,
        num_chains=cfg.num_chains,
        model_kwargs=model_kwargs,
        compute_log_lik=cfg.compute_log_lik,
        chain_method=cfg.chain_method,
    )
    return az.from_numpyro(mcmc, log_likelihood=ll) if ll is not None else az.from_numpyro(mcmc)


def _met_model(
    *,
    x_m_c: jax.Array,               # (N,) centered mass: x_m - mean(x_m)
    x_s_obs: jax.Array,             # (N,) observed stellar metallicity
    sig_meas_p: jax.Array,          # (N,) planet metallicity measurement sigma
    sig_meas_s: jax.Array,          # (N,) stellar metallicity measurement sigma
    y_planet: Optional[jax.Array],  # (N,) planet metallicity
    alpha_p_mu: float,
    alpha_p_sigma: float,
    beta_p_sigma: float,
    beta_s_sigma: float,
    epsilon_p_sigma: float,
) -> None:
    # latent true stellar metallicity
    x_s_true = numpyro.sample("x_s_true", dist.Normal(x_s_obs, sig_meas_s))

    # center by observed mean (deterministic anchor — not the latent mean,
    # which would shift the intercept α_p for every MCMC draw)
    x_s_true_c = x_s_true - jnp.mean(x_s_obs)

    # priors: extension of 2D model + one-to-one expectation on beta_s (Model B)
    alpha_p = numpyro.sample("alpha_p", dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample("beta_p",  dist.Normal(0.0, beta_p_sigma))
    beta_s  = numpyro.sample("beta_s",  dist.Normal(1.0, beta_s_sigma))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_p_sigma))

    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon[..., None]**2)
    numpyro.sample("y_planet", dist.Normal(mu, obs_sigma), obs=y_planet)


def _met_model_no_scatter(
    *,
    x_m_c: jax.Array,
    x_s_obs: jax.Array,
    sig_meas_p: jax.Array,
    sig_meas_s: jax.Array,
    y_planet: Optional[jax.Array],
    alpha_p_mu: float,
    alpha_p_sigma: float,
    beta_p_sigma: float,
    beta_s_sigma: float,
    epsilon_p_sigma: float,
) -> None:
    """Full 3D model with epsilon forced to zero (for WAIC comparison)."""
    x_s_true = numpyro.sample("x_s_true", dist.Normal(x_s_obs, sig_meas_s))
    x_s_true_c = x_s_true - jnp.mean(x_s_obs)
    alpha_p = numpyro.sample("alpha_p", dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample("beta_p",  dist.Normal(0.0, beta_p_sigma))
    beta_s  = numpyro.sample("beta_s",  dist.Normal(1.0, beta_s_sigma))
    numpyro.deterministic("epsilon", jnp.zeros(()))
    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
    numpyro.sample("y_planet", dist.Normal(mu, sig_meas_p), obs=y_planet)


def _met_model_no_stellar(
    *,
    x_m_c: jax.Array,
    x_s_obs: jax.Array,
    sig_meas_p: jax.Array,
    sig_meas_s: jax.Array,
    y_planet: Optional[jax.Array],
    alpha_p_mu: float,
    alpha_p_sigma: float,
    beta_p_sigma: float,
    beta_s_sigma: float,
    epsilon_p_sigma: float,
) -> None:
    """2D mass-only model with beta_s=0 (for WAIC comparison vs full 3D)."""
    alpha_p = numpyro.sample("alpha_p", dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample("beta_p",  dist.Normal(0.0, beta_p_sigma))
    numpyro.deterministic("beta_s", jnp.zeros(()))
    epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_p_sigma))
    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon[..., None]**2)
    numpyro.sample("y_planet", dist.Normal(mu, obs_sigma), obs=y_planet)


def make_met_model_fixed_scatter(fixed_value: float):
    """
    Factory: returns a MetModel variant with epsilon fixed to ``fixed_value``
    (not sampled).  Useful for scatter-threshold experiments.
    """
    fv = float(fixed_value)

    def _fixed_scatter_model(
        *,
        x_m_c: jax.Array,
        x_s_obs: jax.Array,
        sig_meas_p: jax.Array,
        sig_meas_s: jax.Array,
        y_planet: Optional[jax.Array],
        alpha_p_mu: float,
        alpha_p_sigma: float,
        beta_p_sigma: float,
        beta_s_sigma: float,
        epsilon_p_sigma: float,
    ) -> None:
        x_s_true = numpyro.sample("x_s_true", dist.Normal(x_s_obs, sig_meas_s))
        x_s_true_c = x_s_true - jnp.mean(x_s_obs)
        alpha_p = numpyro.sample("alpha_p", dist.Normal(alpha_p_mu, alpha_p_sigma))
        beta_p  = numpyro.sample("beta_p",  dist.Normal(0.0, beta_p_sigma))
        beta_s  = numpyro.sample("beta_s",  dist.Normal(1.0, beta_s_sigma))
        numpyro.deterministic("epsilon", jnp.array(fv))
        mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
        obs_sigma = jnp.sqrt(sig_meas_p**2 + jnp.array(fv) ** 2)
        numpyro.sample("y_planet", dist.Normal(mu, obs_sigma), obs=y_planet)

    return _fixed_scatter_model

def _fit_met_survey_numpyro(
    x_mass: npt.ArrayLike,
    x_star: npt.ArrayLike,
    y_planet: npt.ArrayLike,
    y_planet_err_low: npt.ArrayLike,
    y_planet_err_high: npt.ArrayLike,
    x_star_err_low: npt.ArrayLike,
    x_star_err_high: npt.ArrayLike,
    *,
    cfg: "ModelConfig",
    random_seed: int,
    model_fn=None,
) -> az.InferenceData:
    if model_fn is None:
        model_fn = _met_model
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

    x_m_c_np = x_m - float(x_m.mean())
    x_s_c_obs_np = x_s_obs - float(x_s_obs.mean())  # only for prior scaling

    span_xm = _safe_ptp(x_m_c_np, fallback=1.0)
    span_xs = _safe_ptp(x_s_c_obs_np, fallback=1.0)
    span_yp = _safe_ptp(yp, fallback=1.0)
    yp_sd = _safe_sd(yp, fallback=1.0)

    alpha_mu = float(yp.mean())
    alpha_sigma = max(float(yp_sd / np.sqrt(yp.size)), 1e-3)

    beta_m_sigma = max(float(span_yp / span_xm), 1e-3)
    beta_s_sigma = max(float(span_yp / span_xs), 1e-3)
    epsilon_sigma = max(float(yp_sd), 1e-3)

    dtype = cfg.jax_dtype

    model_kwargs: Dict[str, Any] = dict(
        x_m_c=jnp.asarray(x_m_c_np, dtype=dtype),
        x_s_obs=jnp.asarray(x_s_obs, dtype=dtype),
        sig_meas_p=jnp.asarray(sig_meas_p_np, dtype=dtype),
        sig_meas_s=jnp.asarray(sig_meas_s_np, dtype=dtype),
        y_planet=jnp.asarray(yp, dtype=dtype),
        alpha_p_mu=alpha_mu,
        alpha_p_sigma=alpha_sigma,
        beta_p_sigma=beta_m_sigma,   # same heuristic scale as before, just renamed
        beta_s_sigma=beta_s_sigma,
        epsilon_p_sigma=epsilon_sigma,
    )

    rng_key = jax.random.PRNGKey(int(random_seed))
    mcmc, ll = _run_nuts(
        model_fn,
        rng_key,
        draws=cfg.draws,
        tune=cfg.tune,
        target_accept=cfg.target_accept,
        num_chains=cfg.num_chains,
        model_kwargs=model_kwargs,
        compute_log_lik=cfg.compute_log_lik,
        chain_method=cfg.chain_method,
    )

    return az.from_numpyro(mcmc, log_likelihood=ll) if ll is not None else az.from_numpyro(mcmc)

# -------------------------
# Config + parallel helper
# -------------------------

@dataclass(frozen=True, slots=True)
class ModelConfig:
    draws: int = 800
    tune: int = 800
    target_accept: float = 0.90
    num_chains: int = 4
    compute_log_lik: bool = False
    chain_method: Literal["parallel", "vectorized", "sequential"] = "sequential"
    jax_dtype: JaxDType = jnp.float32


def _fit_one_job(job: Tuple[ModelKind, Dict[str, Any], Survey, int]) -> Dict[str, Any]:
    """
    Module-level for multiprocessing spawn pickling.
    Supports kinds: "lin", "met", "met_no_scatter", "met_no_stellar".
    """
    kind, cfg_dict, survey, seed = job
    cfg = ModelConfig(**cfg_dict)

    if kind == "lin":
        idata = _fit_leverage_survey_numpyro(
            survey.df["logM"].to_numpy(),
            survey.df["log(X_H2O)"].to_numpy(),
            survey.df["uncertainty_lower"].to_numpy(),
            survey.df["uncertainty_upper"].to_numpy(),
            cfg=cfg,
            random_seed=int(seed),
        )
        a = _scalar_stats_from_idata(idata, "alpha")
        b = _scalar_stats_from_idata(idata, "beta")
        e = _scalar_stats_from_idata(idata, "epsilon")
        return {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_logM": float(survey.leverage(col="logM")),
            "alpha_mean": a["mean"], "alpha_sd": a["sd"], "alpha_hdi16": a["hdi16"], "alpha_hdi84": a["hdi84"],
            "beta_mean":  b["mean"], "beta_sd":  b["sd"], "beta_hdi16":  b["hdi16"], "beta_hdi84":  b["hdi84"],
            "sigma_mean": e["mean"], "sigma_sd": e["sd"], "sigma_hdi16": e["hdi16"], "sigma_hdi84": e["hdi84"],
        }

    # MetModel variants — choose model function by kind string
    _met_dispatch: Dict[str, Any] = {
        "met":             _met_model,
        "met_no_scatter":  _met_model_no_scatter,
        "met_no_stellar":  _met_model_no_stellar,
    }
    if kind not in _met_dispatch:
        raise ValueError(f"Unknown model kind: {kind!r}")
    model_fn = _met_dispatch[kind]

    idata = _fit_met_survey_numpyro(
        survey.df["logM"].to_numpy(),
        survey.df["Star Metallicity"].to_numpy(),
        survey.df["log(X_H2O)"].to_numpy(),
        survey.df["uncertainty_lower"].to_numpy(),
        survey.df["uncertainty_upper"].to_numpy(),
        survey.df["Star Metallicity Error Lower"].to_numpy(),
        survey.df["Star Metallicity Error Upper"].to_numpy(),
        cfg=cfg,
        random_seed=int(seed),
        model_fn=model_fn,
    )

    a  = _scalar_stats_from_idata(idata, "alpha_p")
    bp = _scalar_stats_from_idata(idata, "beta_p")
    bs = _scalar_stats_from_idata(idata, "beta_s")
    e  = _scalar_stats_from_idata(idata, "epsilon")

    return {
        "survey_id": survey.survey_id,
        "class_label": survey.class_label,
        "N": survey.n,
        "L_mass": float(survey.leverage(
            col="logM",
            err_lower_col="logM_err_lower",
            err_upper_col="logM_err_upper",
        )),
        "L_stellar": float(survey.leverage(
            col="Star Metallicity",
            err_lower_col="Star Metallicity Error Lower",
            err_upper_col="Star Metallicity Error Upper",
        )),
        "alpha_p_mean": a["mean"], "alpha_p_sd": a["sd"], "alpha_p_hdi16": a["hdi16"], "alpha_p_hdi84": a["hdi84"],
        "beta_p_mean":  bp["mean"], "beta_p_sd":  bp["sd"], "beta_p_hdi16":  bp["hdi16"], "beta_p_hdi84":  bp["hdi84"],
        "beta_s_mean":  bs["mean"], "beta_s_sd":  bs["sd"], "beta_s_hdi16":  bs["hdi16"], "beta_s_hdi84":  bs["hdi84"],
        "epsilon_mean": e["mean"], "epsilon_sd": e["sd"], "epsilon_hdi16": e["hdi16"], "epsilon_hdi84": e["hdi84"],
    }
# -------------------------
# Public API: Model classes
# -------------------------
class Model:
    """
    1D Model: Planetary Metallicity ~ Mass only
    
    Fast NumPyro/JAX linear + intrinsic scatter model.
    
    Model equation:
        y_planet ~ Normal(alpha + beta * (mass - mean_mass), sqrt(sigma_meas^2 + epsilon^2))

    Defaults are tuned for speed (CPU-friendly).
    Set compute_log_lik=True only when you need WAIC/LOO.
    """

    def __init__(
        self,
        draws: int = 800,
        tune: int = 800,
        target_accept: float = 0.90,
        num_chains: int = 4,
        compute_log_lik: bool = False,
        chain_method: Literal["parallel", "vectorized", "sequential"] = "sequential",
        jax_dtype: JaxDType = jnp.float32,
    ) -> None:
        self.cfg = ModelConfig(
            draws=int(draws),
            tune=int(tune),
            target_accept=float(target_accept),
            num_chains=int(num_chains),
            compute_log_lik=bool(compute_log_lik),
            chain_method=chain_method,
            jax_dtype=jax_dtype,
        )

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_leverage_survey_numpyro(
            df["logM"].to_numpy(),
            df["log(X_H2O)"].to_numpy(),
            df["uncertainty_lower"].to_numpy(),
            df["uncertainty_upper"].to_numpy(),
            cfg=self.cfg,
            random_seed=int(random_seed),
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> Dict[str, Any]:
        a = _scalar_stats_from_idata(idata, "alpha")
        b = _scalar_stats_from_idata(idata, "beta")
        e = _scalar_stats_from_idata(idata, "epsilon")

        return {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": float(survey.leverage(col="log(X_H2O)")),
            "L_logM": float(survey.leverage(col="logM")),
            "alpha_mean": a["mean"], "alpha_sd": a["sd"], "alpha_hdi16": a["hdi16"], "alpha_hdi84": a["hdi84"],
            "beta_mean":  b["mean"], "beta_sd":  b["sd"], "beta_hdi16":  b["hdi16"], "beta_hdi84":  b["hdi84"],
            "sigma_mean": e["mean"], "sigma_sd": e["sd"], "sigma_hdi16": e["hdi16"], "sigma_hdi84": e["hdi84"],
        }

    def run_on_surveys(
        self,
        surveys: Sequence[Survey],
        seed: int = 123,
        *,
        parallel: bool = True,
        processes: int = 4,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(int(seed))

        if not parallel:
            rows: List[Dict[str, Any]] = []
            for survey in surveys:
                rs = int(rng.integers(0, 2**32 - 1))
                idata = self.fit_survey(survey, random_seed=rs)
                rows.append(self.summarize_single(survey, idata))
            return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)

        jobs: List[Tuple[ModelKind, Dict[str, Any], Survey, int]] = [
            ("lin", asdict(self.cfg), survey, int(rng.integers(0, 2**32 - 1)))
            for survey in surveys
        ]
        ctx = get_context("spawn")  # macOS safe
        with ctx.Pool(processes=int(processes)) as pool:
            rows = pool.map(_fit_one_job, jobs)

        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)


class MetModel:
    """
    3D Model: Planetary Metallicity ~ Mass + Stellar Metallicity
    
    Fast NumPyro/JAX metallicity model with latent true stellar metallicity.
    
    Model equation:
        x_s_true ~ Normal(x_s_obs, sigma_s_meas)  # Latent true stellar metallicity
        y_planet ~ Normal(alpha_p + beta_p * (mass - mean_mass) + beta_s * (x_s_true - mean_x_s_true), 
                          sqrt(sigma_meas^2 + epsilon^2))

    Defaults are tuned for speed (CPU-friendly).
    Set compute_log_lik=True only when you need WAIC/LOO.
    """

    def __init__(
        self,
        draws: int = 800,
        tune: int = 800,
        target_accept: float = 0.90,
        num_chains: int = 4,
        compute_log_lik: bool = False,
        chain_method: Literal["parallel", "vectorized", "sequential"] = "sequential",
        jax_dtype: JaxDType = jnp.float32,
    ) -> None:
        self.cfg = ModelConfig(
            draws=int(draws),
            tune=int(tune),
            target_accept=float(target_accept),
            num_chains=int(num_chains),
            compute_log_lik=bool(compute_log_lik),
            chain_method=chain_method,
            jax_dtype=jax_dtype,
        )

    def fit_survey(
        self,
        survey: Survey,
        random_seed: int = 14,
        model_fn=None,
    ) -> az.InferenceData:
        """Fit one survey. Pass ``model_fn`` to use a variant (e.g. ``_met_model_no_stellar``)."""
        df = survey.df
        return _fit_met_survey_numpyro(
            df["logM"].to_numpy(),
            df["Star Metallicity"].to_numpy(),
            df["log(X_H2O)"].to_numpy(),
            df["uncertainty_lower"].to_numpy(),
            df["uncertainty_upper"].to_numpy(),
            df["Star Metallicity Error Lower"].to_numpy(),
            df["Star Metallicity Error Upper"].to_numpy(),
            cfg=self.cfg,
            random_seed=int(random_seed),
            model_fn=model_fn,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> Dict[str, Any]:
        a  = _scalar_stats_from_idata(idata, "alpha_p")
        bp = _scalar_stats_from_idata(idata, "beta_p")
        bs = _scalar_stats_from_idata(idata, "beta_s")
        e  = _scalar_stats_from_idata(idata, "epsilon")

        return {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_mass": float(survey.leverage(
                col="logM",
                err_lower_col="logM_err_lower",
                err_upper_col="logM_err_upper",
            )),
            "L_stellar": float(survey.leverage(
                col="Star Metallicity",
                err_lower_col="Star Metallicity Error Lower",
                err_upper_col="Star Metallicity Error Upper",
            )),
            "alpha_p_mean": a["mean"], "alpha_p_sd": a["sd"], "alpha_p_hdi16": a["hdi16"], "alpha_p_hdi84": a["hdi84"],
            "beta_p_mean":  bp["mean"], "beta_p_sd":  bp["sd"], "beta_p_hdi16":  bp["hdi16"], "beta_p_hdi84":  bp["hdi84"],
            "beta_s_mean":  bs["mean"], "beta_s_sd":  bs["sd"], "beta_s_hdi16":  bs["hdi16"], "beta_s_hdi84":  bs["hdi84"],
            "epsilon_mean": e["mean"], "epsilon_sd": e["sd"], "epsilon_hdi16": e["hdi16"], "epsilon_hdi84": e["hdi84"],
        }

    def run_on_surveys(
        self,
        surveys: Sequence[Survey],
        seed: int = 321,
        *,
        model_kind: ModelKind = "met",
        parallel: bool = False,
        processes: int = 4,
    ) -> pd.DataFrame:
        """
        Fit ``model_kind`` to every survey.

        ``model_kind`` may be one of:
          - ``"met"``            : full 3-D model (default)
          - ``"met_no_scatter"`` : epsilon fixed at 0 (WAIC comparison)
          - ``"met_no_stellar"`` : 2-D mass-only (WAIC comparison)
        """
        rng = np.random.default_rng(int(seed))

        # choose model function for sequential path
        _dispatch: Dict[str, Any] = {
            "met":            _met_model,
            "met_no_scatter": _met_model_no_scatter,
            "met_no_stellar": _met_model_no_stellar,
        }
        if model_kind not in _dispatch:
            raise ValueError(f"Unknown model_kind: {model_kind!r}")
        model_fn = _dispatch[model_kind]

        if not parallel:
            rows: List[Dict[str, Any]] = []
            for survey in surveys:
                rs = int(rng.integers(0, 2**32 - 1))
                idata = self.fit_survey(survey, random_seed=rs, model_fn=model_fn)
                rows.append(self.summarize_single(survey, idata))
            return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)

        jobs: List[Tuple[ModelKind, Dict[str, Any], Survey, int]] = [
            (model_kind, asdict(self.cfg), survey, int(rng.integers(0, 2**32 - 1)))
            for survey in surveys
        ]
        ctx = get_context("spawn")  # macOS safe
        with ctx.Pool(processes=int(processes)) as pool:
            rows = pool.map(_fit_one_job, jobs)

        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)
    
    