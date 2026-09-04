#!/usr/bin/env python
"""
prior_sensitivity.py
--------------------
Referee response (R1, Section 2.4): test whether the MetModel posteriors and the
leverage/N scaling results depend on the choice of prior.

We refit the 3D MetModel on a subset of mock surveys under three prior sets:

  current       : as in the paper
                    alpha_p ~ N(mean(y), y_sd/sqrt(N))     <-- sqrt(N) intercept prior
                    beta_p  ~ N(0,   span_y/span_mass)
                    beta_s  ~ N(1.0, span_y/span_stellar)
                    epsilon ~ HalfNormal(y_sd)

  loose         : N-independent, ~5x wider scales, same centres
                    alpha_p ~ N(mean(y), 5*y_sd)            <-- NO sqrt(N) dependence
                    beta_p  ~ N(0,   5*span_y/span_mass)
                    beta_s  ~ N(1.0, 5*span_y/span_stellar)
                    epsilon ~ HalfNormal(5*y_sd)

  uninformative : broad, neutral centres
                    alpha_p ~ N(mean(y), 100)
                    beta_p  ~ N(0,   100)
                    beta_s  ~ N(0,   100)                   <-- centre 0, not 1
                    epsilon ~ HalfNormal(100)

For each fit we record posterior mean/sd for {alpha_p, beta_p, beta_s, epsilon}
plus max R-hat and min ESS. Output: prior_sensitivity_results.csv and a summary.

Key questions answered:
  1. Do posterior means/sds move between prior sets? (should be small if data-dominated)
  2. Does sigma_alpha still scale ~ N^-0.5 under the loose (non-sqrt(N)) intercept
     prior? (tests whether the paper's intercept scaling is a prior artefact)
"""
from __future__ import annotations

import argparse
from typing import Dict, Any

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import arviz as az

from src.data import HermesData
from src.Survey import SurveySampler
from src.Model import (
    _met_model,
    _run_nuts,
    ModelConfig,
    _as_1d_float,
    _finite_mask,
    _safe_ptp,
    _safe_sd,
)

DATA_PATH = "dataset/hermes_synthetic_data_0.6.0.csv"

# prior-set definitions: multipliers on the base scales, plus intercept mode + beta_s centre
PRIOR_SETS: Dict[str, Dict[str, Any]] = {
    "current": dict(alpha_mode="sqrtN", alpha_mult=1.0, slope_mult=1.0,
                    eps_mult=1.0, beta_s_center=1.0, alpha_fixed=None),
    "loose": dict(alpha_mode="fixed_ysd", alpha_mult=5.0, slope_mult=5.0,
                  eps_mult=5.0, beta_s_center=1.0, alpha_fixed=None),
    "uninformative": dict(alpha_mode="absolute", alpha_mult=1.0, slope_mult=1.0,
                          eps_mult=1.0, beta_s_center=0.0, alpha_fixed=100.0),
}


def build_kwargs(survey, cfg, pset):
    """Assemble _met_model kwargs with prior scales overridden per prior set."""
    df = survey.df
    x_m = _as_1d_float(df["logM"].to_numpy())
    x_s_obs = _as_1d_float(df["Star Metallicity"].to_numpy())
    yp = _as_1d_float(df["log(X_H2O)"].to_numpy())
    el_p = _as_1d_float(df["uncertainty_lower"].to_numpy())
    eh_p = _as_1d_float(df["uncertainty_upper"].to_numpy())
    el_s = _as_1d_float(df["Star Metallicity Error Lower"].to_numpy())
    eh_s = _as_1d_float(df["Star Metallicity Error Upper"].to_numpy())

    m = _finite_mask(x_m, x_s_obs, yp, el_p, eh_p, el_s, eh_s)
    x_m, x_s_obs, yp = x_m[m], x_s_obs[m], yp[m]
    el_p, eh_p, el_s, eh_s = el_p[m], eh_p[m], el_s[m], eh_s[m]

    sig_p = np.clip(0.5 * (np.abs(el_p) + np.abs(eh_p)), 1e-6, None)
    sig_s = np.clip(0.5 * (np.abs(el_s) + np.abs(eh_s)), 1e-6, None)

    x_m_c = x_m - float(x_m.mean())
    x_s_c = x_s_obs - float(x_s_obs.mean())

    span_xm = _safe_ptp(x_m_c, 1.0)
    span_xs = _safe_ptp(x_s_c, 1.0)
    span_yp = _safe_ptp(yp, 1.0)
    yp_sd = _safe_sd(yp, 1.0)
    N = yp.size

    # intercept prior width
    if pset["alpha_mode"] == "sqrtN":
        alpha_sigma = yp_sd / np.sqrt(N)
    elif pset["alpha_mode"] == "fixed_ysd":
        alpha_sigma = pset["alpha_mult"] * yp_sd
    elif pset["alpha_mode"] == "absolute":
        alpha_sigma = pset["alpha_fixed"]
    else:
        raise ValueError(pset["alpha_mode"])
    alpha_sigma = max(float(alpha_sigma), 1e-3)

    if pset["alpha_mode"] == "absolute":
        beta_p_sigma = pset["alpha_fixed"]
        beta_s_sigma = pset["alpha_fixed"]
        eps_sigma = pset["alpha_fixed"]
    else:
        beta_p_sigma = max(pset["slope_mult"] * span_yp / span_xm, 1e-3)
        beta_s_sigma = max(pset["slope_mult"] * span_yp / span_xs, 1e-3)
        eps_sigma = max(pset["eps_mult"] * yp_sd, 1e-3)

    dtype = cfg.jax_dtype
    kwargs = dict(
        x_m_c=jnp.asarray(x_m_c, dtype=dtype),
        x_s_obs=jnp.asarray(x_s_obs, dtype=dtype),
        sig_meas_p=jnp.asarray(sig_p, dtype=dtype),
        sig_meas_s=jnp.asarray(sig_s, dtype=dtype),
        y_planet=jnp.asarray(yp, dtype=dtype),
        alpha_p_mu=float(yp.mean()),
        alpha_p_sigma=float(alpha_sigma),
        beta_p_sigma=float(beta_p_sigma),
        beta_s_sigma=float(beta_s_sigma),
        epsilon_p_sigma=float(eps_sigma),
    )
    return kwargs, pset["beta_s_center"]


def make_model(beta_s_center):
    """Return a _met_model variant with beta_s prior centred at beta_s_center."""
    if beta_s_center == 1.0:
        return _met_model
    import numpyro
    import numpyro.distributions as dist

    def _model(*, x_m_c, x_s_obs, sig_meas_p, sig_meas_s, y_planet,
               alpha_p_mu, alpha_p_sigma, beta_p_sigma, beta_s_sigma, epsilon_p_sigma):
        x_s_true = numpyro.sample("x_s_true", dist.Normal(x_s_obs, sig_meas_s))
        x_s_true_c = x_s_true - jnp.mean(x_s_obs)
        alpha_p = numpyro.sample("alpha_p", dist.Normal(alpha_p_mu, alpha_p_sigma))
        beta_p = numpyro.sample("beta_p", dist.Normal(0.0, beta_p_sigma))
        beta_s = numpyro.sample("beta_s", dist.Normal(beta_s_center, beta_s_sigma))
        epsilon = numpyro.sample("epsilon", dist.HalfNormal(epsilon_p_sigma))
        mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
        obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon[..., None]**2)
        numpyro.sample("y_planet", dist.Normal(mu, obs_sigma), obs=y_planet)

    return _model


def fit_stats(survey, cfg, pset, seed):
    kwargs, bsc = build_kwargs(survey, cfg, pset)
    model_fn = make_model(bsc)
    mcmc, _ = _run_nuts(
        model_fn, jax.random.PRNGKey(int(seed)),
        draws=cfg.draws, tune=cfg.tune, target_accept=cfg.target_accept,
        num_chains=cfg.num_chains, model_kwargs=kwargs,
        compute_log_lik=False, chain_method=cfg.chain_method,
    )
    idata = az.from_numpyro(mcmc)
    vars = ["alpha_p", "beta_p", "beta_s", "epsilon"]
    rhat = az.rhat(idata, var_names=vars)
    ess = az.ess(idata, var_names=vars)
    out = {}
    for v in vars:
        s = np.asarray(idata.posterior[v]).reshape(-1)
        out[f"{v}_mean"] = float(s.mean())
        out[f"{v}_sd"] = float(s.std(ddof=1))
    out["max_rhat"] = float(max(float(rhat[v].values) for v in vars))
    out["min_ess"] = float(min(float(ess[v].values) for v in vars))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N-grid", type=int, nargs="+", default=[50, 150, 400])
    ap.add_argument("--classes", nargs="+", default=["S1"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[321, 42, 7])
    ap.add_argument("--draws", type=int, default=800)
    ap.add_argument("--tune", type=int, default=800)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--out", default="prior_sensitivity_results.csv")
    args = ap.parse_args()

    cfg = ModelConfig(draws=args.draws, tune=args.tune, target_accept=0.9,
                      num_chains=args.chains, compute_log_lik=False,
                      chain_method="sequential")

    hermes = HermesData.from_csv(DATA_PATH)
    sampler = SurveySampler(hermes, rng_seed=0)
    surveys = sampler.sample_grid(args.N_grid, n_reps_per_combo=args.reps,
                                  class_order=args.classes)
    print(f"[info] {len(surveys)} surveys x {len(PRIOR_SETS)} prior sets "
          f"x {len(args.seeds)} seeds = {len(surveys)*len(PRIOR_SETS)*len(args.seeds)} fits")

    rows = []
    for si, sv in enumerate(surveys):
        for pname, pset in PRIOR_SETS.items():
            for seed in args.seeds:
                st = fit_stats(sv, cfg, pset, seed)
                rows.append(dict(prior_set=pname, survey_id=sv.survey_id,
                                 class_label=sv.class_label, N=sv.n, seed=seed, **st))
        print(f"  [{si+1}/{len(surveys)}] {sv.class_label} N={sv.n} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[saved] {args.out}  ({len(df)} rows)")

    # ---- summary: fractional shift vs 'current' at matched (survey_id, seed) ----
    print("\n=== POSTERIOR SHIFT vs 'current' (median |delta|/current across fits) ===")
    base = df[df.prior_set == "current"].set_index(["survey_id", "seed"])
    for pname in ["loose", "uninformative"]:
        sub = df[df.prior_set == pname].set_index(["survey_id", "seed"])
        j = sub.join(base, rsuffix="_cur")
        line = [f"{pname:>14s}:"]
        for v in ["beta_p_mean", "beta_s_mean", "alpha_p_sd", "beta_p_sd", "beta_s_sd", "epsilon_mean"]:
            rel = np.abs(j[v] - j[f"{v}_cur"]) / (np.abs(j[f"{v}_cur"]) + 1e-9)
            line.append(f"{v}={100*np.median(rel):.1f}%")
        print("  ".join(line))

    # ---- sigma_alpha vs N scaling per prior set (the circularity test) ----
    print("\n=== sigma_alpha(alpha_p_sd) vs N power-law exponent per prior set ===")
    print("    (paper reports ~ -0.5; loose set drops the sqrt(N) prior)")
    for pname in PRIOR_SETS:
        sub = df[df.prior_set == pname]
        g = sub.groupby("N")["alpha_p_sd"].mean()
        if len(g) >= 2:
            coef = np.polyfit(np.log(g.index.values), np.log(g.values), 1)
            print(f"  {pname:>14s}: sigma_alpha ~ N^{coef[0]:+.3f}   "
                  + "  ".join(f"N={n}:{v:.4f}" for n, v in g.items()))

    print(f"\n[diagnostics] max R-hat={df.max_rhat.max():.4f}  "
          f"min ESS={df.min_ess.min():.0f}")


if __name__ == "__main__":
    main()
