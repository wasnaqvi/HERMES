# src/Model.py
from __future__ import annotations
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import List

from .Survey import Survey  # relative import


def _fit_leverage_survey(
    x,
    y_obs,
    y_err_low,
    y_err_high,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 14,
) -> az.InferenceData:
    """
    Linear + intrinsic scatter model:
        y ~ Normal(alpha + beta * (x - mean(x)), sqrt(meas_sigma^2 + epsilon^2))
    """

    x = np.asarray(x, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_err_low = np.asarray(y_err_low, dtype=float).ravel()
    y_err_high = np.asarray(y_err_high, dtype=float).ravel()

    mask = (
        np.isfinite(x)
        & np.isfinite(y_obs)
        & np.isfinite(y_err_low)
        & np.isfinite(y_err_high)
    )
    if mask.sum() == 0:
        raise ValueError("No finite rows after filtering.")
    x, y_obs, y_err_low, y_err_high = (
        x[mask],
        y_obs[mask],
        y_err_low[mask],
        y_err_high[mask],
    )

    meas_sigma = 0.5 * (np.abs(y_err_low) + np.abs(y_err_high))
    meas_sigma = np.clip(meas_sigma, 1e-6, None)

    x_mean = float(x.mean())
    x_c = x - x_mean

    span_x = float(np.ptp(x_c) or 1.0)
    span_y = float(np.ptp(y_obs) or 1.0)

    with pm.Model() as model:
        x_c_data = pm.Data("x_c", x_c)
        meas_sigma_data = pm.Data("meas_sigma", meas_sigma)

        alpha = pm.Normal(
            "alpha",
            mu=float(y_obs.mean()),
            sigma=max(float(y_obs.std() / np.sqrt(len(y_obs))), 1e-3),
        )
        beta = pm.Normal(
            "beta",
            mu=0.0,
            sigma=span_y / span_x,
        )
        epsilon = pm.HalfNormal(
            "epsilon",
            sigma=max(float(y_obs.std()), 1e-3),
        )

        mu = alpha + beta * x_c_data
        obs_sigma = pm.math.sqrt(meas_sigma_data**2 + epsilon**2)

        pm.Normal("y", mu=mu, sigma=obs_sigma, observed=y_obs)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    return idata

class Model:
    """
    Wraps the PyMC linear+intrinsic-scatter model
    and runs it on a list of Survey objects.
    """

    def __init__(self, draws: int = 2000, tune: int = 1000, target_accept: float = 0.9):
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_leverage_survey(
            df["logM"].values,
            df["log(X_H2O)"].values,
            df["uncertainty_lower"].values,
            df["uncertainty_upper"].values,
            draws=self.draws,
            tune=self.tune,
            target_accept=self.target_accept,
            random_seed=random_seed,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> dict:
        summ = az.summary(
            idata,
            var_names=["alpha", "beta", "epsilon"],
            hdi_prob=0.68,
            round_to=None,
        )
        # rename epsilon -> sigma in the summary dict
        summ = summ.rename(index={"epsilon": "sigma"})

        return {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": survey.leverage(col="log(X_H2O)"),
            "L_logM": survey.leverage(col="logM"),
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

    def run_on_surveys(self, surveys: List[Survey], seed: int = 123) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for survey in surveys:
            rs = int(rng.integers(0, 2**32 - 1))
            idata = self.fit_survey(survey, random_seed=rs)
            rows.append(self.summarize_single(survey, idata))
        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)

import pymc as pm
import numpy as np
import arviz as az
import pandas as pd
from typing import List
from .Survey import Survey  # already there above


def _fit_met_survey(
    x_mass,
    x_star,
    y_planet,
    y_planet_err_low,
    y_planet_err_high,
    x_star_err_low,
    x_star_err_high,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9,
    random_seed: int = 14,
) -> az.InferenceData:
    """
    Bi-variate regression with 2 inputs and one output.

      x_m = logM (planet mass)
      x_s = stellar metallicity [Fe/H]
      y_p = planetary metallicity log(X_H2O)

    Model:

        y_p ~ Normal(
            alpha
            + beta_m * (x_m - mean_xm)
            + beta_s * (x_s_true - mean_xs),
            sqrt(sigma_meas_p^2 + epsilon^2)
        )

    where x_s_true is a latent "true" stellar metallicity with
    measurement error:

        x_s_obs ~ Normal(x_s_true, sigma_meas_s)

    and epsilon is the intrinsic scatter of the regression.
    """

    # --------- arrays and masking ----------
    x_m = np.asarray(x_mass, float).ravel()
    x_s_obs = np.asarray(x_star, float).ravel()
    yp = np.asarray(y_planet, float).ravel()

    el_p = np.asarray(y_planet_err_low, float).ravel()
    eh_p = np.asarray(y_planet_err_high, float).ravel()
    el_s = np.asarray(x_star_err_low, float).ravel()
    eh_s = np.asarray(x_star_err_high, float).ravel()

    m = (
        np.isfinite(x_m) & np.isfinite(x_s_obs) & np.isfinite(yp)
        & np.isfinite(el_p) & np.isfinite(eh_p)
        & np.isfinite(el_s) & np.isfinite(eh_s)
    )
    x_m, x_s_obs, yp, el_p, eh_p, el_s, eh_s = (
        x_m[m], x_s_obs[m], yp[m], el_p[m], eh_p[m], el_s[m], eh_s[m]
    )

    if x_m.size == 0:
        raise ValueError("No finite rows for metallicity model in this survey.")

    # --------- effective measurement sigmas ----------
    sig_meas_p = 0.5 * (np.abs(el_p) + np.abs(eh_p))
    sig_meas_s = 0.5 * (np.abs(el_s) + np.abs(eh_s))
    sig_meas_p = np.clip(sig_meas_p, 1e-6, None)
    sig_meas_s = np.clip(sig_meas_s, 1e-6, None)

    # --------- centering & spans ----------
    xm_mean = float(x_m.mean())
    xs_mean = float(x_s_obs.mean())

    x_m_c = x_m - xm_mean
    x_s_c_obs = x_s_obs - xs_mean

    span_xm = float(np.ptp(x_m_c) or 1.0)
    span_xs = float(np.ptp(x_s_c_obs) or 1.0)
    span_yp = float(np.ptp(yp) or 1.0)

    # --------- PyMC model ----------
    with pm.Model() as model:
        # data containers
        x_m_c_data   = pm.Data("x_m_c", x_m_c)
        x_s_obs_data = pm.Data("x_s_obs", x_s_obs)
        sig_p_data   = pm.Data("sig_meas_p", sig_meas_p)
        sig_s_data   = pm.Data("sig_meas_s", sig_meas_s)

        # latent true stellar metallicity, with measurement error
        x_s_true = pm.Normal(
            "x_s_true",
            mu=x_s_obs_data,
            sigma=sig_s_data,
            shape=x_m.size,
        )

        # center the latent stellar metallicity
        x_s_true_c = x_s_true - pm.math.mean(x_s_true)

        # regression coefficients
        alpha = pm.Normal(
            "alpha",
            mu=float(yp.mean()),
            sigma=float(yp.std() / np.sqrt(len(yp)) + 1e-3),
        )
        beta_m = pm.Normal(
            "beta_m",
            mu=0.0,
            sigma=span_yp / span_xm,
        )
        beta_s = pm.Normal(
            "beta_s",
            mu=0.0,
            sigma=span_yp / span_xs,
        )

        # intrinsic scatter on the regression
        epsilon = pm.HalfNormal(
            "epsilon",
            sigma=float(yp.std() + 1e-3),
        )
        # expose as "sigma_p" for summaries/plots
        sigma_p = pm.Deterministic("sigma_p", epsilon)

        # mean relation
        mu = alpha + beta_m * x_m_c_data + beta_s * x_s_true_c

        # total scatter: measurement + intrinsic
        obs_sigma = pm.math.sqrt(sig_p_data**2 + epsilon**2)

        pm.Normal("y_planet", mu=mu, sigma=obs_sigma, observed=yp)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,
            progressbar=False,
        )

    return idata

class MetModel:
    """
    Bi-variate regression of planetary metallicity on:
      - logM (planet mass)
      - stellar metallicity [Fe/H]

    With:
      - heteroskedastic measurement error on y_p and x_s,
      - intrinsic scatter epsilon (exposed as sigma_p).
    """

    def __init__(self, draws: int = 2000, tune: int = 1000, target_accept: float = 0.9):
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_met_survey(
            df["logM"].values,
            df["Star Metallicity"].values,
            df["log(X_H2O)"].values,
            df["uncertainty_lower"].values,
            df["uncertainty_upper"].values,
            df["Star Metallicity Error Lower"].values,
            df["Star Metallicity Error Upper"].values,
            draws=self.draws,
            tune=self.tune,
            target_accept=self.target_accept,
            random_seed=random_seed,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> dict:
        """
        Extract posterior means, SDs, and 68% HDIs for:
          alpha, beta_m, beta_s, sigma_p
        """
        summ = az.summary(
            idata,
            var_names=["alpha", "beta_m", "beta_s", "sigma_p"],
            hdi_prob=0.68,
            round_to=None,
        )

        row = {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": survey.leverage(col="log(X_H2O)"),
            "L_logM": survey.leverage(col="logM"),
        }

        def add_param(prefix, name):
            row[f"{prefix}_mean"]   = float(summ.loc[name, "mean"])
            row[f"{prefix}_sd"]     = float(summ.loc[name, "sd"])
            row[f"{prefix}_hdi16"]  = float(summ.loc[name, "hdi_16%"])
            row[f"{prefix}_hdi84"]  = float(summ.loc[name, "hdi_84%"])

        add_param("alpha",   "alpha")
        add_param("beta_m",  "beta_m")
        add_param("beta_s",  "beta_s")
        add_param("sigma_p", "sigma_p")

        return row

    def run_on_surveys(self, surveys: List[Survey], seed: int = 321) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for survey in surveys:
            rs = int(rng.integers(0, 2**32 - 1))
            idata = self.fit_survey(survey, random_seed=rs)
            rows.append(self.summarize_single(survey, idata))
        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)

    """
    2D metallicity model: planet + star metallicities vs logM, with intrinsic 2x2 covariance.
    """

    def __init__(self, draws: int = 2000, tune: int = 1000, target_accept: float = 0.9):
        self.draws = draws
        self.tune = tune
        self.target_accept = target_accept

    def fit_survey(self, survey: Survey, random_seed: int = 14) -> az.InferenceData:
        df = survey.df
        return _fit_met_survey(
            df["logM"].values,
            df["log(X_H2O)"].values,
            df["Star Metallicity"].values,
            df["uncertainty_lower"].values,
            df["uncertainty_upper"].values,
            df["Star Metallicity Error Lower"].values,
            df["Star Metallicity Error Upper"].values,
            draws=self.draws,
            tune=self.tune,
            target_accept=self.target_accept,
            random_seed=random_seed,
        )

    def summarize_single(self, survey: Survey, idata: az.InferenceData) -> dict:
        """
        Extract posterior means, SDs, and 68% HDIs for:
          alpha_p, beta_p, alpha_s, beta_s, sigma_p, sigma_s, rho
        """
        summ = az.summary(
            idata,
            var_names=["alpha_p", "beta_p", "alpha_s", "beta_s",
                       "sigma_p", "sigma_s", "rho"],
            hdi_prob=0.68,
            round_to=None,
        )

        row = {
            "survey_id": survey.survey_id,
            "class_label": survey.class_label,
            "N": survey.n,
            "L_met": survey.leverage(col="log(X_H2O)"),
            "L_logM": survey.leverage(col="logM"),
        }

        def add_param(prefix, name):
            row[f"{prefix}_mean"]   = float(summ.loc[name, "mean"])
            row[f"{prefix}_sd"]     = float(summ.loc[name, "sd"])
            row[f"{prefix}_hdi16"]  = float(summ.loc[name, "hdi_16%"])
            row[f"{prefix}_hdi84"]  = float(summ.loc[name, "hdi_84%"])

        add_param("alpha_p", "alpha_p")
        add_param("beta_p",  "beta_p")
        add_param("alpha_s", "alpha_s")
        add_param("beta_s",  "beta_s")
        add_param("sigma_p", "sigma_p")
        add_param("sigma_s", "sigma_s")
        add_param("rho",     "rho")

        return row

    def run_on_surveys(self, surveys: List[Survey], seed: int = 321) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        for survey in surveys:
            rs = int(rng.integers(0, 2**32 - 1))
            idata = self.fit_survey(survey, random_seed=rs)
            rows.append(self.summarize_single(survey, idata))
        return pd.DataFrame(rows).sort_values("survey_id").reset_index(drop=True)
