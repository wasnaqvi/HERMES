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
