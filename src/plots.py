# src/plots.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- small utility ---------------------------------------
def _ensure_dir_for(path: str | Path):
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------- column resolution (NEW) ------------------------------
def _pick_col(df: pd.DataFrame, *candidates: str) -> str:
    """
    Return the first column name in `candidates` that exists in df.
    Raise a helpful error if none exist.
    """
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of these columns exist: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def resolve_alpha_cols(df: pd.DataFrame) -> dict:
    return {
        "mean": _pick_col(df, "alpha_p_mean", "alpha_mean"),
        "sd":   _pick_col(df, "alpha_p_sd",   "alpha_sd"),
        "lo":   _pick_col(df, "alpha_p_hdi16","alpha_hdi16"),
        "hi":   _pick_col(df, "alpha_p_hdi84","alpha_hdi84"),
    }


def resolve_beta_mass_cols(df: pd.DataFrame) -> dict:
    # 1D uses beta_*, 3D uses beta_p_*
    return {
        "mean": _pick_col(df, "beta_p_mean", "beta_mean"),
        "sd":   _pick_col(df, "beta_p_sd",   "beta_sd"),
        "lo":   _pick_col(df, "beta_p_hdi16","beta_hdi16"),
        "hi":   _pick_col(df, "beta_p_hdi84","beta_hdi84"),
    }


def resolve_beta_star_cols(df: pd.DataFrame) -> dict | None:
    # Only exists for MetModel
    if "beta_s_mean" not in df.columns:
        return None
    return {
        "mean": "beta_s_mean",
        "sd":   "beta_s_sd",
        "lo":   "beta_s_hdi16",
        "hi":   "beta_s_hdi84",
    }


def resolve_scatter_cols(df: pd.DataFrame) -> dict:
    # 1D Model stores epsilon under sigma_*, MetModel stores epsilon_*
    return {
        "mean": _pick_col(df, "epsilon_mean", "sigma_mean"),
        "sd":   _pick_col(df, "epsilon_sd",   "sigma_sd"),
        "lo":   _pick_col(df, "epsilon_hdi16","sigma_hdi16"),
        "hi":   _pick_col(df, "epsilon_hdi84","sigma_hdi84"),
    }


def resolve_L_col(df: pd.DataFrame, preferred: str = "L_logM") -> str:
    if preferred in df.columns:
        return preferred
    # common fallbacks
    for c in ("L_3d", "L_2d", "L_logM", "L_met"):
        if c in df.columns:
            return c
    raise KeyError(f"No leverage column found. Tried {preferred} and fallbacks.")


@dataclass
class FitResult:
    a: float
    b: float


def _powerlaw_fit(x, sigma):
    """
    Fit log–log model: log(sigma) = log(a) + b*log(x).
    Returns a,b such that sigma ≈ a * x**b.
    """
    x = np.asarray(x, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(s) & (x > 0) & (s > 0)
    x = x[m]
    s = s[m]
    if x.size < 2:
        raise ValueError("Need at least 2 finite, positive points to fit.")
    lx = np.log(x)
    ly = np.log(s)
    A = np.vstack([np.ones_like(lx), lx]).T
    beta = np.linalg.lstsq(A, ly, rcond=None)[0]
    return FitResult(a=float(np.exp(beta[0])), b=float(beta[1]))


def _linear_fit(x, sigma):
    """OLS in original space: sigma ≈ c + m * x."""
    x = np.asarray(x, dtype=float).ravel()
    s = np.asarray(sigma, dtype=float).ravel()
    msk = np.isfinite(x) & np.isfinite(s)
    x = x[msk]
    s = s[msk]
    if x.size < 2:
        raise ValueError("Need at least 2 points to fit.")
    X = np.vstack([np.ones_like(x), x]).T
    c, m = np.linalg.lstsq(X, s, rcond=None)[0]
    return float(c), float(m)


# ----------------------- bands ------------------------------------------------
def _linear_band(x, y, xg, prediction=False, z=1.0):
    """
    Fit y ≈ b0 + b1*x (OLS). Return (y_hat, lo, hi) at xg for z-sigma band.
    prediction=False -> mean band; True -> prediction band (adds residual var).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    X = np.vstack([np.ones_like(x), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    xg = np.asarray(xg, float)
    y_hat = beta[0] + beta[1] * xg

    r = y - (X @ beta)
    dof = max(len(y) - 2, 1)
    s2 = float(np.dot(r, r) / dof)
    XtX_inv = np.linalg.inv(X.T @ X)

    Xg = np.vstack([np.ones_like(xg), xg]).T
    var_mean = np.einsum("ij,jk,ik->i", Xg, XtX_inv, Xg) * s2
    if prediction:
        var_mean = var_mean + s2
    se = np.sqrt(np.maximum(var_mean, 0.0))
    return y_hat, y_hat - z * se, y_hat + z * se


def _powerlaw_band(x, y, xg, prediction=False, z=1.0):
    """
    Fit log–log: log(y) = b0 + b1*log(x). Return (y_hat, lo, hi) at xg.
    Band is computed in log-space then exponentiated (multiplicative band).
    """
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[m], y[m]
    lx, ly = np.log(x), np.log(y)
    A = np.vstack([np.ones_like(lx), lx]).T
    beta, *_ = np.linalg.lstsq(A, ly, rcond=None)

    lxg = np.log(np.asarray(xg, float))
    mu_log = beta[0] + beta[1] * lxg

    r = ly - (A @ beta)
    dof = max(len(ly) - 2, 1)
    s2 = float(np.dot(r, r) / dof)
    AtA_inv = np.linalg.inv(A.T @ A)
    Ag = np.vstack([np.ones_like(lxg), lxg]).T
    var_mean = np.einsum("ij,jk,ik->i", Ag, AtA_inv, Ag) * s2
    if prediction:
        var_mean = var_mean + s2

    se = np.sqrt(np.maximum(var_mean, 0.0))
    lo_log = mu_log - z * se
    hi_log = mu_log + z * se

    y_hat = np.exp(mu_log)
    lo = np.exp(lo_log)
    hi = np.exp(hi_log)
    return y_hat, lo, hi


# ----------------------- plotting helpers ------------------------------------
def _plot_sigma_vs_x(
    x,
    sigma,
    symbol,
    title,
    outfile,
    add_linear=False,
    x_label=r"$x$",
    xvar="x",
    hdi_lo=None,
    hdi_hi=None,
    show_band=True,
    band_kind="mean",
    band_z=1.0,
):
    """
    Scatter (or HDI errorbars) of sigma vs x.
    Overlays power-law fit in log space with optional band.
    band_kind: 'mean' (default) or 'pred' for prediction band.
    band_z: 1.0 (~68%), 1.96 (~95%), etc.
    """
    x = np.asarray(x, float)
    sigma = np.asarray(sigma, float)
    xg = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)

    plt.figure(figsize=(7.0, 3.6))

    if hdi_lo is not None and hdi_hi is not None:
        hdi_lo = np.asarray(hdi_lo, float).ravel()
        hdi_hi = np.asarray(hdi_hi, float).ravel()
        m = (
            np.isfinite(x)
            & np.isfinite(sigma)
            & np.isfinite(hdi_lo)
            & np.isfinite(hdi_hi)
        )
        x, sigma, hdi_lo, hdi_hi = x[m], sigma[m], hdi_lo[m], hdi_hi[m]
        yerr = np.vstack([sigma - hdi_lo, hdi_hi - sigma])
        plt.errorbar(x, sigma, yerr=yerr, fmt="o", capsize=3, lw=1, alpha=0.9)
    else:
        plt.scatter(x, sigma)

    # central power-law and band
    y_hat, lo, hi = _powerlaw_band(
        x,
        sigma,
        xg,
        prediction=(band_kind == "pred"),
        z=band_z,
    )
    if show_band:
        plt.fill_between(xg, lo, hi, alpha=0.15, linewidth=0)
    plt.plot(xg, y_hat, linestyle="--")

    # optional linear overlay in original space
    if add_linear:
        c, m = _linear_fit(x, sigma)
        plt.plot(xg, c + m * xg, linestyle="-.")

        ax = plt.gca()
        xr, yr = ax.get_xlim(), ax.get_ylim()
        txt_lin = rf"lin: $\sigma_{{{symbol}}}={c:.2g}+{m:.2g}{xvar}$"
        plt.text(
            xr[0] + 0.62 * (xr[1] - xr[0]),
            yr[0] + 0.77 * (yr[1] - yr[0]),
            txt_lin,
        )

    # annotation from power-law fit
    fr = _powerlaw_fit(x, sigma)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(fr"$\sigma_{{{symbol}}}$")
    ax = plt.gca()
    xr, yr = ax.get_xlim(), ax.get_ylim()
    if fr.b < 0:
        ann = rf"fit: $\sigma_{{{symbol}}}=\dfrac{{{fr.a:.2g}}}{{{xvar}^{{{-fr.b:.2f}}}}}$"
    else:
        ann = rf"fit: $\sigma_{{{symbol}}}={fr.a:.2g}\,{xvar}^{{{fr.b:.2f}}}$"
    plt.text(
        xr[0] + 0.62 * (xr[1] - xr[0]),
        yr[0] + 0.85 * (yr[1] - yr[0]),
        ann,
    )

    plt.minorticks_on()
    plt.tight_layout()
    _ensure_dir_for(outfile)
    plt.savefig(outfile)  # vector (PDF)
    plt.close()


def _plot_param_with_hdi(
    x,
    mu,
    lo,
    hi,
    symbol,
    title,
    outfile,
    x_label,
    fit_band=True,
    band_kind="mean",
    band_z=1.0,
):
    """
    Plot α or β posterior means with 68% HDIs, plus optional linear 1σ band.
    band_kind: 'mean' or 'pred'; band_z controls level (1.0≈68%, 1.96≈95%).
    """
    x = np.asarray(x, float).ravel()
    mu = np.asarray(mu, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    m = np.isfinite(x) & np.isfinite(mu) & np.isfinite(lo) & np.isfinite(hi)
    x, mu, lo, hi = x[m], mu[m], lo[m], hi[m]
    yerr = np.vstack([mu - lo, hi - mu])

    plt.figure(figsize=(7.0, 3.6))
    plt.errorbar(x, mu, yerr=yerr, fmt="o", capsize=3, lw=1, alpha=0.95)

    if fit_band and len(x) >= 3:
        xg = np.linspace(x.min() * 0.98, x.max() * 1.02, 200)
        y_hat, blo, bhi = _linear_band(
            x,
            mu,
            xg,
            prediction=(band_kind == "pred"),
            z=band_z,
        )
        plt.fill_between(xg, blo, bhi, alpha=0.15, linewidth=0)
        plt.plot(xg, y_hat, linestyle="--")

    plt.xlabel(x_label)
    plt.ylabel(fr"${symbol}$ (mean ± HDI)")
    plt.title(title)
    plt.minorticks_on()
    plt.tight_layout()
    _ensure_dir_for(outfile)
    plt.savefig(outfile)  # vector
    plt.close()


# ----------------------- leverage panels API ---------------------------------
def make_leverage_panels_with_hdi(
    x,
    alpha_sd,
    beta_sd,
    epsilon_sd,
    add_linear: bool = False,
    mode: Literal["L", "N"] = "L",
    prefix: str = "leverage_panels_",
    # α/β posterior means with HDIs (optional)
    alpha_mean=None,
    alpha_hdi16=None,
    alpha_hdi84=None,
    beta_mean=None,
    beta_hdi16=None,
    beta_hdi84=None,
    plot_param_hdi: bool = False,
    # HDIs for the sigmas themselves (optional)
    sigma_alpha_lo=None,
    sigma_alpha_hi=None,
    sigma_beta_lo=None,
    sigma_beta_hi=None,
    sigma_eps_lo=None,
    sigma_eps_hi=None,
    # band settings
    sigma_band_kind: Literal["mean", "pred"] = "mean",
    sigma_band_z: float = 1.0,
    param_band_kind: Literal["mean", "pred"] = "mean",
    param_band_z: float = 1.0,
):
    """
    Produces:
      - 3 panels: sigma_alpha/beta/epsilon vs x with power-law fit + band.
      - Optionally 2 panels: α and β (posterior means ± HDIs) vs x.

    x is either leverage L or N, depending on `mode`.
    """
    out_prefix = f"{prefix}{mode}"

    if mode == "L":
        x_label = r"Survey leverage $L$"
        xvar = "L"
        t_alpha = r"Fixed $N$: Uncertainty on the intercept $\sigma_\alpha$ vs. $L$"
        t_beta  = r"Fixed $N$: Uncertainty on the slope $\sigma_\beta$ vs. $L$"
        t_eps   = r"Fixed $N$: Uncertainty on intrinsic scatter $\sigma_\varepsilon$ vs. $L$"
    else:
        x_label = r"Number of Targets $N$"
        xvar = "N"
        t_alpha = r"Fixed $L$: Uncertainty on the intercept $\sigma_\alpha$ vs. $N$"
        t_beta  = r"Uncertainty on the slope $\sigma_\beta$ vs. $N$"
        t_eps   = r"Uncertainty on intrinsic scatter $\sigma_\varepsilon$ vs. $N$"

    _plot_sigma_vs_x(
        x, alpha_sd, symbol="\\alpha", title=t_alpha,
        outfile=Path("plots") / f"{out_prefix}_alpha.pdf",
        add_linear=add_linear, x_label=x_label, xvar=xvar,
        hdi_lo=sigma_alpha_lo, hdi_hi=sigma_alpha_hi,
        show_band=True, band_kind=sigma_band_kind, band_z=sigma_band_z,
    )
    _plot_sigma_vs_x(
        x, beta_sd, symbol="\\beta", title=t_beta,
        outfile=Path("plots") / f"{out_prefix}_beta.pdf",
        add_linear=add_linear, x_label=x_label, xvar=xvar,
        hdi_lo=sigma_beta_lo, hdi_hi=sigma_beta_hi,
        show_band=True, band_kind=sigma_band_kind, band_z=sigma_band_z,
    )
    _plot_sigma_vs_x(
        x, epsilon_sd, symbol="\\varepsilon", title=t_eps,
        outfile=Path("plots") / f"{out_prefix}_epsilon.pdf",
        add_linear=add_linear, x_label=x_label, xvar=xvar,
        hdi_lo=sigma_eps_lo, hdi_hi=sigma_eps_hi,
        show_band=True, band_kind=sigma_band_kind, band_z=sigma_band_z,
    )

    if plot_param_hdi and (alpha_mean is not None) and (alpha_hdi16 is not None) and (alpha_hdi84 is not None):
        _plot_param_with_hdi(
            x, np.asarray(alpha_mean, float), np.asarray(alpha_hdi16, float), np.asarray(alpha_hdi84, float),
            symbol="\\alpha",
            title=rf"Posterior $\alpha$ per survey (mean $\pm$ HDI) vs. ${xvar}$",
            outfile=Path("plots") / f"{out_prefix}_alpha_param.pdf",
            x_label=x_label, fit_band=True, band_kind=param_band_kind, band_z=param_band_z,
        )

    if plot_param_hdi and (beta_mean is not None) and (beta_hdi16 is not None) and (beta_hdi84 is not None):
        _plot_param_with_hdi(
            x, np.asarray(beta_mean, float), np.asarray(beta_hdi16, float), np.asarray(beta_hdi84, float),
            symbol="\\beta",
            title=rf"Posterior $\beta$ per survey (mean $\pm$ HDI) vs. ${xvar}$",
            outfile=Path("plots") / f"{out_prefix}_beta_param.pdf",
            x_label=x_label, fit_band=True, band_kind=param_band_kind, band_z=param_band_z,
        )


def make_leverage_panels_from_df(
    df: pd.DataFrame,
    out_prefix: str,
    x_col: Literal["leverage", "N"] = "leverage",
    L_preferred: str = "L_logM",
):
    """
    Schema-robust wrapper for the HERMES fit summary DataFrame.

    Works for BOTH:
      - 1D Model outputs (alpha_*, beta_*, sigma_*)
      - 3D MetModel outputs (alpha_p_*, beta_p_*, beta_s_*, epsilon_*)

    Required:
      N, class_label, and a leverage column (default prefers L_logM).
    """
    df = df.copy()

    L_col = resolve_L_col(df, preferred=L_preferred)

    if x_col == "leverage":
        x = df[L_col].to_numpy(float)
        mode = "L"
    elif x_col == "N":
        x = df["N"].to_numpy(float)
        mode = "N"
    else:
        raise ValueError("x_col must be 'leverage' or 'N'")

    a = resolve_alpha_cols(df)
    b = resolve_beta_mass_cols(df)
    s = resolve_scatter_cols(df)

    make_leverage_panels_with_hdi(
        x=x,
        alpha_sd=df[a["sd"]].to_numpy(float),
        beta_sd=df[b["sd"]].to_numpy(float),
        epsilon_sd=df[s["sd"]].to_numpy(float),
        add_linear=False,
        mode=mode,
        prefix=out_prefix + "_",
        alpha_mean=df.get(a["mean"]),
        alpha_hdi16=df.get(a["lo"]),
        alpha_hdi84=df.get(a["hi"]),
        beta_mean=df.get(b["mean"]),
        beta_hdi16=df.get(b["lo"]),
        beta_hdi84=df.get(b["hi"]),
        plot_param_hdi=False,
    )


# ----------------- design space: N vs std with L contours --------------------
def make_design_space_N_vs_std(
    surveys,
    col: str = "logM",
    out_path: str | Path = "plots/design_N_vs_std_logM.png",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = np.array([s.n for s in surveys], float)
    std = np.array([np.nanstd(s.df[col].to_numpy(float)) for s in surveys], float)
    labels = [s.class_label for s in surveys]

    fig, ax = plt.subplots()

    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = np.array([l == lab for l in labels])
        ax.scatter(N[mask], std[mask], label=lab, alpha=0.7)

    ax.set_xlabel("N (survey size)")
    ax.set_ylabel(f"Std dev of {col}")
    ax.set_title("Survey design space: N vs std dev")
    ax.set_ylim(bottom=0.0,top=1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(title="class label")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_design_space_N_with_L_contours(
    surveys,
    col: str = "logM",
    out_path: str | Path = "plots/design_N_vs_std_logM.pdf",
    L_levels: Sequence[float] | None = None,
):
    out_path = Path(out_path)
    _ensure_dir_for(out_path)

    N = np.array([s.n for s in surveys], float)
    std = np.array([np.nanstd(s.df[col].to_numpy(float)) for s in surveys], float)
    labels = [s.class_label for s in surveys]

    L_actual = np.sqrt((N - 1) * std**2)

    if L_levels is None:
        qs = [0.25, 0.5, 0.75, 0.9]
        L_levels = np.quantile(L_actual[np.isfinite(L_actual)], qs)

    L_levels = list(L_levels)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = np.array([l == lab for l in labels])
        ax.scatter(N[mask], std[mask], label=lab, alpha=0.8)

    N_grid = np.linspace(max(1.0, N.min()), N.max(), 300)
    for L0 in L_levels:
        std_curve = L0 / np.sqrt(N_grid)
        ax.plot(N_grid, std_curve, linestyle="--")
        ax.text(N_grid[-1], std_curve[-1], f"L≈{L0:.1f}", ha="left", va="center", fontsize=8)

    ax.set_xlabel("N (survey size)")
    ax.set_ylim(bottom=0.0,top=1.2)
    ax.set_ylabel(f"Std({col})")
    ax.set_title("Survey design space: N vs $\\sigma$ with Leverage contours")

    ax.legend(
        title="class label",
        loc="upper right",
        fontsize=8,
        title_fontsize=9,
        frameon=False,
        markerscale=0.8,
        handlelength=1.5,
        handletextpad=0.4,
    )

    plt.minorticks_on()
    plt.tight_layout()
    fig.savefig(out_path)  # vector
    plt.close(fig)


def make_fixedN_sigma_vs_L_scatter_from_df(
    df: pd.DataFrame,
    out_dir: str | Path = "plots",
    L_col: str = "L_logM",
) -> None:
    """
    For each distinct N, make a 3-panel figure:
        σ_alpha vs L, σ_beta(mass) vs L, σ_scatter vs L,
    coloured by class_label.

    Schema-robust:
      alpha_sd: alpha_sd or alpha_p_sd
      beta_sd : beta_sd  or beta_p_sd
      scatter : sigma_sd or epsilon_sd
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    L_col = resolve_L_col(df, preferred=L_col)

    a = resolve_alpha_cols(df)
    b = resolve_beta_mass_cols(df)
    s = resolve_scatter_cols(df)

    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {"S1": "C0", "S2": "C1", "S3": "C2", "S4": "C3"}

    y_cols   = [a["sd"], b["sd"], s["sd"]]
    y_labels = [r"$\sigma_\alpha$", r"$\sigma_{\beta}$", r"$\sigma_\varepsilon$"]
    y_syms   = ["\\alpha", "\\beta", "\\varepsilon"]

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
        fig.suptitle(rf"$N = {N0}$", fontsize=12)

        L_all = sub[L_col].to_numpy(float)

        for ax, ycol, ylabel, sym in zip(axes, y_cols, y_labels, y_syms):
            for cls in class_order:
                cls_sub = sub[sub["class_label"] == cls]
                if cls_sub.empty:
                    continue
                ax.scatter(
                    cls_sub[L_col].to_numpy(float),
                    cls_sub[ycol].to_numpy(float),
                    s=18,
                    alpha=0.9,
                    label=cls,
                    color=class_colors.get(cls, "k"),
                )

            ax.set_xlabel(fr"$L_M$")
            ax.set_ylabel(ylabel)
            ax.minorticks_on()

            y_all = sub[ycol].to_numpy(float)

            m = np.isfinite(L_all) & np.isfinite(y_all) & (L_all > 0) & (y_all > 0)
            x_fit = L_all[m]
            y_fit = y_all[m]

            if x_fit.size >= 2:
                xg = np.linspace(x_fit.min() * 0.98, x_fit.max() * 1.02, 200)

                y_hat, lo, hi = _powerlaw_band(x_fit, y_fit, xg, prediction=False, z=1.0)
                ax.fill_between(xg, lo, hi, alpha=0.15, linewidth=0)
                ax.plot(xg, y_hat, linestyle="--", linewidth=1.2)

                try:
                    c, m_lin = _linear_fit(x_fit, y_fit)
                    ax.plot(xg, c + m_lin * xg, linestyle="-.", linewidth=1.0)
                except ValueError:
                    pass

                fr = _powerlaw_fit(x_fit, y_fit)
                xr, yr = ax.get_xlim(), ax.get_ylim()
                if fr.b < 0:
                    ann = rf"fit: $\sigma_{{{sym}}}=\dfrac{{{fr.a:.2g}}}{{L^{{{-fr.b:.2f}}}}}$"
                else:
                    ann = rf"fit: $\sigma_{{{sym}}}={fr.a:.2g}L^{{{fr.b:.2f}}}$"
                ax.text(
                    xr[0] + 0.62 * (xr[1] - xr[0]),
                    yr[0] + 0.85 * (yr[1] - yr[0]),
                    ann,
                    fontsize=8,
                )

        handles, labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color=class_colors.get(cls, "k"))
                handles.append(h)
                labels.append(cls)
        if handles:
            axes[0].legend(handles, labels, title="class", fontsize=8, title_fontsize=9, frameon=False, loc="best")

        fig.tight_layout()
        out_file = out_dir / f"fixedN_N{N0}_sigma_vs_{L_col}.pdf"
        fig.savefig(out_file)
        plt.close(fig)


# =====================================================================
#  METALLICITY MODEL PANELS (MetModel) — schema-robust now
# =====================================================================

def _scatter_with_fits(
    ax,
    x,
    y,
    class_labels,
    class_order,
    class_colors,
    y_symbol_tex: str,
    L_label_tex: str = r"$L(\log M)$",
):
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_fit = x[m]
    y_fit = y[m]

    for cls in class_order:
        mask = (class_labels == cls)
        if not np.any(mask):
            continue
        ax.scatter(
            x[mask],
            y[mask],
            s=18,
            alpha=0.9,
            color=class_colors.get(cls, "k"),
            label=cls,
        )

    ax.set_xlabel(L_label_tex)
    ax.set_ylabel(y_symbol_tex)
    ax.minorticks_on()

    if x_fit.size < 2:
        return

    xg = np.linspace(x_fit.min() * 0.98, x_fit.max() * 1.02, 200)

    y_hat, lo, hi = _powerlaw_band(x_fit, y_fit, xg, prediction=False, z=1.0)
    ax.fill_between(xg, lo, hi, alpha=0.15, linewidth=0)
    ax.plot(xg, y_hat, linestyle="--", linewidth=1.2)

    try:
        c, m_lin = _linear_fit(x_fit, y_fit)
        ax.plot(xg, c + m_lin * xg, linestyle="-.", linewidth=1.0)
    except ValueError:
        pass

    fr = _powerlaw_fit(x_fit, y_fit)
    xr, yr = ax.get_xlim(), ax.get_ylim()
    ax.text(
        xr[0] + 0.60 * (xr[1] - xr[0]),
        yr[0] + 0.86 * (yr[1] - yr[0]),
        rf"fit: {y_symbol_tex} $\propto L^{{{fr.b:.2f}}}$",
        fontsize=8,
    )


def make_met_fixedN_uncertainty_vs_L_from_df(
    df: pd.DataFrame,
    out_dir: str | Path = "plots",
    L_col: str = "L_logM",
) -> None:
    """
    For each N (fixed-N panels), plot posterior SD vs leverage for the *MetModel*:

      alpha: alpha_p_sd
      beta_mass: beta_p_sd
      beta_star: beta_s_sd
      scatter: epsilon_sd

    NOTE: This expects MetModel-style columns; if beta_s_* absent, panel 3 is skipped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    L_col = resolve_L_col(df, preferred=L_col)

    a = resolve_alpha_cols(df)
    bm = resolve_beta_mass_cols(df)
    bs = resolve_beta_star_cols(df)  # may be None
    sc = resolve_scatter_cols(df)

    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {"S1": "C0", "S2": "C1", "S3": "C2", "S4": "C3"}

    # build panels dynamically (so it doesn't crash on missing beta_s)
    panels: list[tuple[str, str]] = [
        (a["sd"],  r"$\sigma_{\alpha_p}$"),
        (bm["sd"], r"$\sigma_{\beta_p}$"),
    ]
    if bs is not None:
        panels.append((bs["sd"], r"$\sigma_{\beta_s}$"))
    panels.append((sc["sd"], r"$\sigma_{\varepsilon}$"))

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if len(sub) < 3:
            continue

        L_all = sub[L_col].to_numpy(float)
        labels = sub["class_label"].to_numpy(str)

        # choose layout
        n_pan = len(panels)
        nrows = 2
        ncols = int(np.ceil(n_pan / nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 5.8), sharex=True)
        axes = np.atleast_1d(axes).ravel()

        fig.suptitle(rf"Fixed $N={N0}$: Posterior uncertainties vs. Leverage", fontsize=12)

        for ax, (col, ylabel) in zip(axes, panels):
            y_all = sub[col].to_numpy(float)
            _scatter_with_fits(
                ax,
                L_all,
                y_all,
                labels,
                class_order,
                class_colors,
                ylabel,
                L_label_tex=rf"$L_M$",
            )

        # hide any unused axes
        for ax in axes[len(panels):]:
            ax.axis("off")

        # legend on first visible axis
        handles, leg_labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color=class_colors.get(cls, "k"))
                handles.append(h)
                leg_labels.append(cls)
        if handles:
            axes[0].legend(handles, leg_labels, title="class", fontsize=8, title_fontsize=9, frameon=False, loc="best")

        fig.tight_layout()
        out_file = out_dir / f"met_fixedN_N{N0}_uncertainty_vs_{L_col}.pdf"
        fig.savefig(out_file)
        plt.close(fig)


def make_met_fixedN_scatter_mean_vs_L_from_df(
    df: pd.DataFrame,
    out_dir: str | Path = "plots",
    L_col: str = "L_logM",
) -> None:
    """
    For each N, show how posterior mean intrinsic scatter changes with leverage.
    Schema-robust: uses epsilon_mean (MetModel) or sigma_mean (1D).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    L_col = resolve_L_col(df, preferred=L_col)
    sc = resolve_scatter_cols(df)

    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {"S1": "C0", "S2": "C1", "S3": "C2", "S4": "C3"}

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if len(sub) < 3:
            continue

        L_all = sub[L_col].to_numpy(float)
        labels = sub["class_label"].to_numpy(str)
        y_all = sub[sc["mean"]].to_numpy(float)

        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2), sharex=True)
        fig.suptitle(rf"Fixed $N={N0}$: intrinsic scatter vs. leverage", fontsize=12)

        _scatter_with_fits(
            ax,
            L_all,
            y_all,
            labels,
            class_order,
            class_colors,
            r"$\varepsilon$",
            L_label_tex=rf"$L_(log(M)$",
        )

        handles, leg_labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D([], [], linestyle="none", marker="o", markersize=5, color=class_colors.get(cls, "k"))
                handles.append(h)
                leg_labels.append(cls)
        if handles:
            ax.legend(handles, leg_labels, title="class", fontsize=8, title_fontsize=9, frameon=False, loc="best")

        fig.tight_layout()
        out_file = out_dir / f"met_fixedN_N{N0}_scatter_mean_vs_{L_col}.pdf"
        fig.savefig(out_file)
        plt.close(fig)


def make_met_global_slope_3d_from_df(
    df: pd.DataFrame,
    out_path: str | Path = "plots/met_slope_plane.pdf",
    L_col: str = "L_logM",
) -> None:
    """
    3D view in (beta_p, beta_s, alpha_p) space (MetModel).
    Falls back gracefully if beta_s is absent.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_path = Path(out_path)
    _ensure_dir_for(out_path)

    L_col = resolve_L_col(df, preferred=L_col)

    a = resolve_alpha_cols(df)
    bm = resolve_beta_mass_cols(df)
    bs = resolve_beta_star_cols(df)
    if bs is None:
        raise ValueError("make_met_global_slope_3d_from_df requires beta_s_* columns (MetModel output).")

    beta_m = df[bm["mean"]].to_numpy(float)
    beta_s = df[bs["mean"]].to_numpy(float)
    alpha  = df[a["mean"]].to_numpy(float)
    L      = df[L_col].to_numpy(float)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    sca = ax.scatter(beta_m, beta_s, alpha, c=L, s=35, alpha=0.9)

    ax.set_xlabel(r"$\beta_p$ (slope on $\log M$)")
    ax.set_ylabel(r"$\beta_s$ (slope on stellar metallicity)")
    ax.set_zlabel(r"$\alpha_p$ (intercept)")
    ax.set_title(r"Survey posteriors in $(\beta_p,\beta_s,\alpha_p)$ space")

    cbar = fig.colorbar(sca, ax=ax, pad=0.1)
    cbar.set_label(rf"$L_(log_M)$")

    # plane fit: alpha ≈ a0 + a1*beta_p + a2*beta_s
    X = np.vstack([np.ones_like(beta_m), beta_m, beta_s]).T
    coef, *_ = np.linalg.lstsq(X, alpha, rcond=None)
    a0, a1, a2 = coef

    bm_lin = np.linspace(beta_m.min(), beta_m.max(), 20)
    bs_lin = np.linspace(beta_s.min(), beta_s.max(), 20)
    BM, BS = np.meshgrid(bm_lin, bs_lin)
    A_plane = a0 + a1 * BM + a2 * BS

    ax.plot_surface(BM, BS, A_plane, alpha=0.2, linewidth=0, antialiased=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---- your histogram function stays unchanged (left as-is) ----
def plot_mass_histogram_nested_classes(
    sampler: "SurveySampler",
    bins: int = 30,
):
    """
    Plot a single overlaid histogram of logM for S1–S4.
    """
    mass_classes = sampler.mass_classes
    subset_order = ["S1", "S2", "S3", "S4"]
    colors = {"S1": "blue", "S2": "orange", "S3": "green", "S4": "red"}

    data_arrays = [mass_classes[label]["logM"].to_numpy(float) for label in subset_order]

    xmin = min(np.min(arr) for arr in data_arrays)
    xmax = max(np.max(arr) for arr in data_arrays)
    bin_edges = np.linspace(xmin, xmax, bins)

    fig, ax = plt.subplots(figsize=(8, 6))

    for label, arr in zip(subset_order, data_arrays):
        ax.hist(
            arr,
            bins=bin_edges,
            density=True,
            alpha=0.5,
            label=f"{label} (n={len(arr)})",
            color=colors[label],
        )

    ax.set_xlabel("logM")
    ax.set_ylabel("Density")
    ax.set_title("Mass Distribution by Nested Survey Class")
    ax.legend()
    fig.tight_layout()

    return fig, ax
