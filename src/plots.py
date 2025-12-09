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


# ----------------------- small fits ------------------------------------------
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
        t_alpha = (
            r"Fixed $N$: Uncertainty on the intercept $\sigma_\alpha$ vs. $L$"
        )
        t_beta = r"Fixed $N$:Uncertainty on the slope $\sigma_\beta$ vs. $L$"
        t_eps = r"Fixed $N$:Uncertainty on intrinsic scatter $\sigma_\varepsilon$ vs. $L$"
    else:
        x_label = r"Number of Targets $N$"
        xvar = "N"
        t_alpha = (
            r"Fixed $L$: Uncertainty on the intercept $\sigma_\alpha$ vs. $N$"
        )
        t_beta = r"Uncertainty on the slope $\sigma_\beta$ vs. $N$"
        t_eps = r"Uncertainty on intrinsic scatter $\sigma_\varepsilon$ vs. $N$"

    # --- sigma panels ---
    _plot_sigma_vs_x(
        x,
        alpha_sd,
        symbol="\\alpha",
        title=t_alpha,
        outfile=Path("plots") / f"{out_prefix}_alpha.pdf",
        add_linear=add_linear,
        x_label=x_label,
        xvar=xvar,
        hdi_lo=sigma_alpha_lo,
        hdi_hi=sigma_alpha_hi,
        show_band=True,
        band_kind=sigma_band_kind,
        band_z=sigma_band_z,
    )

    _plot_sigma_vs_x(
        x,
        beta_sd,
        symbol="\\beta",
        title=t_beta,
        outfile=Path("plots") / f"{out_prefix}_beta.pdf",
        add_linear=add_linear,
        x_label=x_label,
        xvar=xvar,
        hdi_lo=sigma_beta_lo,
        hdi_hi=sigma_beta_hi,
        show_band=True,
        band_kind=sigma_band_kind,
        band_z=sigma_band_z,
    )

    _plot_sigma_vs_x(
        x,
        epsilon_sd,
        symbol="\\varepsilon",
        title=t_eps,
        outfile=Path("plots") / f"{out_prefix}_epsilon.pdf",
        add_linear=add_linear,
        x_label=x_label,
        xvar=xvar,
        hdi_lo=sigma_eps_lo,
        hdi_hi=sigma_eps_hi,
        show_band=True,
        band_kind=sigma_band_kind,
        band_z=sigma_band_z,
    )

    # --- optional α/β mean ± HDI panels ---
    if plot_param_hdi and (
        (alpha_mean is not None)
        and (alpha_hdi16 is not None)
        and (alpha_hdi84 is not None)
    ):
        _plot_param_with_hdi(
            x,
            np.asarray(alpha_mean, float),
            np.asarray(alpha_hdi16, float),
            np.asarray(alpha_hdi84, float),
            symbol="\\alpha",
            title=rf"Posterior $\alpha$ per survey (mean $\pm$ HDI) vs. ${xvar}$",
            outfile=Path("plots") / f"{out_prefix}_alpha_param.pdf",
            x_label=x_label,
            fit_band=True,
            band_kind=param_band_kind,
            band_z=param_band_z,
        )

    if plot_param_hdi and (
        (beta_mean is not None)
        and (beta_hdi16 is not None)
        and (beta_hdi84 is not None)
    ):
        _plot_param_with_hdi(
            x,
            np.asarray(beta_mean, float),
            np.asarray(beta_hdi16, float),
            np.asarray(beta_hdi84, float),
            symbol="\\beta",
            title=rf"Posterior $\beta$ per survey (mean $\pm$ HDI) vs. ${xvar}$",
            outfile=Path("plots") / f"{out_prefix}_beta_param.pdf",
            x_label=x_label,
            fit_band=True,
            band_kind=param_band_kind,
            band_z=param_band_z,
        )


def make_leverage_panels_from_df(
    df: pd.DataFrame,
    out_prefix: str,
    x_col: Literal["leverage", "N"] = "leverage",
):
    """
    Convenience wrapper for the HERMES fit summary DataFrame.

    df must contain columns:
      N, L_logM,
      alpha_sd, beta_sd, sigma_sd,
      (optionally) alpha_mean/beta_mean and HDIs if you want param panels.
    """
    if x_col == "leverage":
        x = df["L_logM"].to_numpy(float)
        mode = "L"
    elif x_col == "N":
        x = df["N"].to_numpy(float)
        mode = "N"
    else:
        raise ValueError("x_col must be 'leverage' or 'N'")

    make_leverage_panels_with_hdi(
        x=x,
        alpha_sd=df["alpha_sd"].to_numpy(float),
        beta_sd=df["beta_sd"].to_numpy(float),
        epsilon_sd=df["sigma_sd"].to_numpy(float),
        add_linear=False,
        mode=mode,
        prefix=out_prefix + "_",
        # hook up these if you later add them into df:
        alpha_mean=df.get("alpha_mean"),
        alpha_hdi16=df.get("alpha_hdi16"),
        alpha_hdi84=df.get("alpha_hdi84"),
        beta_mean=df.get("beta_mean"),
        beta_hdi16=df.get("beta_hdi16"),
        beta_hdi84=df.get("beta_hdi84"),
        plot_param_hdi=False,
    )


# ----------------- design space: N vs std with L contours --------------------
def make_design_space_N_vs_std(
    surveys,
    col: str = "logM",
    out_path: str | Path = "plots/design_N_vs_std_logM.png",
) -> None:
    """
    Design-space diagnostic: each point is one mock survey.

    x-axis: N (survey size)
    y-axis: std dev of `col` within that survey (e.g. logM or log(X_H2O)).
    Points are coloured by survey.class_label (S1, S2, S3, S4).
    """
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
    """
    Design-space diagnostic before running any models.

    Each point = one mock survey.
      x-axis : N (survey size)
      y-axis : std dev of `col` within that survey (e.g. logM)
    Overplotted: curves of constant leverage L, using

        L^2 = sum_i (x_i - mean)^2 ≈ (N-1) * std^2  ~ N * std^2

    so   std(N) ~ L / sqrt(N).
    """
    out_path = Path(out_path)
    _ensure_dir_for(out_path)

    N = np.array([s.n for s in surveys], float)
    std = np.array([np.nanstd(s.df[col].to_numpy(float)) for s in surveys], float)
    labels = [s.class_label for s in surveys]

    # actual leverage (for picking contour levels)
    L_actual = np.sqrt((N - 1) * std**2)

    if L_levels is None:
        qs = [0.25, 0.5, 0.75, 0.9]
        L_levels = np.quantile(L_actual[np.isfinite(L_actual)], qs)

    L_levels = list(L_levels)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # scatter by class label
    unique_labels = sorted(set(labels))
    for lab in unique_labels:
        mask = np.array([l == lab for l in labels])
        ax.scatter(N[mask], std[mask], label=lab, alpha=0.8)

    # constant-L curves: std(N) = L / sqrt(N)
    N_grid = np.linspace(max(1.0, N.min()), N.max(), 300)
    for L0 in L_levels:
        std_curve = L0 / np.sqrt(N_grid)
        ax.plot(N_grid, std_curve, linestyle="--")
        ax.text(
            N_grid[-1],
            std_curve[-1],
            f"L≈{L0:.1f}",
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlabel("N (survey size)")
    ax.set_ylabel(f"Std dev of {col}")
    ax.set_title("Survey design space: N vs std dev with leverage contours")

    # smaller, unobtrusive legend
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
    For each distinct N, make a 3-panel vector figure:
        σ_alpha vs L, σ_beta vs L, σ_ε vs L,
    using only surveys with that N, coloured by class_label.

    Now also overlays:
      - power-law fit σ ~ a L^b with 1σ band (mean band in log-space)
      - linear fit in original space, like the other leverage panels.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    # consistent class ordering & colours
    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {
        "S1": "C0",
        "S2": "C1",
        "S3": "C2",
        "S4": "C3",
    }

    y_cols   = ["alpha_sd", "beta_sd", "sigma_sd"]
    y_labels = [r"$\sigma_\alpha$", r"$\sigma_\beta$", r"$\sigma_\varepsilon$"]
    y_syms   = ["\\alpha", "\\beta", "\\varepsilon"]  # for annotations

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if sub.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharex=True)
        fig.suptitle(rf"$N = {N0}$", fontsize=12)

        L_all = sub[L_col].to_numpy(float)

        for ax, y, ylabel, sym in zip(axes, y_cols, y_labels, y_syms):
            # --- scatter, coloured by class_label ---
            for cls in class_order:
                cls_sub = sub[sub["class_label"] == cls]
                if cls_sub.empty:
                    continue
                ax.scatter(
                    cls_sub[L_col].to_numpy(float),
                    cls_sub[y].to_numpy(float),
                    s=18,
                    alpha=0.9,
                    label=cls,
                    color=class_colors.get(cls, "k"),
                )

            ax.set_xlabel(r"$L(\log M)$")
            ax.set_ylabel(ylabel)
            ax.minorticks_on()

            # --- power-law + 1σ band + linear overlay, using ALL surveys at this N ---
            y_all = sub[y].to_numpy(float)

            # mask for sensible fit (positive, finite)
            m = (
                np.isfinite(L_all)
                & np.isfinite(y_all)
                & (L_all > 0)
                & (y_all > 0)
            )
            x_fit = L_all[m]
            y_fit = y_all[m]

            if x_fit.size >= 2:
                xg = np.linspace(x_fit.min() * 0.98, x_fit.max() * 1.02, 200)

                # power-law band (mean band, z=1 ~ 1σ)
                y_hat, lo, hi = _powerlaw_band(
                    x_fit,
                    y_fit,
                    xg,
                    prediction=False,
                    z=1.0,
                )
                ax.fill_between(xg, lo, hi, alpha=0.15, linewidth=0)
                ax.plot(xg, y_hat, linestyle="--", linewidth=1.2)

                # linear overlay in original space
                try:
                    c, m_lin = _linear_fit(x_fit, y_fit)
                    ax.plot(xg, c + m_lin * xg, linestyle="-.", linewidth=1.0)
                except ValueError:
                    # not enough points after masking, just skip linear
                    pass

                # annotation from power-law fit (like _plot_sigma_vs_x)
                fr = _powerlaw_fit(x_fit, y_fit)
                xr, yr = ax.get_xlim(), ax.get_ylim()
                if fr.b < 0:
                    ann = (
                        rf"fit: $\sigma_{{{sym}}}"
                        rf"=\dfrac{{{fr.a:.2g}}}{{L^{{{-fr.b:.2f}}}}}$"
                    )
                else:
                    ann = (
                        rf"fit: $\sigma_{{{sym}}}"
                        rf"={fr.a:.2g}L^{{{fr.b:.2f}}}$"
                    )
                ax.text(
                    xr[0] + 0.62 * (xr[1] - xr[0]),
                    yr[0] + 0.85 * (yr[1] - yr[0]),
                    ann,
                    fontsize=8,
                )

        # One legend, left-most panel, small & unobtrusive
        handles, labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    color=class_colors.get(cls, "k"),
                )
                handles.append(h)
                labels.append(cls)
        if handles:
            axes[0].legend(
                handles,
                labels,
                title="class",
                fontsize=8,
                title_fontsize=9,
                frameon=False,
                loc="best",
            )

        fig.tight_layout()
        out_file = out_dir / f"fixedN_N{N0}_sigma_vs_{L_col}.pdf"
        fig.savefig(out_file)  # vector (PDF)
        plt.close(fig)


# =====================================================================
#  METALLICITY MODEL PANELS (MetModel)
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
    """
    Helper: scatter by class, then power-law + 1σ band + linear overlay.
    Intended for positive y (uncertainties, sigmas, etc.).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x_fit = x[m]
    y_fit = y[m]

    # scatter points coloured by class
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
        return  # not enough points for a fit

    xg = np.linspace(x_fit.min() * 0.98, x_fit.max() * 1.02, 200)

    # power-law band (mean, 1σ in log-space)
    y_hat, lo, hi = _powerlaw_band(
        x_fit,
        y_fit,
        xg,
        prediction=False,
        z=1.0,
    )
    ax.fill_between(xg, lo, hi, alpha=0.15, linewidth=0)
    ax.plot(xg, y_hat, linestyle="--", linewidth=1.2)

    # linear overlay
    try:
        c, m_lin = _linear_fit(x_fit, y_fit)
        ax.plot(xg, c + m_lin * xg, linestyle="-.", linewidth=1.0)
    except ValueError:
        pass

    # annotate power-law fit
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
    For each distinct N (fixed-N panels), plot *uncertainty* (posterior SD)
    vs leverage for the metallicity model:

        alpha_sd, beta_m_sd, beta_s_sd, sigma_p_sd.

    Panels are 2x2; points are coloured by survey class (S1–S4),
    with power-law + 1σ band and linear overlay in each panel.

    Columns expected in df:
        N, class_label, L_col,
        alpha_sd, beta_m_sd, beta_s_sd, sigma_p_sd.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {
        "S1": "C0",
        "S2": "C1",
        "S3": "C2",
        "S4": "C3",
    }

    panels = [
        ("alpha_sd",   r"$\sigma_{\alpha}$"),
        ("beta_m_sd",  r"$\sigma_{\beta_m}$"),
        ("beta_s_sd",  r"$\sigma_{\beta_s}$"),
        ("sigma_p_sd", r"$\sigma_{\sigma_p}$"),
    ]

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if len(sub) < 3:
            continue

        L_all = sub[L_col].to_numpy(float)
        labels = sub["class_label"].to_numpy(str)

        fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)
        fig.suptitle(
            rf"Fixed $N={N0}$: posterior uncertainties vs. leverage",
            fontsize=12,
        )

        for ax, (col, ylabel) in zip(axes.ravel(), panels):
            y_all = sub[col].to_numpy(float)
            _scatter_with_fits(
                ax,
                L_all,
                y_all,
                labels,
                class_order,
                class_colors,
                ylabel,
                L_label_tex=r"$L(\log M)$",
            )

        # one legend on the first panel
        handles, leg_labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    color=class_colors.get(cls, "k"),
                )
                handles.append(h)
                leg_labels.append(cls)
        if handles:
            axes[0, 0].legend(
                handles,
                leg_labels,
                title="class",
                fontsize=8,
                title_fontsize=9,
                frameon=False,
                loc="best",
            )

        fig.tight_layout()
        out_file = out_dir / f"met_fixedN_N{N0}_uncertainty_vs_{L_col}.pdf"
        fig.savefig(out_file)
        plt.close(fig)


def make_met_fixedN_covariance_vs_L_from_df(
    df: pd.DataFrame,
    out_dir: str | Path = "plots",
    L_col: str = "L_logM",
) -> None:
    """
    For each N, show how the *intrinsic scatter* of the planetary metallicity
    changes with leverage:

        sigma_p_mean vs L_logM.

    Written so you can add more columns later (e.g. sigma_s_mean, rho_mean).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["N"] = df["N"].astype(int)

    class_order = ["S1", "S2", "S3", "S4"]
    class_colors = {
        "S1": "C0",
        "S2": "C1",
        "S3": "C2",
        "S4": "C3",
    }

    for N0 in sorted(df["N"].unique()):
        sub = df[df["N"] == N0]
        if len(sub) < 3:
            continue

        L_all = sub[L_col].to_numpy(float)
        labels = sub["class_label"].to_numpy(str)
        y_all = sub["sigma_p_mean"].to_numpy(float)

        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2), sharex=True)
        fig.suptitle(
            rf"Fixed $N={N0}$: intrinsic scatter vs. leverage",
            fontsize=12,
        )

        _scatter_with_fits(
            ax,
            L_all,
            y_all,
            labels,
            class_order,
            class_colors,
            r"$\sigma_p$",
            L_label_tex=r"$L(\log M)$",
        )

        # legend
        handles, leg_labels = [], []
        for cls in class_order:
            if (sub["class_label"] == cls).any():
                h = plt.Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=5,
                    color=class_colors.get(cls, "k"),
                )
                handles.append(h)
                leg_labels.append(cls)
        if handles:
            ax.legend(
                handles,
                leg_labels,
                title="class",
                fontsize=8,
                title_fontsize=9,
                frameon=False,
                loc="best",
            )

        fig.tight_layout()
        out_file = out_dir / f"met_fixedN_N{N0}_covariance_vs_{L_col}.pdf"
        fig.savefig(out_file)
        plt.close(fig)


def make_met_global_slope_3d_from_df(
    df: pd.DataFrame,
    out_path: str | Path = "plots/met_slope_plane.pdf",
    L_col: str = "L_logM",
) -> None:
    """
    3D view of the *plane* being fit in parameter space.

    Each point is one survey, at coordinates

        (beta_m_mean, beta_s_mean, alpha_mean),

    colour–coded by leverage L(logM). A best-fit plane

        alpha ≈ a0 + a1 * beta_m + a2 * beta_s

    is overplotted.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    out_path = Path(out_path)
    _ensure_dir_for(out_path)

    beta_m = df["beta_m_mean"].to_numpy(float)
    beta_s = df["beta_s_mean"].to_numpy(float)
    alpha  = df["alpha_mean"].to_numpy(float)
    L      = df[L_col].to_numpy(float)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        beta_m,
        beta_s,
        alpha,
        c=L,
        s=35,
        alpha=0.9,
    )

    ax.set_xlabel(r"$\beta_m$ (slope on $\log M$)")
    ax.set_ylabel(r"$\beta_s$ (slope on stellar metallicity)")
    ax.set_zlabel(r"$\alpha$ (intercept)")
    ax.set_title(r"Survey planes in $(\beta_m,\beta_s,\alpha)$ space")

    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label(r"$L(\log M)$")

    # Fit plane: alpha ≈ a0 + a1 * beta_m + a2 * beta_s
    X = np.vstack([np.ones_like(beta_m), beta_m, beta_s]).T
    coef, *_ = np.linalg.lstsq(X, alpha, rcond=None)
    a0, a1, a2 = coef

    bm_lin = np.linspace(beta_m.min(), beta_m.max(), 20)
    bs_lin = np.linspace(beta_s.min(), beta_s.max(), 20)
    BM, BS = np.meshgrid(bm_lin, bs_lin)
    A_plane = a0 + a1 * BM + a2 * BS

    ax.plot_surface(
        BM,
        BS,
        A_plane,
        alpha=0.2,
        linewidth=0,
        antialiased=True,
    )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
