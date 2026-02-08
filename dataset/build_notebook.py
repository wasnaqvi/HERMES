#!/usr/bin/env python3
"""Generate the HERMES Extended MetModel Analysis notebook (runs locally on Mac)."""
import json, os

def to_source(text):
    text = text.strip('\n')
    lines = text.split('\n')
    if len(lines) == 1:
        return [lines[0]]
    return [line + '\n' for line in lines[:-1]] + [lines[-1]]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": to_source(text)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "source": to_source(text),
            "execution_count": None, "outputs": []}

cells = []

# ===================== CELL 0: Title =====================
cells.append(md(r"""# HERMES Extended 3D MetModel Analysis

**Features:**
1. Toggle intrinsic scatter (free / fixed / off)
2. Multiple MCMC seeds for diversity
3. Log-space toggle
4. Two independent leverages: $L_{\mathrm{mass}}$ and $L_{\mathrm{stellar}}$
5. Z-scores per parameter per survey
6. WAIC model comparison (full vs no-scatter vs no-stellar)
7. Hierarchical MetModel extension (non-centered, shared $\varepsilon$)
8. HERMES `plots.py` style"""))

# ===================== CELL 1: Imports =====================
cells.append(code(r"""import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_likelihood
import arviz as az
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

numpyro.set_platform('cpu')
print('JAX devices:', jax.devices())"""))

# ===================== CELL 2: Load data (local) =====================
cells.append(md(r"""## 1. Load Data"""))

cells.append(code(r"""DATA_PATH = Path('hermes_synthetic_data_0.2.0.csv')
if not DATA_PATH.exists():
    DATA_PATH = Path(__file__).parent / 'hermes_synthetic_data_0.2.0.csv' if '__file__' in dir() else DATA_PATH
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Put hermes_synthetic_data_0.2.0.csv in the working directory. Tried: {DATA_PATH}')

raw_df = pd.read_csv(DATA_PATH)
print(f'Loaded {len(raw_df)} rows from {DATA_PATH}')
raw_df.head()"""))

# ===================== CELL: Config =====================
cells.append(md(r"""## 2. Configuration"""))

cells.append(code(r"""# ==================== USER CONFIGURATION ====================

MCMC_SEEDS = [321, 42, 7]

# Survey design -- more reps = more plot points
SURVEY_SEED = 42
N_GRID = [20, 30, 40, 50, 75, 100, 150]
N_REPS = 15

# MCMC settings (local Mac can handle more than Colab)
DRAWS = 800
TUNE = 800
TARGET_ACCEPT = 0.9
NUM_CHAINS = 1

# Toggle: intrinsic scatter  ('free', 'fixed', 'off')
SCATTER_MODE = 'free'
SCATTER_FIXED_VALUE = 0.3

# Toggle: log space
USE_LOG_SPACE = True

# Ground truth for z-scores (None = oracle fit)
GROUND_TRUTH = None

COMPUTE_WAIC = True

n_surveys = len(N_GRID) * 4 * N_REPS
n_models = 3 + (1 if SCATTER_MODE == 'fixed' else 0)
print(f'Surveys: {n_surveys}')
print(f'Models: {n_models} variants x {len(MCMC_SEEDS)} seeds')
print(f'Total fits: {n_surveys * n_models * len(MCMC_SEEDS)}')"""))

# ===================== CELL: Survey =====================
cells.append(md(r"""## 3. Survey Infrastructure"""))

cells.append(code(r"""class Survey:
    def __init__(self, survey_id, class_label, df):
        self.survey_id = int(survey_id)
        self.class_label = str(class_label)
        self.df = df.reset_index(drop=True)

    @property
    def n(self):
        return len(self.df)

    def leverage(self, col='logM'):
        arr = self.df[col].to_numpy(float)
        m = np.isfinite(arr)
        arr = arr[m]
        if arr.size < 2:
            return 0.0
        return float(np.sqrt(np.sum((arr - arr.mean()) ** 2)))


class SurveySampler:
    def __init__(self, raw_df, rng_seed=None):
        self.raw_df = raw_df
        self.rng = np.random.default_rng(rng_seed)
        self.mass_classes = self._build_mass_classes()

    def _build_mass_classes(self):
        df = self.raw_df
        q25, q50, q75 = df['logM'].quantile([0.25, 0.5, 0.75])
        return {
            'S1': df.copy(),
            'S2': df[df['logM'] >= q25].copy(),
            'S3': df[df['logM'] >= q50].copy(),
            'S4': df[df['logM'] >= q75].copy(),
        }

    def sample_grid(self, N_grid, n_reps_per_combo=10, class_order=None):
        if class_order is None:
            class_order = ['S1', 'S2', 'S3', 'S4']
        surveys, sid = [], 1
        for label in class_order:
            if label not in self.mass_classes:
                continue
            subset = self.mass_classes[label]
            for N in N_grid:
                if N > len(subset):
                    continue
                for _ in range(n_reps_per_combo):
                    rs = int(self.rng.integers(0, 2**32 - 1))
                    sdf = subset.sample(n=N, replace=False, random_state=rs)
                    surveys.append(Survey(sid, label, sdf))
                    sid += 1
        return surveys

print('Survey infrastructure ready.')"""))

# ===================== CELL: Models =====================
cells.append(md(r"""## 4. Model Definitions

| Model | Equation | Purpose |
|---|---|---|
| **B\_full** | $y = \alpha_p + \beta_p m_c + \beta_s s_c + \varepsilon$ | Current HERMES (free scatter) |
| **A\_no\_scatter** | $y = \alpha_p + \beta_p m_c + \beta_s s_c$ | Is scatter needed? |
| **C\_no\_stellar** | $y = \alpha_p + \beta_p m_c + \varepsilon$ | Is stellar met needed? ($\varepsilon$ absorbs stellar signal) |"""))

cells.append(code(r"""def met_model_full(*, x_m_c, x_s_obs, sig_meas_p, sig_meas_s, y_planet,
                   alpha_p_mu, alpha_p_sigma, beta_p_sigma, beta_s_sigma, epsilon_p_sigma):
    x_s_true = numpyro.sample('x_s_true', dist.Normal(x_s_obs, sig_meas_s))
    x_s_true_c = x_s_true - jnp.mean(x_s_true, axis=-1, keepdims=True)
    alpha_p = numpyro.sample('alpha_p', dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample('beta_p',  dist.Normal(0.0, beta_p_sigma))
    beta_s  = numpyro.sample('beta_s',  dist.Normal(1.0, beta_s_sigma))
    epsilon = numpyro.sample('epsilon',  dist.HalfNormal(epsilon_p_sigma))
    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon[..., None]**2)
    numpyro.sample('y_planet', dist.Normal(mu, obs_sigma), obs=y_planet)

def met_model_no_scatter(*, x_m_c, x_s_obs, sig_meas_p, sig_meas_s, y_planet,
                          alpha_p_mu, alpha_p_sigma, beta_p_sigma, beta_s_sigma, epsilon_p_sigma):
    x_s_true = numpyro.sample('x_s_true', dist.Normal(x_s_obs, sig_meas_s))
    x_s_true_c = x_s_true - jnp.mean(x_s_true, axis=-1, keepdims=True)
    alpha_p = numpyro.sample('alpha_p', dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample('beta_p',  dist.Normal(0.0, beta_p_sigma))
    beta_s  = numpyro.sample('beta_s',  dist.Normal(1.0, beta_s_sigma))
    numpyro.deterministic('epsilon', jnp.zeros(()))
    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
    numpyro.sample('y_planet', dist.Normal(mu, sig_meas_p), obs=y_planet)

def met_model_no_stellar(*, x_m_c, x_s_obs, sig_meas_p, sig_meas_s, y_planet,
                          alpha_p_mu, alpha_p_sigma, beta_p_sigma, beta_s_sigma, epsilon_p_sigma):
    alpha_p = numpyro.sample('alpha_p', dist.Normal(alpha_p_mu, alpha_p_sigma))
    beta_p  = numpyro.sample('beta_p',  dist.Normal(0.0, beta_p_sigma))
    numpyro.deterministic('beta_s', jnp.zeros(()))
    epsilon = numpyro.sample('epsilon', dist.HalfNormal(epsilon_p_sigma))
    mu = alpha_p[..., None] + beta_p[..., None] * x_m_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon[..., None]**2)
    numpyro.sample('y_planet', dist.Normal(mu, obs_sigma), obs=y_planet)

def make_met_model_fixed_scatter(fixed_value):
    fv = float(fixed_value)
    def model(*, x_m_c, x_s_obs, sig_meas_p, sig_meas_s, y_planet,
              alpha_p_mu, alpha_p_sigma, beta_p_sigma, beta_s_sigma, epsilon_p_sigma):
        x_s_true = numpyro.sample('x_s_true', dist.Normal(x_s_obs, sig_meas_s))
        x_s_true_c = x_s_true - jnp.mean(x_s_true, axis=-1, keepdims=True)
        alpha_p = numpyro.sample('alpha_p', dist.Normal(alpha_p_mu, alpha_p_sigma))
        beta_p  = numpyro.sample('beta_p',  dist.Normal(0.0, beta_p_sigma))
        beta_s  = numpyro.sample('beta_s',  dist.Normal(1.0, beta_s_sigma))
        numpyro.deterministic('epsilon', jnp.array(fv))
        mu = alpha_p[..., None] + beta_p[..., None] * x_m_c + beta_s[..., None] * x_s_true_c
        obs_sigma = jnp.sqrt(sig_meas_p**2 + fv**2)
        numpyro.sample('y_planet', dist.Normal(mu, obs_sigma), obs=y_planet)
    return model

print('Models ready.')"""))

# ===================== CELL: Fitting =====================
cells.append(md(r"""## 5. Fitting Infrastructure"""))

cells.append(code(r"""def prepare_model_kwargs(df_in, use_log_space=True):
    x_m = df_in['logM'].to_numpy(float)
    x_s_obs = df_in['Star Metallicity'].to_numpy(float)
    yp = df_in['log(X_H2O)'].to_numpy(float) if use_log_space else 10.0**df_in['log(X_H2O)'].to_numpy(float)
    el_p = df_in['uncertainty_lower'].to_numpy(float)
    eh_p = df_in['uncertainty_upper'].to_numpy(float)
    el_s = df_in['Star Metallicity Error Lower'].to_numpy(float)
    eh_s = df_in['Star Metallicity Error Upper'].to_numpy(float)
    sig_p = np.clip(0.5*(np.abs(el_p)+np.abs(eh_p)), 1e-6, None)
    sig_s = np.clip(0.5*(np.abs(el_s)+np.abs(eh_s)), 1e-6, None)
    if not use_log_space:
        sig_p = np.abs(yp) * sig_p * np.log(10)
    m = np.isfinite(x_m)&np.isfinite(x_s_obs)&np.isfinite(yp)&np.isfinite(sig_p)&np.isfinite(sig_s)
    x_m,x_s_obs,yp,sig_p,sig_s = x_m[m],x_s_obs[m],yp[m],sig_p[m],sig_s[m]
    x_m_c = x_m - float(x_m.mean())
    span_xm = max(float(np.ptp(x_m_c)),1e-3)
    span_xs = max(float(np.ptp(x_s_obs-x_s_obs.mean())),1e-3)
    span_yp = max(float(np.ptp(yp)),1e-3)
    yp_sd = max(float(np.std(yp,ddof=1)) if len(yp)>1 else 1.0, 1e-3)
    return dict(
        x_m_c=jnp.array(x_m_c,dtype=jnp.float32), x_s_obs=jnp.array(x_s_obs,dtype=jnp.float32),
        sig_meas_p=jnp.array(sig_p,dtype=jnp.float32), sig_meas_s=jnp.array(sig_s,dtype=jnp.float32),
        y_planet=jnp.array(yp,dtype=jnp.float32),
        alpha_p_mu=float(yp.mean()), alpha_p_sigma=max(float(yp_sd/np.sqrt(len(yp))),1e-3),
        beta_p_sigma=max(float(span_yp/span_xm),1e-3), beta_s_sigma=max(float(span_yp/span_xs),1e-3),
        epsilon_p_sigma=max(float(yp_sd),1e-3),
    )

def fit_model(model_fn, mkw, seed, draws, tune, ta, nchains=1, do_ll=False):
    rng_key = jax.random.PRNGKey(int(seed))
    kernel = NUTS(model_fn, target_accept_prob=float(ta))
    mcmc = MCMC(kernel, num_warmup=int(tune), num_samples=int(draws),
                num_chains=int(nchains), progress_bar=False)
    mcmc.run(rng_key, **mkw)
    if do_ll:
        post = mcmc.get_samples(group_by_chain=True)
        ll = log_likelihood(model_fn, post, **mkw)
        return az.from_numpyro(mcmc, log_likelihood=ll)
    return az.from_numpyro(mcmc)

def extract_summary(idata):
    row = {}
    for param in ['alpha_p','beta_p','beta_s','epsilon']:
        if param in idata.posterior:
            s = np.asarray(idata.posterior[param]).reshape(-1)
            row[f'{param}_mean'] = float(s.mean())
            row[f'{param}_sd'] = float(s.std(ddof=1)) if s.size>1 else 0.0
            lo,hi = np.quantile(s,[0.16,0.84])
            row[f'{param}_hdi16'] = float(lo)
            row[f'{param}_hdi84'] = float(hi)
    return row

print('Fitting infrastructure ready.')"""))

# ===================== CELL: Run =====================
cells.append(md(r"""## 6. Run Experiments"""))

cells.append(code(r"""sampler = SurveySampler(raw_df, rng_seed=SURVEY_SEED)
surveys = sampler.sample_grid(N_GRID, n_reps_per_combo=N_REPS)
print(f'Built {len(surveys)} surveys')

MODEL_VARIANTS = [
    (met_model_full, 'B_full'),
    (met_model_no_scatter, 'A_no_scatter'),
    (met_model_no_stellar, 'C_no_stellar'),
]
if SCATTER_MODE == 'fixed':
    MODEL_VARIANTS.append((make_met_model_fixed_scatter(SCATTER_FIXED_VALUE), 'D_fixed_scatter'))

PRIMARY = {'free':'B_full','off':'A_no_scatter','fixed':'D_fixed_scatter'}[SCATTER_MODE]

all_rows = []
total = len(surveys)*len(MODEL_VARIANTS)*len(MCMC_SEEDS)
count = 0

for model_fn, mname in MODEL_VARIANTS:
    print(f'\n--- {mname} ---')
    for mseed in MCMC_SEEDS:
        rng = np.random.default_rng(mseed)
        for survey in surveys:
            count += 1
            if count % 50 == 0 or count == total:
                print(f'  [{count}/{total}]', flush=True)
            rs = int(rng.integers(0,2**32-1))
            mkw = prepare_model_kwargs(survey.df, use_log_space=USE_LOG_SPACE)
            idata = fit_model(model_fn, mkw, rs, draws=DRAWS, tune=TUNE, ta=TARGET_ACCEPT,
                              nchains=NUM_CHAINS, do_ll=COMPUTE_WAIC)
            row = extract_summary(idata)
            row.update({'model':mname,'seed':mseed,'survey_id':survey.survey_id,
                        'class_label':survey.class_label,'N':survey.n,
                        'L_mass':survey.leverage('logM'),
                        'L_stellar':survey.leverage('Star Metallicity')})
            if COMPUTE_WAIC:
                try:
                    w = az.waic(idata)
                    row['waic'] = float(w.elpd_waic); row['waic_se'] = float(w.se)
                except Exception:
                    row['waic'] = np.nan; row['waic_se'] = np.nan
            all_rows.append(row)

df_results = pd.DataFrame(all_rows)
print(f'\nDone! {len(df_results)} rows.')
df_results.to_csv('hermes_extended_results.csv', index=False)"""))

# ===================== CELL: Z-scores =====================
cells.append(md(r"""## 7. Oracle Fit & Z-Scores

$z_\theta^{(k)} = (\hat\theta^{(k)} - \theta_{\rm ref}) / \sigma_\theta^{(k)}$

Well-calibrated $\Rightarrow$ z-scores $\sim \mathcal{N}(0,1)$."""))

cells.append(code(r"""if GROUND_TRUTH is None:
    print('Fitting oracle on full catalog ...')
    oracle_mkw = prepare_model_kwargs(raw_df, use_log_space=USE_LOG_SPACE)
    oracle_idata = fit_model(met_model_full, oracle_mkw, seed=0,
                             draws=DRAWS*2, tune=TUNE*2, ta=TARGET_ACCEPT, nchains=NUM_CHAINS)
    REFERENCE = {}
    for p in ['alpha_p','beta_p','beta_s','epsilon']:
        REFERENCE[p] = float(np.asarray(oracle_idata.posterior[p]).mean())
    print('Oracle:', REFERENCE)
else:
    REFERENCE = GROUND_TRUTH

for p in ['alpha_p','beta_p','beta_s','epsilon']:
    mc,sc = f'{p}_mean', f'{p}_sd'
    if mc in df_results.columns and sc in df_results.columns:
        df_results[f'z_{p}'] = (df_results[mc]-REFERENCE[p]) / df_results[sc].clip(lower=1e-10)

mask = (df_results['model']==PRIMARY) & (df_results['seed']==MCMC_SEEDS[0])
z_cols = [c for c in df_results.columns if c.startswith('z_')]
print('\nZ-score summary:')
print(df_results.loc[mask, z_cols].describe().round(3).to_string())"""))

# ===================== CELL: WAIC =====================
cells.append(md(r"""## 8. WAIC Model Comparison

$\Delta$WAIC = elpd(full) $-$ elpd(variant). Positive = full model wins."""))

cells.append(code(r"""waic_wide = None
if COMPUTE_WAIC and 'waic' in df_results.columns:
    piv = df_results.pivot_table(
        index=['survey_id','seed','class_label','N','L_mass','L_stellar'],
        columns='model', values='waic').reset_index()
    waic_wide = piv.copy()
    if 'B_full' in piv.columns and 'A_no_scatter' in piv.columns:
        waic_wide['delta_scatter'] = piv['B_full'] - piv['A_no_scatter']
    if 'B_full' in piv.columns and 'C_no_stellar' in piv.columns:
        waic_wide['delta_stellar'] = piv['B_full'] - piv['C_no_stellar']
    for col in ['delta_scatter','delta_stellar']:
        if col in waic_wide.columns:
            v = waic_wide[col].dropna()
            print(f'{col}: mean={v.mean():.2f}  median={v.median():.2f}  frac>0={((v>0).mean()):.0%}')
else:
    print('WAIC not available.')"""))

# ===================== CELL: Summary =====================
cells.append(md(r"""## 9. Results Summary"""))

cells.append(code(r"""df_prim = df_results[df_results['model']==PRIMARY].copy()
print(f'Primary model: {PRIMARY}')
sd_cols = [c for c in ['alpha_p_sd','beta_p_sd','beta_s_sd','epsilon_sd'] if c in df_prim.columns]
print('\nMean posterior SD by class:')
print(df_prim.groupby('class_label')[sd_cols].mean().round(4).to_string())"""))

# ===================== CELL: Plot utilities =====================
cells.append(md(r"""## 10. Plots"""))

cells.append(code(r"""plt.rcParams.update({
    'figure.dpi':130,'savefig.dpi':300,'font.size':11,
    'axes.titlesize':12,'axes.labelsize':11,
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':1.0,
    'xtick.direction':'out','ytick.direction':'out',
    'xtick.major.size':4,'ytick.major.size':4,
})
CLS_ORD = ['S1','S2','S3','S4']
CLS_CLR = {'S1':'C0','S2':'C1','S3':'C2','S4':'C3'}

def _pl_fit(x,y):
    x,y = np.asarray(x,float),np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    lx,ly = np.log(x[m]),np.log(y[m])
    A = np.vstack([np.ones_like(lx),lx]).T
    b = np.linalg.lstsq(A,ly,rcond=None)[0]
    return float(np.exp(b[0])),float(b[1])

def _pl_band(x,y,xg,z=1.0):
    x,y = np.asarray(x,float),np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    lx,ly = np.log(x[m]),np.log(y[m])
    A = np.vstack([np.ones_like(lx),lx]).T
    b,*_ = np.linalg.lstsq(A,ly,rcond=None)
    lxg = np.log(np.asarray(xg,float))
    mu = b[0]+b[1]*lxg
    r = ly-(A@b); dof = max(len(ly)-2,1); s2 = float(np.dot(r,r)/dof)
    Ai = np.linalg.inv(A.T@A)
    Ag = np.vstack([np.ones_like(lxg),lxg]).T
    v = np.einsum('ij,jk,ik->i',Ag,Ai,Ag)*s2
    se = np.sqrt(np.maximum(v,0.0))
    return np.exp(mu),np.exp(mu-z*se),np.exp(mu+z*se)

def _lin_fit(x,y):
    x,y = np.asarray(x,float),np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y)
    X = np.vstack([np.ones_like(x[m]),x[m]]).T
    return np.linalg.lstsq(X,y[m],rcond=None)[0]

def scatter_fits(ax,x,y,labels,ylabel_tex,xlabel_tex):
    x,y = np.asarray(x,float),np.asarray(y,float)
    m = np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
    for cls in CLS_ORD:
        cm = (labels==cls)
        if not cm.any(): continue
        ax.scatter(x[cm],y[cm],s=18,alpha=0.9,color=CLS_CLR.get(cls,'k'),label=cls)
    ax.set_xlabel(xlabel_tex); ax.set_ylabel(ylabel_tex); ax.minorticks_on()
    xf,yf = x[m],y[m]
    if xf.size<2: return
    xg = np.linspace(xf.min()*0.98,xf.max()*1.02,200)
    yh,lo,hi = _pl_band(xf,yf,xg)
    ax.fill_between(xg,lo,hi,alpha=0.15,linewidth=0)
    ax.plot(xg,yh,ls='--',lw=1.2)
    try:
        c,sl = _lin_fit(xf,yf); ax.plot(xg,c+sl*xg,ls='-.',lw=1.0)
    except Exception: pass
    a,b = _pl_fit(xf,yf)
    xr,yr = ax.get_xlim(),ax.get_ylim()
    ann = ylabel_tex + r' $\propto L^{'+f'{b:.2f}'+r'}$'
    ax.text(xr[0]+0.55*(xr[1]-xr[0]),yr[0]+0.86*(yr[1]-yr[0]),ann,fontsize=8)

def add_legend(ax,sub):
    hs,ls = [],[]
    for cls in CLS_ORD:
        if (sub['class_label']==cls).any():
            hs.append(plt.Line2D([],[],ls='none',marker='o',ms=5,color=CLS_CLR.get(cls,'k')))
            ls.append(cls)
    if hs: ax.legend(hs,ls,title='class',fontsize=8,title_fontsize=9,frameon=False,loc='best')

print('Plot utilities ready.')"""))

# ===================== CELL: beta_p vs both leverages =====================
cells.append(code(r"""df_pl = df_results[(df_results['model']==PRIMARY)&(df_results['seed']==MCMC_SEEDS[0])].copy()

for N0 in sorted(df_pl['N'].unique()):
    sub = df_pl[df_pl['N']==N0]
    if len(sub)<3: continue
    fig,axes = plt.subplots(1,2,figsize=(10.5,4))
    fig.suptitle(rf'Fixed $N={N0}$: $\sigma_{{\beta_p}}$ vs Leverage',fontsize=12)
    labels = sub['class_label'].to_numpy(str)
    scatter_fits(axes[0],sub['L_mass'].values,sub['beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$',r'$L_{\mathrm{mass}}$')
    scatter_fits(axes[1],sub['L_stellar'].values,sub['beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$',r'$L_{\mathrm{stellar}}$')
    add_legend(axes[0],sub); fig.tight_layout(); plt.show()"""))

# ===================== CELL: beta_s vs both leverages =====================
cells.append(code(r"""for N0 in sorted(df_pl['N'].unique()):
    sub = df_pl[df_pl['N']==N0]
    if len(sub)<3 or 'beta_s_sd' not in sub.columns: continue
    fig,axes = plt.subplots(1,2,figsize=(10.5,4))
    fig.suptitle(rf'Fixed $N={N0}$: $\sigma_{{\beta_s}}$ vs Leverage',fontsize=12)
    labels = sub['class_label'].to_numpy(str)
    scatter_fits(axes[0],sub['L_mass'].values,sub['beta_s_sd'].values,labels,r'$\sigma_{\beta_s}$',r'$L_{\mathrm{mass}}$')
    scatter_fits(axes[1],sub['L_stellar'].values,sub['beta_s_sd'].values,labels,r'$\sigma_{\beta_s}$',r'$L_{\mathrm{stellar}}$')
    add_legend(axes[0],sub); fig.tight_layout(); plt.show()"""))

# ===================== CELL: alpha + epsilon vs both leverages =====================
cells.append(code(r"""for N0 in sorted(df_pl['N'].unique()):
    sub = df_pl[df_pl['N']==N0]
    if len(sub)<3: continue
    fig,axes = plt.subplots(1,2,figsize=(10.5,4))
    fig.suptitle(rf'Fixed $N={N0}$: $\sigma_{{\alpha_p}}$ and $\sigma_{{\varepsilon}}$',fontsize=12)
    labels = sub['class_label'].to_numpy(str)
    if 'alpha_p_sd' in sub.columns:
        scatter_fits(axes[0],sub['L_mass'].values,sub['alpha_p_sd'].values,labels,r'$\sigma_{\alpha_p}$',r'$L_{\mathrm{mass}}$')
    if 'epsilon_sd' in sub.columns:
        scatter_fits(axes[1],sub['L_mass'].values,sub['epsilon_sd'].values,labels,r'$\sigma_{\varepsilon}$',r'$L_{\mathrm{mass}}$')
    add_legend(axes[0],sub); fig.tight_layout(); plt.show()"""))

# ===================== CELL: Z-score plots =====================
cells.append(code(r"""df_z = df_results[(df_results['model']==PRIMARY)&(df_results['seed']==MCMC_SEEDS[0])].copy()
z_params = [p for p in ['alpha_p','beta_p','beta_s','epsilon'] if f'z_{p}' in df_z.columns]

# Histograms
n_par = len(z_params)
fig,axes = plt.subplots(1,n_par,figsize=(3.8*n_par,3.5))
if n_par==1: axes=[axes]
fig.suptitle('Z-score distributions (should be ~ N(0,1))',fontsize=12)
xgrid = np.linspace(-4,4,200)
gauss = np.exp(-0.5*xgrid**2)/np.sqrt(2*np.pi)
for ax,p in zip(axes,z_params):
    vals = df_z[f'z_{p}'].dropna()
    ax.hist(vals,bins=25,density=True,alpha=0.6,edgecolor='k',lw=0.5)
    ax.plot(xgrid,gauss,'r--',lw=1.2,label='N(0,1)')
    ax.set_xlabel(f'z({p})'); ax.set_ylabel('density')
    ax.set_title(f'{p}: |z|<1 = {(np.abs(vals)<1).mean():.0%}')
    ax.legend(fontsize=7,frameon=False); ax.minorticks_on()
fig.tight_layout(); plt.show()

# Z vs both leverages
for p in z_params:
    fig,axes = plt.subplots(1,2,figsize=(10.5,4))
    fig.suptitle(rf'$z({p})$ vs Leverage',fontsize=12)
    for ax,Lcol,Llab in [(axes[0],'L_mass',r'$L_{\mathrm{mass}}$'),(axes[1],'L_stellar',r'$L_{\mathrm{stellar}}$')]:
        for cls in CLS_ORD:
            cm = df_z['class_label']==cls
            if not cm.any(): continue
            ax.scatter(df_z.loc[cm,Lcol],df_z.loc[cm,f'z_{p}'],s=18,alpha=0.7,color=CLS_CLR.get(cls,'k'),label=cls)
        ax.axhline(0,color='grey',ls='--',lw=0.8)
        ax.axhline(1,color='grey',ls=':',lw=0.6); ax.axhline(-1,color='grey',ls=':',lw=0.6)
        ax.set_xlabel(Llab); ax.set_ylabel(f'z({p})'); ax.minorticks_on()
    add_legend(axes[0],df_z); fig.tight_layout(); plt.show()"""))

# ===================== CELL: WAIC plots (redesigned) =====================
cells.append(code(r"""if waic_wide is not None:
    delta_cols = [c for c in ['delta_scatter','delta_stellar'] if c in waic_wide.columns]
    nice = {'delta_scatter':'Scatter detection: elpd(full) - elpd(no scatter)',
            'delta_stellar':'Stellar met detection: elpd(full) - elpd(no stellar)'}

    # Scatter vs both leverages with clear annotations
    for dc in delta_cols:
        fig,axes = plt.subplots(1,2,figsize=(11,4.5))
        fig.suptitle(nice.get(dc,dc),fontsize=12)
        labels = waic_wide['class_label'].to_numpy(str)
        for ax,Lcol,Llab in [(axes[0],'L_mass',r'$L_{\mathrm{mass}}$'),(axes[1],'L_stellar',r'$L_{\mathrm{stellar}}$')]:
            vals = waic_wide[dc].values; Lv = waic_wide[Lcol].values
            for cls in CLS_ORD:
                cm = labels==cls
                if not cm.any(): continue
                ax.scatter(Lv[cm],vals[cm],s=18,alpha=0.7,color=CLS_CLR.get(cls,'k'),label=cls)
            ax.axhline(0,color='k',ls='-',lw=1.0)
            ax.fill_between(ax.get_xlim(),[0,0],[-999,-999],alpha=0.05,color='red')
            ax.set_xlabel(Llab)
            ylabel = 'full model wins' + r' $\leftarrow$ $\Delta$elpd $\rightarrow$ ' + 'variant wins'
            ax.set_ylabel(ylabel,fontsize=9)
            ax.minorticks_on()
        add_legend(axes[0],waic_wide); fig.tight_layout(); plt.show()

    # Bar chart: fraction of surveys where full model wins, by N
    for dc in delta_cols:
        Ns = sorted(waic_wide['N'].unique())
        fracs_by_class = {}
        for cls in CLS_ORD:
            fracs = []
            for N0 in Ns:
                sub = waic_wide[(waic_wide['N']==N0)&(waic_wide['class_label']==cls)]
                v = sub[dc].dropna()
                fracs.append((v>0).mean() if len(v)>0 else np.nan)
            fracs_by_class[cls] = fracs

        fig,ax = plt.subplots(figsize=(8,4))
        x = np.arange(len(Ns)); w = 0.18
        for i,cls in enumerate(CLS_ORD):
            vals = fracs_by_class[cls]
            ax.bar(x+i*w, vals, w, label=cls, color=CLS_CLR[cls], alpha=0.8)
        ax.set_xticks(x+1.5*w); ax.set_xticklabels([str(n) for n in Ns])
        ax.set_xlabel('N'); ax.set_ylabel('Fraction full model wins')
        ax.set_title(nice.get(dc,dc)+': fraction of surveys where full model has higher elpd')
        ax.axhline(0.5,color='grey',ls='--',lw=0.8)
        ax.legend(fontsize=8,frameon=False); ax.set_ylim(0,1.05); ax.minorticks_on()
        fig.tight_layout(); plt.show()
else:
    print('WAIC not computed.')"""))

# ===================== CELL: Multi-seed diversity (redesigned) =====================
cells.append(code(r"""# Multi-seed: show spread of posterior MEAN across seeds for each survey
if len(MCMC_SEEDS) > 1:
    df_div = df_results[df_results['model']==PRIMARY].copy()

    # For each survey, get range of beta_p_mean across seeds
    grp = df_div.groupby('survey_id').agg(
        N=('N','first'), class_label=('class_label','first'),
        L_mass=('L_mass','first'), L_stellar=('L_stellar','first'),
        bp_mean_spread=('beta_p_mean', lambda x: x.max()-x.min()),
        bp_sd_mean=('beta_p_sd','mean'),
        bs_mean_spread=('beta_s_mean', lambda x: x.max()-x.min()),
        bs_sd_mean=('beta_s_sd','mean'),
    ).reset_index()

    # Ratio: seed spread / posterior SD  (should be << 1 if MCMC is converged)
    grp['bp_ratio'] = grp['bp_mean_spread'] / grp['bp_sd_mean'].clip(lower=1e-10)
    grp['bs_ratio'] = grp['bs_mean_spread'] / grp['bs_sd_mean'].clip(lower=1e-10)

    fig,axes = plt.subplots(1,2,figsize=(11,4.5))
    fig.suptitle(f'MCMC convergence check: seed-to-seed range / posterior SD ({len(MCMC_SEEDS)} seeds)',fontsize=11)
    labels = grp['class_label'].to_numpy(str)
    for ax,col,pname,Lcol,Llab in [
        (axes[0],'bp_ratio',r'$\beta_p$','L_mass',r'$L_{\mathrm{mass}}$'),
        (axes[1],'bs_ratio',r'$\beta_s$','L_stellar',r'$L_{\mathrm{stellar}}$'),
    ]:
        for cls in CLS_ORD:
            cm = labels==cls
            if not cm.any(): continue
            ax.scatter(grp.loc[cm,Lcol],grp.loc[cm,col],s=18,alpha=0.7,color=CLS_CLR.get(cls,'k'),label=cls)
        ax.axhline(0.1,color='green',ls='--',lw=0.8,alpha=0.6)
        ax.axhline(0.5,color='red',ls='--',lw=0.8,alpha=0.6)
        ax.set_xlabel(Llab); ax.set_ylabel(f'Seed range / posterior SD ({pname})')
        ax.set_title(f'{pname}: < 0.1 = well converged'); ax.minorticks_on()
    add_legend(axes[0],grp); fig.tight_layout(); plt.show()
else:
    print('Only 1 seed.')"""))

# ===================== CELL: Scatter toggle with BOTH leverages =====================
cells.append(code(r"""# Compare sigma(beta_p) across model variants vs BOTH leverages
seed0 = MCMC_SEEDS[0]
model_names = list(df_results['model'].unique())
n_mod = len(model_names)

for N0 in sorted(df_results['N'].unique()):
    sub = df_results[(df_results['N']==N0)&(df_results['seed']==seed0)]
    if len(sub)<3: continue

    # vs L_mass
    fig,axes = plt.subplots(1,n_mod,figsize=(4.2*n_mod,4),sharey=True)
    if n_mod==1: axes=[axes]
    fig.suptitle(rf'$N={N0}$: $\sigma_{{\beta_p}}$ by model vs $L_{{\mathrm{{mass}}}}$',fontsize=12)
    for ax,mn in zip(axes,model_names):
        msub = sub[sub['model']==mn]
        if msub.empty: ax.set_title(mn); continue
        labels = msub['class_label'].to_numpy(str)
        scatter_fits(ax,msub['L_mass'].values,msub['beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$',r'$L_{\mathrm{mass}}$')
        ax.set_title(mn,fontsize=10)
    add_legend(axes[0],sub); fig.tight_layout(); plt.show()

    # vs L_stellar
    fig,axes = plt.subplots(1,n_mod,figsize=(4.2*n_mod,4),sharey=True)
    if n_mod==1: axes=[axes]
    fig.suptitle(rf'$N={N0}$: $\sigma_{{\beta_p}}$ by model vs $L_{{\mathrm{{stellar}}}}$',fontsize=12)
    for ax,mn in zip(axes,model_names):
        msub = sub[sub['model']==mn]
        if msub.empty: ax.set_title(mn); continue
        labels = msub['class_label'].to_numpy(str)
        scatter_fits(ax,msub['L_stellar'].values,msub['beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$',r'$L_{\mathrm{stellar}}$')
        ax.set_title(mn,fontsize=10)
    add_legend(axes[0],sub); fig.tight_layout(); plt.show()"""))

# ===================== CELL: Scatter mean vs leverage =====================
cells.append(code(r"""df_sc = df_results[(df_results['model']==PRIMARY)&(df_results['seed']==MCMC_SEEDS[0])].copy()
if 'epsilon_mean' in df_sc.columns:
    for N0 in sorted(df_sc['N'].unique()):
        sub = df_sc[df_sc['N']==N0]
        if len(sub)<3: continue
        fig,axes = plt.subplots(1,2,figsize=(10.5,4))
        fig.suptitle(rf'Fixed $N={N0}$: Intrinsic scatter $\hat\varepsilon$ vs Leverage',fontsize=12)
        labels = sub['class_label'].to_numpy(str)
        scatter_fits(axes[0],sub['L_mass'].values,sub['epsilon_mean'].values,labels,r'$\hat{\varepsilon}$',r'$L_{\mathrm{mass}}$')
        scatter_fits(axes[1],sub['L_stellar'].values,sub['epsilon_mean'].values,labels,r'$\hat{\varepsilon}$',r'$L_{\mathrm{stellar}}$')
        add_legend(axes[0],sub); fig.tight_layout(); plt.show()"""))

# ===================== CELL: Hier header =====================
cells.append(md(r"""## 11. Hierarchical MetModel Extension

Same science equation as MetModel. Key improvements over v1:
- **Non-centered parameterization** (avoids Neal's funnel when $\tau \to 0$)
- **Shared $\varepsilon$** (one scatter for all surveys -- they sample the same population)
- **Data-informed hyperprior scales** (MetModel's empirical scales, but 3$\times$ wider)
- **HalfCauchy for $\tau$** (heavy tail, well-behaved near zero)
- **Precomputed stellar centering** (observed means, not latent -- avoids MCMC coupling)"""))

# ===================== CELL: Hier model =====================
cells.append(code(r"""def hier_met_model(*, survey_idx, x_m_c, x_s_obs, sig_meas_p, sig_meas_s,
                   y_planet, K, survey_sizes, xs_obs_mean_per_survey,
                   alpha_p_mu_global, yp_sd_global, beta_p_scale, beta_s_scale):
    # ---- Population hyperpriors (data-informed, weakly informative) ----
    mu_alpha = numpyro.sample('mu_alpha_p', dist.Normal(alpha_p_mu_global, 3.0*yp_sd_global))
    mu_bp    = numpyro.sample('mu_beta_p',  dist.Normal(0.0, 3.0*beta_p_scale))
    mu_bs    = numpyro.sample('mu_beta_s',  dist.Normal(1.0, 3.0*beta_s_scale))

    tau_alpha = numpyro.sample('tau_alpha_p', dist.HalfCauchy(0.5*yp_sd_global))
    tau_bp    = numpyro.sample('tau_beta_p',  dist.HalfCauchy(0.5*beta_p_scale))
    tau_bs    = numpyro.sample('tau_beta_s',  dist.HalfCauchy(0.5*beta_s_scale))

    # Shared epsilon (one for all surveys, same population)
    epsilon = numpyro.sample('epsilon', dist.HalfNormal(yp_sd_global))

    # ---- Per-survey: NON-CENTERED parameterization ----
    with numpyro.plate('surveys', K):
        a_raw  = numpyro.sample('alpha_p_raw', dist.Normal(0, 1))
        bp_raw = numpyro.sample('beta_p_raw',  dist.Normal(0, 1))
        bs_raw = numpyro.sample('beta_s_raw',  dist.Normal(0, 1))

    alpha_p = numpyro.deterministic('alpha_p', mu_alpha + tau_alpha * a_raw)
    beta_p  = numpyro.deterministic('beta_p',  mu_bp    + tau_bp    * bp_raw)
    beta_s  = numpyro.deterministic('beta_s',  mu_bs    + tau_bs    * bs_raw)

    # ---- Latent stellar met ----
    x_s_true = numpyro.sample('x_s_true', dist.Normal(x_s_obs, sig_meas_s))

    # Center using PRECOMPUTED observed means (no MCMC coupling)
    x_s_true_c = x_s_true - xs_obs_mean_per_survey[survey_idx]

    # ---- Map to observations ----
    mu = alpha_p[survey_idx] + beta_p[survey_idx]*x_m_c + beta_s[survey_idx]*x_s_true_c
    obs_sigma = jnp.sqrt(sig_meas_p**2 + epsilon**2)
    numpyro.sample('y_planet', dist.Normal(mu, obs_sigma), obs=y_planet)

print('Hierarchical MetModel defined (non-centered, shared epsilon).')"""))

# ===================== CELL: Hier data prep + fit =====================
cells.append(code(r"""def prepare_hier_data(surveys, use_log_space=True):
    all_xmc,all_xs,all_y,all_sp,all_ss = [],[],[],[],[]
    idx_list,sizes,xs_means = [],[],[]

    for k,sv in enumerate(surveys):
        df = sv.df
        xm = df['logM'].to_numpy(float); xmc = xm-xm.mean()
        xs = df['Star Metallicity'].to_numpy(float)
        y = df['log(X_H2O)'].to_numpy(float) if use_log_space else 10.0**df['log(X_H2O)'].to_numpy(float)
        sp = np.clip(0.5*(np.abs(df['uncertainty_lower'].to_numpy(float))+np.abs(df['uncertainty_upper'].to_numpy(float))),1e-6,None)
        ss = np.clip(0.5*(np.abs(df['Star Metallicity Error Lower'].to_numpy(float))+np.abs(df['Star Metallicity Error Upper'].to_numpy(float))),1e-6,None)
        if not use_log_space: sp = np.abs(y)*sp*np.log(10)
        m = np.isfinite(xmc)&np.isfinite(xs)&np.isfinite(y)&np.isfinite(sp)&np.isfinite(ss)
        all_xmc.append(xmc[m]); all_xs.append(xs[m]); all_y.append(y[m])
        all_sp.append(sp[m]); all_ss.append(ss[m])
        idx_list.extend([k]*int(m.sum()))
        sizes.append(int(m.sum()))
        xs_means.append(float(xs[m].mean()))

    y_cat = np.concatenate(all_y)
    yp_sd = max(float(np.std(y_cat,ddof=1)),1e-3)
    xm_cat = np.concatenate(all_xmc)
    xs_cat = np.concatenate(all_xs)
    span_xm = max(float(np.ptp(xm_cat)),1e-3)
    span_xs = max(float(np.ptp(xs_cat-xs_cat.mean())),1e-3)
    span_y = max(float(np.ptp(y_cat)),1e-3)

    return dict(
        survey_idx=jnp.array(idx_list,dtype=jnp.int32),
        x_m_c=jnp.array(np.concatenate(all_xmc),dtype=jnp.float32),
        x_s_obs=jnp.array(np.concatenate(all_xs),dtype=jnp.float32),
        sig_meas_p=jnp.array(np.concatenate(all_sp),dtype=jnp.float32),
        sig_meas_s=jnp.array(np.concatenate(all_ss),dtype=jnp.float32),
        y_planet=jnp.array(y_cat,dtype=jnp.float32),
        K=len(surveys), survey_sizes=jnp.array(sizes,dtype=jnp.float32),
        xs_obs_mean_per_survey=jnp.array(xs_means,dtype=jnp.float32),
        alpha_p_mu_global=float(y_cat.mean()),
        yp_sd_global=yp_sd,
        beta_p_scale=max(float(span_y/span_xm),1e-3),
        beta_s_scale=max(float(span_y/span_xs),1e-3),
    )

print('Preparing hierarchical data ...')
hier_mkw = prepare_hier_data(surveys, use_log_space=USE_LOG_SPACE)
print(f'  K={hier_mkw["K"]} surveys, N_total={len(hier_mkw["x_m_c"])} obs')

print('Fitting hierarchical model ...')
hier_rng = jax.random.PRNGKey(MCMC_SEEDS[0])
hier_kernel = NUTS(hier_met_model, target_accept_prob=TARGET_ACCEPT, max_tree_depth=12)
hier_mcmc = MCMC(hier_kernel, num_warmup=TUNE, num_samples=DRAWS,
                 num_chains=NUM_CHAINS, progress_bar=True)
hier_mcmc.run(hier_rng, **hier_mkw)
hier_idata = az.from_numpyro(hier_mcmc)
print('Done.')

hyper_params = ['mu_alpha_p','mu_beta_p','mu_beta_s','tau_alpha_p','tau_beta_p','tau_beta_s','epsilon']
print('\n=== Population posteriors ===')
for hp in hyper_params:
    if hp in hier_idata.posterior:
        s = np.asarray(hier_idata.posterior[hp]).reshape(-1)
        print(f'  {hp:18s}: {s.mean():.4f} +/- {s.std():.4f}  [{np.quantile(s,0.16):.4f}, {np.quantile(s,0.84):.4f}]')"""))

# ===================== CELL: Hier comparison =====================
cells.append(code(r"""hier_bp = np.asarray(hier_idata.posterior['beta_p'])
hier_bs = np.asarray(hier_idata.posterior['beta_s'])
hier_ap = np.asarray(hier_idata.posterior['alpha_p'])

K = hier_mkw['K']
hier_rows = []
for k in range(K):
    sv = surveys[k]
    bp_k = hier_bp[...,k].reshape(-1)
    bs_k = hier_bs[...,k].reshape(-1)
    ap_k = hier_ap[...,k].reshape(-1)
    hier_rows.append({
        'survey_id':sv.survey_id,'class_label':sv.class_label,'N':sv.n,
        'L_mass':sv.leverage('logM'),'L_stellar':sv.leverage('Star Metallicity'),
        'hier_beta_p_mean':float(bp_k.mean()),'hier_beta_p_sd':float(bp_k.std()),
        'hier_beta_s_mean':float(bs_k.mean()),'hier_beta_s_sd':float(bs_k.std()),
        'hier_alpha_p_mean':float(ap_k.mean()),'hier_alpha_p_sd':float(ap_k.std()),
    })
df_hier = pd.DataFrame(hier_rows)

df_indep = df_results[(df_results['model']==PRIMARY)&(df_results['seed']==MCMC_SEEDS[0])].copy()
df_comp = df_indep.merge(df_hier, on=['survey_id','class_label','N','L_mass','L_stellar'])

for p in ['beta_p','beta_s']:
    df_comp[f'shrinkage_{p}'] = 1.0 - np.clip(
        df_comp[f'hier_{p}_sd'].values / np.maximum(df_comp[f'{p}_sd'].values, 1e-10), 0, 1)

print(f'Mean shrinkage beta_p: {df_comp["shrinkage_beta_p"].mean():.3f}')
print(f'Mean shrinkage beta_s: {df_comp["shrinkage_beta_s"].mean():.3f}')"""))

# ===================== CELL: Hier plots =====================
cells.append(md(r"""## 12. Hierarchical Model Plots"""))

cells.append(code(r"""# Independent vs Hierarchical uncertainty, both leverages
for N0 in sorted(df_comp['N'].unique()):
    sub = df_comp[df_comp['N']==N0]
    if len(sub)<3: continue
    fig,axes = plt.subplots(2,2,figsize=(11,9))
    fig.suptitle(rf'$N={N0}$: Independent vs Hierarchical',fontsize=13)
    labels = sub['class_label'].to_numpy(str)
    scatter_fits(axes[0,0],sub['L_mass'].values,sub['beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$ (indep)',r'$L_{\mathrm{mass}}$')
    scatter_fits(axes[0,1],sub['L_mass'].values,sub['hier_beta_p_sd'].values,labels,r'$\sigma_{\beta_p}$ (hier)',r'$L_{\mathrm{mass}}$')
    scatter_fits(axes[1,0],sub['L_stellar'].values,sub['beta_s_sd'].values,labels,r'$\sigma_{\beta_s}$ (indep)',r'$L_{\mathrm{stellar}}$')
    scatter_fits(axes[1,1],sub['L_stellar'].values,sub['hier_beta_s_sd'].values,labels,r'$\sigma_{\beta_s}$ (hier)',r'$L_{\mathrm{stellar}}$')
    add_legend(axes[0,0],sub); fig.tight_layout(); plt.show()

# Shrinkage vs leverage
fig,axes = plt.subplots(1,2,figsize=(11,4.5))
fig.suptitle('Shrinkage: 0 = fully pooled, 1 = data dominates',fontsize=12)
labels = df_comp['class_label'].to_numpy(str)
for ax,Lcol,Llab,param in [
    (axes[0],'L_mass',r'$L_{\mathrm{mass}}$','shrinkage_beta_p'),
    (axes[1],'L_stellar',r'$L_{\mathrm{stellar}}$','shrinkage_beta_s')]:
    for cls in CLS_ORD:
        cm = labels==cls
        if not cm.any(): continue
        ax.scatter(df_comp.loc[cm,Lcol],df_comp.loc[cm,param],s=20,alpha=0.7,color=CLS_CLR.get(cls,'k'),label=cls)
    ax.axhline(0.5,color='grey',ls='--',lw=0.8)
    ax.set_xlabel(Llab); ax.set_ylabel(f'Shrinkage ({param.split("_")[-1]})')
    ax.set_ylim(-0.05,1.05); ax.minorticks_on()
add_legend(axes[0],df_comp); fig.tight_layout(); plt.show()

# Posterior mean: hier vs indep
fig,axes = plt.subplots(1,2,figsize=(11,4.5))
fig.suptitle('Posterior mean: Hierarchical vs Independent',fontsize=12)
for ax,param,label in [(axes[0],'beta_p',r'$\beta_p$'),(axes[1],'beta_s',r'$\beta_s$')]:
    iv = df_comp[f'{param}_mean'].values; hv = df_comp[f'hier_{param}_mean'].values
    lims = [min(iv.min(),hv.min()),max(iv.max(),hv.max())]
    ax.plot(lims,lims,'k--',lw=0.8,alpha=0.5)
    labs = df_comp['class_label'].to_numpy(str)
    for cls in CLS_ORD:
        cm = labs==cls
        if not cm.any(): continue
        ax.scatter(iv[cm],hv[cm],s=20,alpha=0.7,color=CLS_CLR.get(cls,'k'),label=cls)
    ax.set_xlabel(f'{label} (independent)'); ax.set_ylabel(f'{label} (hierarchical)'); ax.minorticks_on()
add_legend(axes[0],df_comp); fig.tight_layout(); plt.show()"""))

# ===================== ASSEMBLE NOTEBOOK =====================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": cells,
}

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "HERMES_Extended_MetModel.ipynb")
with open(outpath, "w") as f:
    json.dump(notebook, f, indent=1)
print(f"Wrote: {outpath}")
