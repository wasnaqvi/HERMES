# Comparing the 2D and 3D metallicity models

This note explains what the model–comparison plots from `comparison.py` are
actually telling you, and how to read them in terms of WAIC / ELPD and
posterior uncertainties.

I’ll call

- **Model (`lin1d`)**: the original *2D* relation  
  $y_p = \log X_{\mathrm{H_2O}}$ vs. $x = \log M$ with intrinsic scatter.
- **MetModel (`met`)**: the *3D* relation with two predictors  
  $x_m = \log M$, $x_s = \mathrm{[Fe/H]}_\star$, and one response  
  $y_p = \log X_{\mathrm{H_2O}}$, with a 2×2 intrinsic covariance between
  planet and star metallicities.

Both models are fitted to the same synthetic surveys drawn from your
`hermes_synthetic_data*.csv` files.  
For each survey you compute:

- a **WAIC** value for each model, and
- the posterior standard deviations of key parameters (slopes and intrinsic
  scatter).

---

## WAIC and ELPD 

For a given model, let $p(y_i \mid \theta)$ be the likelihood for datum
$y_i$ and $\theta$ a draw from the posterior.

- The **log pointwise predictive density (lppd)** is  
  $\mathrm{lppd} = \sum_i \log \left( \frac{1}{S}\sum_{s=1}^S
  p(y_i \mid \theta^{(s)}) \right)$.

- The **effective number of parameters** for WAIC is  
  $p_{\mathrm{waic}} = \sum_i \mathrm{Var}_s\big[\log p(y_i \mid \theta^{(s)})\big]$.

- The **WAIC estimate of the expected log predictive density (ELPD)** is  
  $\mathrm{ELPD}_{\mathrm{waic}} \approx \mathrm{lppd} - p_{\mathrm{waic}}$.

- Then the **WAIC** is just.... 
  $\mathrm{WAIC} = -2\,\mathrm{ELPD}_{\mathrm{waic}}$.

So:
- **Lower WAIC is better**.

When I plot  
$\Delta\mathrm{WAIC} = \mathrm{WAIC}(\text{met}) -
\mathrm{WAIC}(\text{lin1d})$,

- $\Delta\mathrm{WAIC} < 0$ means the **3D MetModel** has better
  predictive performance on that survey.
- $\Delta\mathrm{WAIC} > 0$ means the **2D Model** is preferred.

---

## Plot 1 – $\Delta$WAIC vs leverage

**What the plot shows**

- Each point = one survey.
- $x$–axis: leverage $L(\log M)$ of the survey.
- $y$–axis:  
  $\Delta\mathrm{WAIC} = \mathrm{WAIC}(\text{met}) -
  \mathrm{WAIC}(\text{lin1d})$.
- Colour: survey size $N$.

**How to read it**

- Points **above** the dashed line at 0:  
  $\Delta\mathrm{WAIC} > 0$ → the 2D mass–only model has **lower WAIC**
  (better predictive fit) than the 3D MetModel for that survey.
- Points **below** 0:  
  $\Delta\mathrm{WAIC} < 0$ → the 3D MetModel wins.

In your figure, most surveys sit slightly **above** zero, with
$\Delta\mathrm{WAIC}$ typically between 0 and +3, and a handful dipping to
negative values (down to about −8).

**Interpretation**

1. **The 3D model is not giving a dramatic predictive advantage overall.**

   For the bulk of surveys, the difference is small (a few WAIC units).
   That means that, given your current noise levels and sample sizes,
   adding stellar metallicity as a predictor does *not* change the
   predictive log–likelihood enough to overcome the complexity penalty.

2. **Where you see clearly negative $\Delta\mathrm{WAIC}$, the 3D model *does*
   help.**

   Those are surveys where the joint structure in $(\log M,\,[\mathrm{Fe/H}],
   \log X_{\mathrm{H_2O}})$ is strong enough, and the survey is large or
   high–leverage enough, that the MetModel captures real extra signal.

3. **Positive but small $\Delta\mathrm{WAIC}$ does *not* mean the 3D model is
   “wrong”; it just means it’s not clearly better.**

   WAIC differences of $\lesssim 2$ are often considered weak evidence.
   You’d want to look at per–survey standard errors of ELPD (ArviZ
   provides this) before claiming a decisive win for either model.

4. **WAIC warnings.**

   ArviZ warns you that for some surveys the posterior variance of the log
   predictive density is large. That just says: for those small / noisy
   surveys, WAIC is a noisy estimate; PSIS–LOO (`az.loo`) would be a more
   robust comparison.

---

## Plot 2 – Relative uncertainty in intrinsic scatter

$y$–axis:  
$\mathrm{SD}(\sigma_p) / \mathrm{SD}(\sigma_{\mathrm{1D}})$  
where

- $\sigma_p$ = intrinsic planetary scatter in the **3D** MetModel,
- $\sigma_{\mathrm{1D}}$ = intrinsic scatter in the **2D** Model.

$y = 1$ (dashed line) means both models have the same posterior SD on the
intrinsic scatter; $y > 1$ means the **3D** model is *less certain*
(wider posterior) on $\sigma_p$.

**What you see**

- Ratios are clustered quite close to 1, typically between about 0.95 and
  1.2.
- Many low–leverage, small–$N$ surveys actually have **ratios > 1**:
  the 3D model is *more* uncertain about $\sigma_p$ than the 2D model.

**Interpretation**

1. **Adding an extra predictor and covariance structure costs you precision.**

   MetModel has more parameters: two slopes, an intercept, two intrinsic
   scales, and a correlation. With limited data per survey, the
   additional flexibility means the posterior on $\sigma_p$ spreads out a
   bit compared to the simpler 2D model.

2. **The 3D model is not catastrophically worse, but it mostly doesn’t
   tighten $\sigma_p$.**

   If stellar metallicity carried strong extra information about planet
   metallicity beyond what $\log M$ already captures, you’d expect the
   multi–predictor model to *reduce* the unexplained scatter and shrink
   the uncertainty on $\sigma_p$. Instead you see at best neutral, at
   worst slightly inflated uncertainties.

3. **Science takeaway:** for your synthetic population and current survey
   design, intrinsic scatter in $\log X_{\mathrm{H_2O}}$ is largely
   explained by mass alone; stellar metallicity does not yet look like a
   dominant second–order driver at the survey level.

---

## Plot 3 – Relative uncertainty in the mass slope

$y$–axis:  
$\mathrm{SD}(\beta_m) / \mathrm{SD}(\beta_{\mathrm{1D}})$,  
where

- $\beta_{\mathrm{1D}}$ = slope of $\log X_{\mathrm{H_2O}}$ vs $\log M$ in
  the 2D model,
- $\beta_m$ = mass slope in the 3D MetModel.

Again, 1 means equal precision; $>1$ means the 3D model is less certain
about the mass slope; $<1$ means adding stellar metallicity actually gave
you a tighter constraint on $\beta_m$.

**What you see**

- Ratios are mostly between about 0.9 and 1.3.
- At low leverage there is more scatter (some surveys with ratios < 1,
  some > 1).
- At high leverage, points cluster closer to 1 with a slight tendency
  above 1.

**Interpretation**

1. **No strong evidence that the 3D model pins down the mass slope better.**

   The distribution is roughly symmetric around 1; if anything, the 3D
   model often has *slightly* larger SD on $\beta_m$.

2. **This is consistent with a parameter–tradeoff picture.**

   When you allow a second predictor (stellar metallicity) with its own
   slope, the model can trade off between “explain variance with $\log M$”
   and “explain variance with $[\mathrm{Fe/H}]_\star$”. That weakens the
   constraints on each individual slope unless the data are extremely
   informative.

3. **Science takeaway:** in your synthetic data, $\log M$ already carries
   most of the predictive power for $\log X_{\mathrm{H_2O}}$. Letting the
   model also use $[\mathrm{Fe/H}]_\star$ does not dramatically sharpen
   the inferred mass slope; it just spreads information across two
   partially degenerate directions.

---

## Are we answering the “2D vs 3D” science questions?

### 1. Does the 3D model work better on the 2D data?

If the *true* generative relation for a given dataset is really 2D
($\log X_{\mathrm{H_2O}}$ depends only on $\log M$), then:

- The **2D model is correctly specified**.
- The 3D MetModel is an **over–parameterised extension** (it can collapse
  to the 2D model when $\beta_s \approx 0$ and the intrinsic covariance
  decouples).

In that regime you expect exactly what you see:

- WAIC very slightly prefers the 2D model (same fit, fewer effective
  parameters).
- Uncertainties in $\beta_m$ and $\sigma_p$ under the 3D model are similar
  or marginally larger.

So yes: these panels are **consistent with the 3D model not giving you a
big advantage when the true relation is effectively 2D**.

### 2. Does the 2D model work “sufficiently well” on the 3D data?

When the *true* generative relation is 3D (planet metallicity depends on
both mass and stellar metallicity), but you ignore stellar metallicity and
fit only the 2D model:

- The 2D model is **misspecified**, but
- It can still give decent predictive performance if most of the
  variation is really along the mass direction or if $[\mathrm{Fe/H}]_\star$
  is strongly correlated with $\log M$.

Your plots show:

- For some surveys, WAIC strongly favours the 3D model ($\Delta\mathrm{WAIC}
  \ll 0$). Those are the places where the 2D model does **not** capture
  all the structure—likely the surveys with a wide range in stellar
  metallicity at fixed mass.
- For many surveys, the difference is small, meaning the 2D approximation
  is “good enough” for predictive purposes, even though it ignores true
  3D structure.

So the answer is nuanced:

- **Globally**, MetModel is the more faithful description of the 3D
  generative process.
- **Survey–by–survey**, a blind 2D model is often adequate, but can fail
  on high–information, high–dynamic–range surveys.

---

## Could we be plotting something more informative?

Your current diagnostics are **reasonable**, but you can sharpen them:

1. **Use ELPD differences with standard errors.**

   ArviZ’s `az.waic` / `az.loo` return an `ELPDData` object with
   per–observation contributions and an overall standard error.
   For each survey:

   - plot $\Delta\mathrm{ELPD} = \mathrm{ELPD}_{\text{met}} -
     \mathrm{ELPD}_{\text{lin1d}}$ with its $\pm 2\sigma$ error bar.
   - classify surveys where 0 is within the error bar as “no clear
     winner”.

2. **Residual plots for the 2D model.**

   For each survey, fit the 2D model and then plot the posterior residuals
   in $\log X_{\mathrm{H_2O}}$ **against $[\mathrm{Fe/H}]_\star$**:

   - If you see a clear trend, that’s direct evidence that the 2D model is
     missing the stellar–metallicity dependence.
   - If residuals vs $[\mathrm{Fe/H}]_\star$ look flat, the 2D model is
     adequate for that survey.

3. **Show example 3D predictive planes.**

   For a few representative surveys (low vs high leverage, low vs high
   $N$):

   - draw a grid in $(\log M, [\mathrm{Fe/H}]_\star)$,
   - plot the posterior mean predicted $\log X_{\mathrm{H_2O}}$ under
     MetModel as a surface,
   - overlay the actual data points.

   That directly visualises the "plane you are fitting" and makes the 3D
   structure tangible.

4. **Look at the posterior of $\beta_s$.**

   Because the 2D model is nested inside the 3D model when $\beta_s = 0$,
   the posterior of $\beta_s$ is a direct diagnostic:

   - If $\beta_s$ is tightly concentrated away from 0 for many surveys,
     the 3D model is doing something important.
   - If $\beta_s$ is broad and always compatible with 0, the data are
     telling you that stellar metallicity is not yet a strong predictor at
     the survey level.

---

## Summary

- WAIC / ELPD are telling you that the **mass–only model is often
  sufficient**, and the 3D MetModel only clearly wins for a subset of
  surveys with enough information to support the extra predictor.
- Posterior uncertainty ratios confirm this: the 3D model does not
  dramatically sharpen constraints on mass slope or intrinsic scatter,
  and sometimes slightly inflates them because of extra flexibility.
- To really see the 3D structure, complement the global WAIC plots with
  residual checks and a few concrete 3D predictive surfaces for individual
  surveys.

These behaviours are exactly what you’d expect when you try to recover
mostly 2D structure with a richer but more expensive 3D model.
