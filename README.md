# gwex_py

A Python translation of the [GWEX R package](https://github.com/guillaumeevin/GWEX) (Evin et al. 2018) — a multi-site stochastic weather generator for precipitation. This implementation targets the Aburrá Valley hydrology project and covers precipitation only; temperature is not implemented.

---

## Table of Contents

1. [What is GWEX?](#1-what-is-gwex)
2. [Statistical Model](#2-statistical-model)
3. [Module Overview](#3-module-overview)
4. [How the Modules Work Together](#4-how-the-modules-work-together)
5. [Input Data Requirements](#5-input-data-requirements)
6. [Outputs](#6-outputs)
7. [What You Learn from Running GWEX](#7-what-you-learn-from-running-gwex)
8. [Quick-Start Usage](#8-quick-start-usage)
9. [Key Function Reference](#9-key-function-reference)

---

## 1. What is GWEX?

GWEX is a **stochastic weather generator** — a statistical model that learns the behaviour of rainfall from observed records and then generates arbitrarily long synthetic time series that share the same statistical properties. It operates simultaneously at **multiple gauging stations**, preserving not just the marginal distribution at each site but also the **spatial correlation** between sites and **temporal autocorrelation** over time.

Primary use cases in the Aburrá Valley project:

- **Extending short records**: generate thousands of years of synthetic rainfall to estimate extreme events (1-in-100, 1-in-1000 year) that cannot be observed directly.
- **IDF curve derivation**: pool synthetic extremes across many realisations for robust frequency analysis.
- **Ensemble inputs for rainfall-runoff models**: feed multiple stochastic scenarios into SWAT or similar to quantify uncertainty in flood peaks.
- **Climate sensitivity testing**: condition the generator on ENSO state (La Niña / El Niño / neutral) and examine how the tail of the distribution shifts.

---

## 2. Statistical Model

GWEX decomposes the multi-site precipitation process into three separable components, each estimated independently and then recombined during simulation.

### 2.1 Occurrence — Markov Chain

Whether it rains at any site on any day is modelled as a **first-order Markov chain**. Each calendar month has its own pair of transition probabilities:

- `p01(m)` — probability of rain today given dry yesterday
- `p11(m)` — probability of rain today given wet yesterday

Spatial dependence between sites is introduced through a **Gaussian copula** whose correlation matrix `Ω` is inferred from the empirical tetrachoric (polychoric) correlations of the binary wet/dry sequences.

### 2.2 Amount — Extended Generalised Pareto Distribution (EGPD)

On wet days, the rainfall amount at each site is modelled with the **EGPD** (Naveau et al. 2016):

```
F(y; κ, σ, ξ) = G(y/σ; ξ)^κ
```

where `G` is the standard GPD CDF, `κ > 0` is a shape parameter that controls the body of the distribution, `σ > 0` is a scale parameter, and `ξ` is the tail index (GPD shape). This family smoothly interpolates between thin-tailed and heavy-tailed regimes and avoids the discontinuity at zero that plagues mixed discrete–continuous models. Parameters are fitted separately for each calendar month by **probability-weighted moments (PWM)**.

As a fallback for sites with sparse data the package also supports a **mixture of two exponentials** parameterised by `(prob, rate1, rate2)`.

### 2.3 Spatial Dependence — Gaussian / Student Copula

Spatial co-occurrence and co-amounts are captured through a **latent Gaussian field**. On each day a correlated Gaussian vector `Z ~ N(0, Ω)` is drawn; its components are transformed to uniform marginals and then inverted through each site's fitted CDF to produce correlated rainfall amounts. The correlation matrices `Ω_occ` (for occurrence) and `Ω_amt` (for amount) are estimated separately.

An optional **Student-t copula** replaces the Gaussian for heavier joint tails.

### 2.4 Temporal Autocorrelation — MAR(1)

Short-range persistence in rainfall amounts is captured by a **multivariate autoregressive model of order 1** (MAR(1)) applied to the uniform-scale (copula) residuals. This reproduces multi-day wet-spell structure. Autocorrelation matrices `A` and `B` satisfy `A A' + B B' = I` and are estimated from lagged cross-correlations of the observed uniform residuals.

---

## 3. Module Overview

```
gwex_py/
├── __init__.py        Public API
├── core.py            Entry points: fit_gwex_model, sim_gwex_model; dataclasses
├── precipitation.py   All precipitation fitting and simulation logic
├── distributions.py   EGPD and mixture-of-exponentials distributions
├── simulation.py      Low-level Markov chain and disaggregation routines
└── utils.py           Shared helpers: transition probs, correlation repair, CDFs
```

### `core.py` — Orchestrator

Defines the three principal **data containers** and the two **user-facing entry points**.

**Dataclasses:**

| Class | Purpose | Key fields |
|---|---|---|
| `GwexObs` | Holds observed data | `obs` (T×S array), `date` (T datetime64 array), `type_var` |
| `GwexFit` | Holds fitted parameters | `fit` (nested dict), `options` (dict), `type_var` |
| `GwexSim` | Holds simulation output | `sim` (T×S×N array), `date`, `options`, `type_var` |

**Entry points:**

`fit_gwex_model(obs, options)` — validates options, delegates to `fit_GWex_prec()` in `precipitation.py`, wraps the result in a `GwexFit`.

`sim_gwex_model(fit, n_sim, d_start, d_end, seed, obs)` — builds a date vector, runs `sim_GWex_prec_1it()` for each of `n_sim` realisations, stacks results into `GwexSim`.

### `precipitation.py` — Core Model

The largest module (≈1 475 lines). Contains every function that touches precipitation physics.

**Fitting pipeline:**

| Function | What it does |
|---|---|
| `fit_GWex_prec` | Top-level fitting coordinator |
| `infer_mat_omega` | Estimates the occurrence copula matrix `Ω_occ` via tetrachoric correlation inversion |
| `get_M0` | Computes the concurrent cross-correlation matrix of standardised residuals |
| `infer_dep_amount` | Estimates the amount copula matrix `Ω_amt` |
| `infer_autocor_amount` | Estimates the MAR(1) matrices `A` and `B` |
| `fit_margin_cdf` | Fits EGPD (or mixExp) parameters month-by-month at each site |

**Simulation pipeline:**

| Function | What it does |
|---|---|
| `sim_GWex_prec_1it` | One full realisation: calls occurrence then amount simulators |
| `sim_GWex_occ` | Simulates the wet/dry sequence using the Markov chain |
| `sim_GWex_Yt` | Simulates uniform-scale latent amounts via MAR(1) or iid Gaussian |
| `sim_GWex_Yt_Pr` | Transforms uniform amounts to physical rainfall via inverse CDF |
| `mask_GWex_Yt` | Applies the occurrence mask to zero out dry days |

### `distributions.py` — Statistical Distributions

Self-contained implementations of the two marginal distribution families, plus PWM fitting.

| Function group | Description |
|---|---|
| `pdf_egpd_gi`, `cdf_egpd_gi`, `ppf_egpd_gi`, `rvs_egpd_gi` | EGPD pdf / CDF / quantile / random samples |
| `fit_egpd_gi_pwm` | Fits EGPD by probability-weighted moments |
| `pdf_mixexp`, `cdf_mixexp`, `ppf_mixexp`, `rvs_mixexp` | Mixture-of-exponentials pdf / CDF / quantile / random samples |
| `fit_mixexp` | Fits mixture-of-exponentials by MLE |
| `fit_margin_cdf` | Selects EGPD or mixExp, fits month-by-month, returns parameter array |
| `unif_to_prec` | Converts uniform-scale values to precipitation amounts via fitted CDF inverse |
| `pwm_moment` | Probability-weighted moment estimator (replaces R `EnvStats::pwMoment`) |

### `simulation.py` — Markov Chain Engine

Low-level NumPy translation of the C++/Rcpp routines in the original `src/toolsSimDisag.cpp`.

| Function | Description |
|---|---|
| `sim_markov_chain` | Simulates a single-site binary Markov chain given `p01`, `p11` |
| `cor_markov_chain` | Generates spatially correlated Markov chains via a shared latent Gaussian |
| `sim_cond_multinorm` | Draws from a conditional multivariate normal (used in MAR(1) step) |
| `disaggregate_3day` | Temporal disaggregation helper for sub-daily downscaling (future use) |

### `utils.py` — Shared Utilities

Stateless helper functions used by multiple modules.

| Function | Description |
|---|---|
| `get_transition_probs` | Computes empirical `p01` / `p11` from binary wet/dry series |
| `get_empir_cdf` | Computes empirical CDF with Weibull plotting positions |
| `repair_cor_matrix` | Projects a near-positive-definite matrix to the nearest valid correlation matrix |
| `get_seasonal_periods` | Returns month-grouping indices for bimodal or other seasonal structures |
| `validate_options` | Checks the options dict for required keys and valid value ranges |
| `mat_omega_to_cor` | Converts a copula correlation matrix to Pearson correlations |

---

## 4. How the Modules Work Together

### Fitting call hierarchy

```
fit_gwex_model(obs, options)          ← core.py
└── fit_GWex_prec(obs, options)       ← precipitation.py
    ├── fit_margin_cdf(obs, ...)      ← distributions.py (via precipitation.py)
    │   ├── fit_egpd_gi_pwm(...)      ← distributions.py
    │   └── fit_mixexp(...)           ← distributions.py
    ├── get_transition_probs(obs)     ← utils.py
    ├── infer_mat_omega(obs, ...)     ← precipitation.py
    │   └── repair_cor_matrix(...)   ← utils.py
    ├── get_M0(obs, ...)              ← precipitation.py
    ├── infer_dep_amount(obs, ...)    ← precipitation.py
    │   └── repair_cor_matrix(...)   ← utils.py
    └── infer_autocor_amount(obs)     ← precipitation.py
```

### Simulation call hierarchy

```
sim_gwex_model(fit, n_sim, ...)       ← core.py
└── sim_GWex_prec_1it(fit, dates)    ← precipitation.py
    ├── sim_GWex_occ(fit, dates)     ← precipitation.py
    │   └── cor_markov_chain(...)    ← simulation.py
    ├── sim_GWex_Yt(fit, dates)      ← precipitation.py
    │   └── sim_cond_multinorm(...)  ← simulation.py
    ├── sim_GWex_Yt_Pr(fit, Yt)     ← precipitation.py
    │   └── unif_to_prec(...)        ← distributions.py
    └── mask_GWex_Yt(occ, Yt_Pr)    ← precipitation.py
```

Data flow summary:

1. `core.py` receives a `GwexObs` object and passes `.obs` (numpy array) down to `precipitation.py`.
2. `precipitation.py` calls `utils.py` for transition probabilities and matrix repair, and `distributions.py` for marginal fitting.
3. All fitted parameters are returned as a nested Python dict, wrapped in `GwexFit` by `core.py`.
4. During simulation, `core.py` unpacks `fit.fit` and passes the dict to `precipitation.py`.
5. `precipitation.py` calls `simulation.py` for Markov chain draws and `distributions.py` for CDF inversion.
6. Final output is a 3-D NumPy array `(T, S, N)` wrapped in `GwexSim`.

---

## 5. Input Data Requirements

### Observed precipitation array

| Property | Requirement |
|---|---|
| Type | `np.ndarray`, shape `(T, S)` — T days × S stations |
| Units | Any consistent unit (mm/day recommended) |
| Missing values | `np.nan` is accepted; heavily gapped records degrade fit quality |
| Dry-day threshold | Days with `obs <= threshold` (default 0.2 mm) are treated as dry |
| Minimum record length | At least 5–10 years per station recommended for reliable parameter estimation |
| Seasonality | At least one full annual cycle required; bimodal regimes (like Aburrá Valley) need ≥ 5 years to resolve both wet seasons |

### Date array

A 1-D `np.ndarray` of `datetime64[D]` values aligned to rows of the observation array. Used to extract calendar months for seasonal parameterisation.

### Options dictionary

```python
options = {
    "typevar":        "Prec",        # must be "Prec"
    "th":             0.2,           # wet-day threshold in same units as obs
    "type.margin":    "EGPD",        # "EGPD" or "mixExp"
    "type.dep":       "Gaussian",    # "Gaussian" or "Student"
    "autocor":        True,          # include MAR(1) temporal autocorrelation
    "nlag":           1,             # MAR lag order (1 is standard)
    "nb.obs.month":   20,            # min wet-day observations per month to fit EGPD
}
```

### Constructing `GwexObs`

```python
import numpy as np
from gwex_py import GwexObs

obs = np.load("precip_daily_aburra.npy")   # shape (T, S)
dates = np.arange(
    np.datetime64("1990-01-01"),
    np.datetime64("2020-01-01"),
    dtype="datetime64[D]"
)

gwex_obs = GwexObs(obs=obs, date=dates, type_var="Prec")
```

---

## 6. Outputs

### `GwexFit` — fitted parameter object

`fit.fit` is a nested dict with the following top-level keys:

| Key | Content |
|---|---|
| `"p01"` | Array (12, S) — monthly dry-to-wet transition probabilities |
| `"p11"` | Array (12, S) — monthly wet-to-wet transition probabilities |
| `"omega_occ"` | Array (S, S) — occurrence copula correlation matrix |
| `"omega_amt"` | Array (S, S) — amount copula correlation matrix |
| `"parMargin"` | Array (12, S, 3) — monthly marginal parameters (κ, σ, ξ for EGPD) |
| `"typeMargin"` | List (S,) — marginal model chosen per station ("EGPD" or "mixExp") |
| `"A"` | Array (S, S) — MAR(1) coefficient matrix (if `autocor=True`) |
| `"B"` | Array (S, S) — MAR(1) noise scaling matrix (if `autocor=True`) |

### `GwexSim` — simulation output object

| Attribute | Description |
|---|---|
| `sim.sim` | NumPy array `(T, S, N)` — T simulated days × S stations × N realisations |
| `sim.date` | DateRange of the simulated period |
| `sim.options` | Options dict used for the run |
| `sim.type_var` | "Prec" |

Each slice `sim.sim[:, :, i]` is one independent synthetic daily precipitation record with the same statistical properties as the observations.

---

## 7. What You Learn from Running GWEX

Running GWEX on your observed record answers the following questions:

**About rainfall occurrence:**
- How likely is it to rain at each station in each calendar month?
- If it rains today, how much more likely is it to rain tomorrow? (persistence)
- How correlated are wet/dry states across the station network?

**About rainfall amounts:**
- What is the full marginal distribution of wet-day amounts, including the heavy tail, at each station in each month?
- How well does an EGPD or mixture-of-exponentials fit the observed amounts?

**About spatial structure:**
- How strongly are rainfall amounts co-varying across the network?
- Which pairs of stations share the most spatially coherent rainfall signals? (useful for regionalisation)

**About temporal structure:**
- Is there significant multi-day autocorrelation in rainfall amounts beyond what the Markov chain captures?

**From the synthetic ensemble:**
- What are the T-year return-level estimates for daily totals at each station and for spatially averaged totals?
- What is the uncertainty range around those return levels?
- What is the joint probability of exceeding a threshold simultaneously at multiple stations (relevant for catchment-wide flood risk)?
- How does an ENSO-conditioned run shift the frequency of extreme events? (if conditioning is applied externally by splitting observations by ENSO state before fitting)

---

## 8. Quick-Start Usage

### Fit the model

```python
import numpy as np
from gwex_py import GwexObs, fit_gwex_model

# Load data
obs_array = np.load("precip_mm_day.npy")   # shape (T, S)
dates = np.load("dates.npy").astype("datetime64[D]")

obs = GwexObs(obs=obs_array, date=dates, type_var="Prec")

options = {
    "typevar":     "Prec",
    "th":          0.2,
    "type.margin": "EGPD",
    "type.dep":    "Gaussian",
    "autocor":     True,
    "nlag":        1,
    "nb.obs.month": 20,
}

fit = fit_gwex_model(obs, options)
```

### Simulate synthetic records

```python
from gwex_py import sim_gwex_model

result = sim_gwex_model(
    fit=fit,
    n_sim=100,                              # 100 realisations
    d_start=np.datetime64("2000-01-01"),
    d_end=np.datetime64("2099-12-31"),      # 100 years each
    seed=42,
    obs=obs,                                # optional: used for conditioning checks
)

# result.sim has shape (T, S, 100)
synthetic_precip = result.sim              # np.ndarray
```

### Inspect fitted parameters

```python
import pandas as pd

p01 = fit.fit["p01"]         # (12, S) — monthly dry-to-wet probabilities
p11 = fit.fit["p11"]         # (12, S) — monthly wet-to-wet probabilities
par = fit.fit["parMargin"]   # (12, S, 3) — EGPD kappa, sigma, xi per month per site

# Example: wet-season persistence at site 0
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
df = pd.DataFrame({"p01": p01[:, 0], "p11": p11[:, 0]}, index=months)
print(df)
```

### Extract extremes from the ensemble

```python
from scipy.stats import genextreme

# Annual maxima at station 0 across all realisations
ann_max = result.sim[:, 0, :].reshape(100, 365, 100).max(axis=1)  # adjust for actual year length

# Fit GEV to pooled annual maxima
shape, loc, scale = genextreme.fit(ann_max.ravel())
T100 = genextreme.ppf(1 - 1/100, shape, loc, scale)
print(f"100-year daily rainfall at station 0: {T100:.1f} mm")
```

---

## 9. Key Function Reference

### `core.py`

```python
fit_gwex_model(obs: GwexObs, options: dict) -> GwexFit
```
Fits a GWEX precipitation model to observed data. Returns a `GwexFit` containing all estimated parameters.

```python
sim_gwex_model(
    fit: GwexFit,
    n_sim: int = 1,
    d_start: Optional[np.datetime64] = None,
    d_end:   Optional[np.datetime64] = None,
    seed:    Optional[int] = None,
    obs:     Optional[GwexObs] = None,
) -> GwexSim
```
Generates `n_sim` independent synthetic precipitation realisations over the specified date range.

### `distributions.py`

```python
fit_egpd_gi_pwm(data: np.ndarray) -> tuple[float, float, float]
```
Fits EGPD parameters `(kappa, sigma, xi)` by probability-weighted moments.

```python
ppf_egpd_gi(u: np.ndarray, kappa: float, sigma: float, xi: float) -> np.ndarray
```
Quantile function (inverse CDF) of the EGPD; maps uniform variates to rainfall amounts.

```python
fit_margin_cdf(obs: np.ndarray, dates: np.ndarray, options: dict) -> dict
```
Fits marginal distributions month-by-month for all stations; returns parameter arrays and model-type indicators.

### `precipitation.py`

```python
fit_GWex_prec(objGwexObs: GwexObs, options: dict) -> dict
```
Full fitting pipeline for multi-site precipitation. Returns the complete nested parameter dict.

```python
sim_GWex_prec_1it(
    objGwexFit: dict,
    vecDates: np.ndarray,
    myseed: int = 0,
    objGwexObs: Optional[dict] = None,
    prob_class: Optional[np.ndarray] = None,
) -> np.ndarray
```
Simulates one realisation of daily precipitation. Returns array of shape `(T, S)`.

### `utils.py`

```python
get_transition_probs(obs: np.ndarray, th: float = 0.2) -> tuple[np.ndarray, np.ndarray]
```
Returns `(p01, p11)` arrays of shape `(12, S)`.

```python
repair_cor_matrix(mat: np.ndarray) -> np.ndarray
```
Projects a matrix to the nearest valid correlation matrix (eigenvalue flooring + rescaling).

---

## References

- Evin, G., Favre, A.-C., & Hingray, B. (2018). Stochastic generation of multi-site daily precipitation focusing on extreme events. *Hydrology and Earth System Sciences*, 22(1), 655–672. https://doi.org/10.5194/hess-22-655-2018
- Naveau, P., Huser, R., Ribereau, P., & Hannart, A. (2016). Modeling jointly low, moderate, and heavy rainfall intensities without a threshold selection. *Water Resources Research*, 52(4), 2753–2769.
- Original R package: https://github.com/guillaumeevin/GWEX

---

*gwex_py is a precipitation-only Python translation of GWEX. Temperature simulation is not implemented. Developed for the Aburrá Valley hydrology project at Emergente.*
