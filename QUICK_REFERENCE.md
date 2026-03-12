# GWEX distributions.py Quick Reference

## Module Overview

Complete Python translation of EGPD (Extended Generalized Pareto Distribution) and mixture of exponentials distribution functions from R's GWEX package.

**File**: `distributions.py` (852 lines)  
**Functions**: 27 (20 main + 7 helpers)  
**Dependencies**: NumPy, SciPy

## Quick Start

```python
from distributions import pdf_egpd_gi, cdf_egpd_gi, ppf_egpd_gi, rvs_egpd_gi
from distributions import egpd_gi_fit_pwm, fit_margin_cdf, unif_to_prec
import numpy as np

# Generate random EGPD samples
samples = rvs_egpd_gi(100, kappa=1.5, sig=50.0, xi=0.1, random_state=42)

# Compute density and CDF
pdf = pdf_egpd_gi(75, kappa=1.5, sig=50.0, xi=0.1)
cdf = cdf_egpd_gi(75, kappa=1.5, sig=50.0, xi=0.1)

# Fit to data using PWM method
fit_result = egpd_gi_fit_pwm(samples)
print(f"Fitted: kappa={fit_result['kappa']:.3f}, sig={fit_result['sig']:.3f}")

# Fit distributions to precipitation data
P_mat = np.random.exponential(20, (365*10, 5))  # 10 years, 5 stations
is_period = np.ones(len(P_mat), dtype=bool)
params = fit_margin_cdf(P_mat, is_period, threshold=1.0, distribution_type='EGPD')

# Transform uniform samples to precipitation
u = np.random.uniform(0, 1, 100)
precip = unif_to_prec(u, params[0], 'EGPD')
```

## Function Categories

### EGPD Power Transforms (G(v) = v^kappa)

- `egpd_p_gi(v, kappa)` - Transform probability
- `egpd_d_gi(v, kappa)` - Derivative (density of transform)
- `egpd_q_gi(p, kappa)` - Inverse (quantile of transform)

### EGPD Distribution

- `pdf_egpd_gi(x, kappa, sig, xi)` - Probability density
- `cdf_egpd_gi(x, kappa, sig, xi)` - Cumulative probability
- `ppf_egpd_gi(p, kappa, sig, xi)` - Quantile (inverse CDF)
- `rvs_egpd_gi(n, kappa, sig, xi, random_state)` - Random samples

### EGPD PWM Moments

- `egpd_gi_mu0(kappa, sig, xi)` - 0th PWM
- `egpd_gi_mu1(kappa, sig, xi)` - 1st PWM
- `egpd_gi_mu2(kappa, sig, xi)` - 2nd PWM

### PWM Estimation & Fitting

- `egpd_gi_fit_pwm(x, xi=0.05)` - Fit EGPD using PWM method
- `egpd_gi_pwm_equations(par, pwm, xi)` - System of nonlinear equations
- `_compute_pwm(x, k)` - Helper: compute k-th PWM from data

### Mixture of Exponentials

- `pdf_mixexp(x, prob, rate1, rate2)` - PDF
- `cdf_mixexp(x, prob, rate1, rate2)` - CDF
- `ppf_mixexp(p, prob, rate1, rate2)` - Quantile function
- `rvs_mixexp(n, prob, rate1, rate2, random_state)` - Random samples
- `_fit_mixexp_em(x, max_iter, tol)` - EM algorithm for fitting

### Precipitation Utilities

- `fit_margin_cdf(P_mat, is_period, threshold, distribution_type)` - Fit marginals
- `unif_to_prec(u, params, distribution_type)` - Uniform to precipitation transform

## Parameter Meanings

### EGPD Parameters

- **kappa**: Transformation parameter (>0), controls shape. Default ~1.5
- **sig**: GPD scale parameter (>0), controls spread. Related to mean
- **xi**: GPD shape parameter, controls tail behavior. Typically 0.01-0.2

### Mixture Parameters

- **prob**: Mixture weight for first exponential (0 < prob < 1)
- **rate1**: Rate of first exponential (larger = heavier left tail)
- **rate2**: Rate of second exponential (usually smaller than rate1)

## Common Workflows

### 1. Fit EGPD to precipitation data

```python
data = np.array([10, 15, 20, 30, 45, 60, 100])  # values > threshold
fit = egpd_gi_fit_pwm(data, xi=0.05)
kappa, sig, xi = fit['x']
```

### 2. Evaluate PDF/CDF at specific points

```python
x = 50.0
density = pdf_egpd_gi(x, kappa, sig, xi)
prob = cdf_egpd_gi(x, kappa, sig, xi)
```

### 3. Generate synthetic samples

```python
synthetic = rvs_egpd_gi(n=1000, kappa=kappa, sig=sig, xi=xi)
```

### 4. Transform uniform samples to precipitation

```python
u = np.random.uniform(0, 1, 1000)
prec_samples = unif_to_prec(u, params=[kappa, sig, xi], distribution_type='EGPD')
```

### 5. Fit multiple stations at once

```python
# P_mat shape: (n_times, n_stations)
params_array = fit_margin_cdf(P_mat, is_period, threshold=1.0, 'EGPD')
# params_array shape: (n_stations, 3)
for i_st in range(n_stations):
    kappa, sig, xi = params_array[i_st]
```

## Key Formulas

### EGPD CDF

```
F(. ) = G(H(x)) = [H(x)]^kappa
where H(x) is the GPD CDF
```

### EGPD PDF

```
f(x) = h(x) * dG/dv[H(x)] = h(x) * kappa * H(x)^(kappa-1)
where h(x) is the GPD PDF
```

### PWM Moments

```
mu_0 = (sig/xi) * (kappa * beta(kappa, 1-xi) - 1)
mu_1 = (sig/xi) * (kappa * (beta(kappa, 1-xi) - beta(2*kappa, 1-xi)) - 1/2)
mu_2 = (sig/xi) * (kappa * (beta(kappa, 1-xi) - 2*beta(2*kappa, 1-xi) + beta(3*kappa, 1-xi)) - 1/3)
```

### Mixture of Exponentials CDF

```
F(x) = prob * (1 - exp(-rate1*x)) + (1-prob) * (1 - exp(-rate2*x))
```

## Return Values

### egpd_gi_fit_pwm()

Returns dict with keys:

- `'x'` - Parameter array [kappa, sig, xi]
- `'kappa'` - Estimated kappa
- `'sig'` - Estimated sig
- `'xi'` - Fixed xi parameter
- `'success'` - Convergence success (bool)
- `'nfev'` - Number of function evaluations
- `'residuals'` - Final residuals

### fit_margin_cdf()

Returns ndarray of shape (n_stations, 3):

- Each row: [param1, param2, param3]
- For EGPD: [kappa, sig, xi]
- For MixExp: [prob, rate1, rate2]

## Vectorization

All functions support NumPy vectorization:

```python
# Single value
pdf = pdf_egpd_gi(50.0, 1.5, 50, 0.1)  # scalar

# Array input
x = np.array([10, 50, 100])
pdf = pdf_egpd_gi(x, 1.5, 50, 0.1)  # array

# Broadcasting
x = np.linspace(0, 200, 100)
kappas = np.array([1.0, 1.5, 2.0])[:, None]
pdf = pdf_egpd_gi(x, kappas, 50, 0.1)  # shape (3, 100)
```

## Error Handling

```python
try:
    fit = egpd_gi_fit_pwm(data, xi=0.05)
    if not fit['success']:
        print(f"Warning: Convergence may not be optimal")
except Exception as e:
    print(f"Error during fitting: {e}")
```

## Performance Tips

1. **Batch fitting**: Use `fit_margin_cdf()` for multiple stations
2. **Caching**: Store fitted parameters to avoid repeated fitting
3. **Vectorization**: Use array operations instead of loops
4. **Random states**: Set `random_state` for reproducibility

## References

- Source R code: `GWexPrec_lib.r` (lines 450-595, 2423-2452, 1721-1730)
- Evin et al. (2019): Weather-type conditional precipitation generator
- Hosking & Wallis (1997): Regional frequency analysis with L-moments

---

For detailed documentation see:

- `TRANSLATION_NOTES.md` - R to Python mapping and design decisions
- `IMPLEMENTATION_SUMMARY.md` - Complete technical summary
- Module docstring in `distributions.py` - Full API reference
