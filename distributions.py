"""
GWEX distributions module.

This module provides probability weighted moments (PWM) estimation and distribution
functions for the Extended Generalized Pareto Distribution (EGPD) and mixture of
exponentials models. Translation from R's Renext and EnvStats packages.

The EGPD (Extended GPD) family is defined by applying a power transformation G(v)=v^kappa
to a Generalized Pareto Distribution (GPD). This allows more flexible fitting of
precipitation extremes while maintaining parsimony.

Key functions:
    - EGPD distribution: pdf_egpd_gi, cdf_egpd_gi, ppf_egpd_gi, rvs_egpd_gi
    - EGPD PWM: egpd_gi_mu0, egpd_gi_mu1, egpd_gi_mu2
    - EGPD fitting: egpd_gi_fit_pwm
    - Mixture of exponentials: pdf_mixexp, cdf_mixexp, ppf_mixexp, rvs_mixexp
    - Precipitation utilities: fit_margin_cdf, unif_to_prec

GPD Parameterization:
    CDF: H(x) = 1 - (1 + xi*x/sig)^(-1/xi)  for xi != 0
         H(x) = 1 - exp(-x/sig)             for xi = 0

    This matches scipy.stats.genpareto(c=xi, scale=sig, loc=0).

References:
    Evin et al. (2019). A weather-type conditional generator of global
    precipitation using neural networks. Water Resources Research, 55.

Author: Translated from Guillaume Evin's R code (GWEX package)
"""

import numpy as np
from scipy.special import beta
from scipy.stats import genpareto, uniform
from scipy.optimize import fsolve, brentq
from typing import Tuple, Union


# =============================================================================
# EGPD Power Transform Functions (G(v) = v^kappa)
# =============================================================================


def egpd_p_gi(v: Union[float, np.ndarray], kappa: float) -> Union[float, np.ndarray]:
    """
    CDF transformation for EGPD.GI family.

    Applies the power transform G(v) = v^kappa to transform probabilities.

    Parameters
    ----------
    v : float or array-like
        Probability value(s) in [0, 1]
    kappa : float
        Transformation parameter, must be > 0

    Returns
    -------
    float or ndarray
        Transformed probability v^kappa
    """
    return v ** kappa


def egpd_d_gi(v: Union[float, np.ndarray], kappa: float) -> Union[float, np.ndarray]:
    """
    Derivative of EGPD.GI power transform.

    Computes the density associated with G(v) = v^kappa:
    dG/dv = kappa * v^(kappa-1)

    Parameters
    ----------
    v : float or array-like
        Probability value(s) in [0, 1]
    kappa : float
        Transformation parameter, must be > 0

    Returns
    -------
    float or ndarray
        Derivative kappa * v^(kappa-1)
    """
    return kappa * (v ** (kappa - 1))


def egpd_q_gi(p: Union[float, np.ndarray], kappa: float) -> Union[float, np.ndarray]:
    """
    Quantile function for EGPD.GI power transform.

    Inverts G(v) = v^kappa to find v given p:
    v = p^(1/kappa)

    Parameters
    ----------
    p : float or array-like
        Probability value(s) in [0, 1]
    kappa : float
        Transformation parameter, must be > 0

    Returns
    -------
    float or ndarray
        Quantile p^(1/kappa)
    """
    return p ** (1.0 / kappa)


# =============================================================================
# Full EGPD Distribution Functions
# =============================================================================


def pdf_egpd_gi(
    x: Union[float, np.ndarray],
    kappa: float,
    sig: float,
    xi: float
) -> Union[float, np.ndarray]:
    """
    Probability density function of the EGPD.GI distribution.

    Applies the power transformation G(v) = v^kappa to GPD random variables.
    The density is: f(x) = f_GPD(x) * dG/dv(F_GPD(x))

    Parameters
    ----------
    x : float or array-like
        Quantile value(s)
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter

    Returns
    -------
    float or ndarray
        Density values
    """
    # Get GPD CDF and PDF at x
    pH = genpareto.cdf(x, c=xi, scale=sig)
    dH = genpareto.pdf(x, c=xi, scale=sig)

    # Apply chain rule: d/dx[G(F(x))] = dG/dv(F(x)) * dF/dx(x)
    dEGPD = dH * egpd_d_gi(pH, kappa)

    return dEGPD


def cdf_egpd_gi(
    x: Union[float, np.ndarray],
    kappa: float,
    sig: float,
    xi: float
) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function of the EGPD.GI distribution.

    Parameters
    ----------
    x : float or array-like
        Quantile value(s)
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter

    Returns
    -------
    float or ndarray
        CDF values in [0, 1]
    """
    pH = genpareto.cdf(x, c=xi, scale=sig)
    return egpd_p_gi(pH, kappa)


def ppf_egpd_gi(
    p: Union[float, np.ndarray],
    kappa: float,
    sig: float,
    xi: float
) -> Union[float, np.ndarray]:
    """
    Quantile function (percent point function) of the EGPD.GI distribution.

    Parameters
    ----------
    p : float or array-like
        Probability value(s) in [0, 1]
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter

    Returns
    -------
    float or ndarray
        Quantile value(s)
    """
    qG = egpd_q_gi(p, kappa)
    qH = genpareto.ppf(qG, c=xi, scale=sig)
    return qH


def rvs_egpd_gi(
    n: int,
    kappa: float,
    sig: float,
    xi: float,
    random_state: int = None
) -> np.ndarray:
    """
    Generate random samples from the EGPD.GI distribution.

    Parameters
    ----------
    n : int
        Number of samples
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    ndarray
        Array of n random samples
    """
    rng = np.random.RandomState(random_state)
    u = rng.uniform(0, 1, n)
    return ppf_egpd_gi(u, kappa, sig, xi)


# =============================================================================
# Probability Weighted Moments (PWM) for EGPD
# =============================================================================


def egpd_gi_mu0(kappa: float, sig: float, xi: float) -> float:
    """
    First PWM (moment of order 0) of the EGPD.GI distribution.

    mu_0 = (sig/xi) * (kappa * beta(kappa, 1-xi) - 1)

    Parameters
    ----------
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter, must be < 1

    Returns
    -------
    float
        PWM of order 0
    """
    return (sig / xi) * (kappa * beta(kappa, 1 - xi) - 1)


def egpd_gi_mu1(kappa: float, sig: float, xi: float) -> float:
    """
    Second PWM (moment of order 1) of the EGPD.GI distribution.

    mu_1 = (sig/xi) * (kappa * (beta(kappa, 1-xi) - beta(2*kappa, 1-xi)) - 1/2)

    Parameters
    ----------
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter, must be < 1

    Returns
    -------
    float
        PWM of order 1
    """
    return (sig / xi) * (
        kappa * (beta(kappa, 1 - xi) - beta(2 * kappa, 1 - xi)) - 0.5
    )


def egpd_gi_mu2(kappa: float, sig: float, xi: float) -> float:
    """
    Third PWM (moment of order 2) of the EGPD.GI distribution.

    mu_2 = (sig/xi) * (kappa * (beta(kappa, 1-xi) - 2*beta(2*kappa, 1-xi)
                                + beta(3*kappa, 1-xi)) - 1/3)

    Parameters
    ----------
    kappa : float
        Transformation parameter, must be > 0
    sig : float
        GPD scale parameter, must be > 0
    xi : float
        GPD shape parameter, must be < 1

    Returns
    -------
    float
        PWM of order 2
    """
    return (sig / xi) * (
        kappa * (
            beta(kappa, 1 - xi)
            - 2 * beta(2 * kappa, 1 - xi)
            + beta(3 * kappa, 1 - xi)
        )
        - 1.0 / 3.0
    )


# =============================================================================
# PWM Estimation: System of Equations and Solver
# =============================================================================


def egpd_gi_pwm_equations(
    par: np.ndarray,
    pwm: Tuple[float, float],
    xi: float
) -> np.ndarray:
    """
    System of equations for PWM-based EGPD parameter estimation.

    Returns residuals that should be zero at the solution:
    - eq[0] = mu_0(kappa, sig, xi) - pwm[0]
    - eq[1] = mu_1(kappa, sig, xi) - pwm[1]

    Parameters
    ----------
    par : ndarray
        Parameter vector [kappa, sig]
    pwm : tuple
        Target PWM values (mu0, mu1)
    xi : float
        Shape parameter (fixed)

    Returns
    -------
    ndarray
        Residual vector of length 2
    """
    kappa, sig = par[0], par[1]

    y = np.zeros(2)
    y[0] = egpd_gi_mu0(kappa, sig, xi) - pwm[0]
    y[1] = egpd_gi_mu1(kappa, sig, xi) - pwm[1]

    return y


def egpd_gi_fit_pwm(
    x: np.ndarray,
    xi: float = 0.05
) -> dict:
    """
    Fit EGPD.GI distribution using Probability Weighted Moments.

    Estimates the parameters (kappa, sig, xi) by solving the PWM equations
    using numerical optimization. The shape parameter xi is fixed.

    Parameters
    ----------
    x : ndarray
        Sample data (should be positive values above a threshold)
    xi : float, optional
        Shape parameter of the underlying GPD (default: 0.05).
        This parameter is typically fixed based on prior knowledge or
        a separate estimation procedure.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'x' : ndarray of shape (3,) containing [kappa, sig, xi]
        - 'kappa' : estimated transformation parameter
        - 'sig' : estimated GPD scale parameter
        - 'xi' : fixed shape parameter
        - 'success' : bool indicating convergence success

    Notes
    -----
    Uses scipy.optimize.fsolve with initial guesses:
    - kappa: estimated from sample variance
    - sig: estimated from sample mean

    The numerical solution may be unstable if the data does not follow
    the EGPD model well or if xi is far from optimal.
    """
    x = np.asarray(x)
    n = len(x)

    # Compute sample PWM estimates
    pwm_0 = _compute_pwm(x, k=0)
    pwm_1 = _compute_pwm(x, k=1)
    pwm = (pwm_0, pwm_1)

    # Better initial guess based on sample statistics
    # For exponential-like distribution, kappa is typically in (0.5, 3)
    # sig is related to the mean
    x_mean = np.mean(x)
    x_std = np.std(x)

    # Estimate kappa from coefficient of variation
    cv = x_std / x_mean if x_mean > 0 else 1.0
    initial_kappa = max(0.5, min(3.0, 2.0 / (cv + 0.1)))

    # sig estimate based on PWM
    initial_sig = x_mean * 0.5

    initial_guess = np.array([initial_kappa, initial_sig])

    # Solve the system
    sol, info, ier, mesg = fsolve(
        egpd_gi_pwm_equations,
        initial_guess,
        args=(pwm, xi),
        full_output=True,
        xtol=1e-8
    )

    # Check success (ier == 1 means solution found)
    success = ier == 1

    # Return in R-compatible format
    return {
        'x': np.concatenate([sol, [xi]]),
        'kappa': sol[0],
        'sig': sol[1],
        'xi': xi,
        'success': success,
        'nfev': info['nfev'],
        'residuals': info['fvec'],
    }


def _compute_pwm(x: np.ndarray, k: int = 0) -> float:
    """
    Compute the k-th Probability Weighted Moment of a sample.

    The k-th PWM is defined as:
    M_k = E[X * F(X)^k]

    where F is the empirical CDF.

    Parameters
    ----------
    x : ndarray
        Sample data, must be sorted
    k : int, optional
        Order of the PWM (default: 0)

    Returns
    -------
    float
        Estimated PWM of order k
    """
    x = np.sort(np.asarray(x))
    n = len(x)

    # Use formula: M_k = (1/n) * sum_{j=1}^{n} x_j * (j-1 choose k) / (n-1 choose k)
    # Equivalent to: M_k = (1/n) * sum_{j=1}^{n} x_j * prod_{i=0}^{k-1} (j-1-i)/(n-1-i)

    pwm = 0.0
    for j in range(n):
        # Weight: (j choose k) / (n choose k)
        if k == 0:
            weight = 1.0
        else:
            weight = 1.0
            for i in range(k):
                weight *= (j - i) / (n - 1 - i)

        pwm += weight * x[j] / n

    return pwm


# =============================================================================
# Mixture of Exponentials
# =============================================================================


def cdf_mixexp(
    x: Union[float, np.ndarray],
    prob: float,
    rate1: float,
    rate2: float
) -> Union[float, np.ndarray]:
    """
    CDF of mixture of two exponential distributions.

    F(x) = prob * (1 - exp(-rate1*x)) + (1-prob) * (1 - exp(-rate2*x))

    Parameters
    ----------
    x : float or array-like
        Quantile value(s)
    prob : float
        Mixture probability for the first exponential (0 < prob < 1)
    rate1 : float
        Rate parameter for first exponential (rate = 1/mean)
    rate2 : float
        Rate parameter for second exponential

    Returns
    -------
    float or ndarray
        CDF values
    """
    cdf1 = 1.0 - np.exp(-rate1 * x)
    cdf2 = 1.0 - np.exp(-rate2 * x)
    return prob * cdf1 + (1.0 - prob) * cdf2


def pdf_mixexp(
    x: Union[float, np.ndarray],
    prob: float,
    rate1: float,
    rate2: float
) -> Union[float, np.ndarray]:
    """
    PDF of mixture of two exponential distributions.

    f(x) = prob * rate1 * exp(-rate1*x) + (1-prob) * rate2 * exp(-rate2*x)

    Parameters
    ----------
    x : float or array-like
        Quantile value(s)
    prob : float
        Mixture probability for the first exponential (0 < prob < 1)
    rate1 : float
        Rate parameter for first exponential
    rate2 : float
        Rate parameter for second exponential

    Returns
    -------
    float or ndarray
        PDF values
    """
    pdf1 = rate1 * np.exp(-rate1 * x)
    pdf2 = rate2 * np.exp(-rate2 * x)
    return prob * pdf1 + (1.0 - prob) * pdf2


def ppf_mixexp(
    p: Union[float, np.ndarray],
    prob: float,
    rate1: float,
    rate2: float
) -> Union[float, np.ndarray]:
    """
    Quantile function (inverse CDF) of mixture of exponentials.

    Solves F(x) = p numerically for each probability value using Brent's method.

    Parameters
    ----------
    p : float or array-like
        Probability value(s) in [0, 1]
    prob : float
        Mixture probability for the first exponential (0 < prob < 1)
    rate1 : float
        Rate parameter for first exponential
    rate2 : float
        Rate parameter for second exponential

    Returns
    -------
    float or ndarray
        Quantile value(s)
    """
    p = np.atleast_1d(np.asarray(p))
    scalar_input = p.ndim == 0

    result = np.zeros_like(p, dtype=float)

    for i, pi in enumerate(p):
        if pi <= 0.0:
            result[i] = 0.0
        elif pi >= 1.0:
            result[i] = np.inf
        else:
            # Use Brent's method to find x such that F(x) = p
            # Search in a reasonable range
            try:
                result[i] = brentq(
                    lambda x: cdf_mixexp(x, prob, rate1, rate2) - pi,
                    0,
                    -np.log(1 - pi) / min(rate1, rate2) * 10
                )
            except ValueError:
                # If brentq fails, use a fallback
                result[i] = np.inf

    return result[0] if scalar_input else result


def rvs_mixexp(
    n: int,
    prob: float,
    rate1: float,
    rate2: float,
    random_state: int = None
) -> np.ndarray:
    """
    Generate random samples from mixture of exponentials.

    Parameters
    ----------
    n : int
        Number of samples
    prob : float
        Mixture probability for the first exponential
    rate1 : float
        Rate parameter for first exponential
    rate2 : float
        Rate parameter for second exponential
    random_state : int, optional
        Seed for reproducibility

    Returns
    -------
    ndarray
        Array of n random samples
    """
    rng = np.random.RandomState(random_state)
    u = rng.uniform(0, 1, n)
    return ppf_mixexp(u, prob, rate1, rate2)


# =============================================================================
# Marginal Distribution Fitting
# =============================================================================


def fit_margin_cdf(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    threshold: float,
    distribution_type: str = 'EGPD'
) -> np.ndarray:
    """
    Fit marginal distributions to precipitation data.

    Fits either EGPD or mixture of exponentials to precipitation values
    above a threshold for each station (column) in the data matrix.

    Parameters
    ----------
    P_mat : ndarray of shape (n_times, n_stations)
        Precipitation data matrix
    is_period : ndarray of shape (n_times,) and dtype bool
        Boolean mask for times in the period of interest
    threshold : float
        Precipitation threshold for filtering
    distribution_type : str, optional
        Either 'EGPD' or 'mixExp' (default: 'EGPD')

    Returns
    -------
    ndarray of shape (n_stations, 3)
        Fitted parameters for each station:
        - For EGPD: [kappa, sig, xi]
        - For mixExp: [prob, rate1, rate2]

    Raises
    ------
    ValueError
        If distribution_type is not 'EGPD' or 'mixExp'

    Notes
    -----
    Only uses precipitation values that are:
    1. In the specified period (is_period == True)
    2. Greater than the threshold
    3. Not NaN
    """
    if distribution_type not in ('EGPD', 'mixExp'):
        raise ValueError("distribution_type must be 'EGPD' or 'mixExp'")

    P_mat = np.asarray(P_mat)
    is_period = np.asarray(is_period, dtype=bool)

    # Filter to period
    P_period = P_mat[is_period, :]

    n_stations = P_mat.shape[1]
    list_out = np.zeros((n_stations, 3))

    for i_st in range(n_stations):
        P_st = P_period[:, i_st]

        # Filter: P > threshold and not NaN
        is_precip = (P_st > threshold) & ~np.isnan(P_st)
        P_nz = P_st[is_precip]

        if len(P_nz) == 0:
            # No valid data, return NaN
            list_out[i_st, :] = np.nan
        elif distribution_type == 'EGPD':
            # Fit EGPD
            fit = egpd_gi_fit_pwm(P_nz)
            list_out[i_st, :] = fit['x']
        else:  # 'mixExp'
            # Fit mixture of exponentials using EM algorithm
            fit = _fit_mixexp_em(P_nz)
            list_out[i_st, :] = fit

    return list_out


def unif_to_prec(
    u: Union[float, np.ndarray],
    params: np.ndarray,
    distribution_type: str = 'EGPD'
) -> Union[float, np.ndarray]:
    """
    Transform uniform random variables to precipitation using inverse CDF.

    Given parameters of a fitted distribution (EGPD or mixExp) and uniform
    random values, generates precipitation samples.

    Parameters
    ----------
    u : float or array-like
        Uniform random value(s) in [0, 1]
    params : ndarray of shape (3,)
        Fitted distribution parameters:
        - For EGPD: [kappa, sig, xi]
        - For mixExp: [prob, rate1, rate2]
    distribution_type : str, optional
        Either 'EGPD' or 'mixExp' (default: 'EGPD')

    Returns
    -------
    float or ndarray
        Precipitation value(s)

    Raises
    ------
    ValueError
        If distribution_type is not 'EGPD' or 'mixExp'
    """
    if distribution_type not in ('EGPD', 'mixExp'):
        raise ValueError("distribution_type must be 'EGPD' or 'mixExp'")

    if distribution_type == 'EGPD':
        return ppf_egpd_gi(u, params[0], params[1], params[2])
    else:  # 'mixExp'
        return ppf_mixexp(u, params[0], params[1], params[2])


# =============================================================================
# Helper Functions for Mixture of Exponentials EM Fitting
# =============================================================================


def _fit_mixexp_em(
    x: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6
) -> np.ndarray:
    """
    Fit mixture of exponentials using EM algorithm.

    Estimates parameters (prob, rate1, rate2) for a mixture of two exponentials.
    This is a translation of Renext::EM.mixexp.

    Parameters
    ----------
    x : ndarray
        Sample data
    max_iter : int, optional
        Maximum number of EM iterations (default: 100)
    tol : float, optional
        Convergence tolerance (default: 1e-6)

    Returns
    -------
    ndarray
        Estimated parameters [prob, rate1, rate2]

    Notes
    -----
    Uses a standard EM algorithm for mixture models. Initial parameters are
    estimated from the data.
    """
    x = np.asarray(x)
    n = len(x)

    # Initial parameter estimates
    mean_x = np.mean(x)
    prob = 0.5
    rate1 = 1.0 / mean_x
    rate2 = 1.0 / mean_x * 0.5

    for iteration in range(max_iter):
        # E-step: compute posterior probabilities
        lik1 = prob * rate1 * np.exp(-rate1 * x)
        lik2 = (1 - prob) * rate2 * np.exp(-rate2 * x)
        total_lik = lik1 + lik2

        # Avoid division by zero
        total_lik = np.maximum(total_lik, 1e-300)

        z1 = lik1 / total_lik  # posterior prob of component 1

        # M-step: update parameters
        n1 = np.sum(z1)
        n2 = n - n1

        prob_new = n1 / n

        if n1 > 0:
            rate1_new = n1 / np.sum(z1 * x)
        else:
            rate1_new = rate1

        if n2 > 0:
            rate2_new = n2 / np.sum((1 - z1) * x)
        else:
            rate2_new = rate2

        # Check convergence
        params_old = np.array([prob, rate1, rate2])
        params_new = np.array([prob_new, rate1_new, rate2_new])

        if np.max(np.abs(params_new - params_old)) < tol:
            return params_new

        prob, rate1, rate2 = prob_new, rate1_new, rate2_new

    return np.array([prob, rate1, rate2])
