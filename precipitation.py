"""
GWEX precipitation model fitting and simulation module.

This module provides the main precipitation model estimation and simulation functions
for the GWEX weather generator. It includes:

- Fitting functions: infer_mat_omega, get_mat_omega, find_omega, cor_emp_occ,
  get_M0, find_zeta, cor_emp_int, get_vec_autocor, find_autocor, autocor_emp_int,
  infer_dep_amount, infer_autocor_amount, fit_copula_amount, fit_MAR1_amount,
  fit_GWex_prec

- Simulation functions: sim_GWex_occ, sim_GWex_Yt_Pr, sim_GWex_Yt, mask_GWex_Yt,
  sim_GWex_prec_1it

- Helper functions for disaggregation, helper routines

Translation from R GWexPrec_lib.r by Guillaume Evin.

References:
    Evin, G., A.-C. Favre, and B. Hingray. 2018. "Stochastic Generation of
    Multi-Site Daily Precipitation Focusing on Extreme Events." Hydrol. Earth
    Syst. Sci. 22 (1): 655–72. https://doi.org/10.5194/hess-22-655-2018.

    Wilks, D.S. (1998) "Multisite generalization of a daily stochastic
    precipitation generation model", J Hydrol, 210: 178-191
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, fsolve
from scipy import stats
from itertools import combinations, product
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Optional

if TYPE_CHECKING:
    from .core import GwexObs

from .utils import (
    modify_cor_matrix, dry_day_frequency, wet_day_frequency,
    joint_proba_occ, cor_obs_occ, lag_trans_proba_matrix,
    get_emp_cdf_matrix, get_list_option, get_list_month,
    get_period_fitting_month, month2season, agg_matrix,
    get_df_student
)
from .distributions import (
    unif_to_prec, fit_margin_cdf, ppf_egpd_gi, cdf_mixexp
)
from .simulation import (
    sim_precip_occurrences, sim_precip_occurrences_4_fitting,
    cor_markov_chain, disag_3day_gwex_prec
)


# =============================================================================
# Fitting Functions for Occurrence Process
# =============================================================================


def infer_mat_omega(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    th: float,
    nLag: int,
    pr_state: List[pd.DataFrame],
    nChainFit: int,
    is_parallel: bool = False
) -> Dict:
    """
    Infer spatial correlation matrix omega for occurrence process.

    For all stations, estimates the correlation matrix of the Gaussian random
    variates that drive the occurrence Markov chain, given observed correlations
    between dry/wet occurrences.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, n_stations)
        Precipitation matrix
    is_period : ndarray, shape (n_days,)
        Boolean mask for the 3-month period of interest
    th : float
        Wet/dry threshold
    nLag : int
        Number of lag days for Markov chain
    pr_state : list of pd.DataFrame
        Transition probability DataFrames for each station
    nChainFit : int
        Number of simulated chain steps for fitting
    is_parallel : bool, optional
        Whether to use parallel computation (default: False, currently sequential)

    Returns
    -------
    dict
        Dictionary with keys:
        - 'Qtrans_mat': ndarray of shape (n_stations, 2^nLag), normal quantiles
        - 'mat_comb': ndarray of shape (2^nLag, nLag), possible combinations
        - 'mat_omega': ndarray of shape (n_stations, n_stations), omega matrix
    """
    # Filter to the period of interest
    P_mat_class = P_mat[is_period, :]

    # Compute observed correlation of dry/wet occurrences
    pi0 = dry_day_frequency(P_mat_class, th)
    pi1 = wet_day_frequency(P_mat_class, th)
    pi_occ = joint_proba_occ(P_mat_class, th)
    cor_occ_obs = cor_obs_occ(pi_occ['p00'], pi0, pi1)

    # Number of possible transitions
    n_comb = 2 ** nLag

    # Matrix of possible wet/dry combinations
    lag_cols = [f't{i}' for i in range(-nLag, 0)]
    mat_comb = pr_state[0][lag_cols].values.astype(bool)

    # Transition probabilities to normal quantiles
    Ptrans_list = [pr_state[i_st]['P'].values for i_st in range(len(pr_state))]
    Qtrans_list = [stats.norm.ppf(Ptrans) for Ptrans in Ptrans_list]
    Qtrans_mat = np.array(Qtrans_list)

    # Handle infinite values from probabilities 0 or 1
    Qtrans_mat[np.isinf(Qtrans_mat) & (Qtrans_mat < 0)] = -1e5
    Qtrans_mat[np.isinf(Qtrans_mat) & (Qtrans_mat > 0)] = 1e5

    # Estimate omega matrices
    mat_omega = get_mat_omega(
        cor_occ_obs, Qtrans_mat, mat_comb, nLag, nChainFit, is_parallel
    )
    mat_omega = modify_cor_matrix(mat_omega)

    return {
        'Qtrans_mat': Qtrans_mat,
        'mat_comb': mat_comb,
        'mat_omega': mat_omega
    }


def get_mat_omega(
    cor_obs: np.ndarray,
    Qtrans_mat: np.ndarray,
    mat_comb: np.ndarray,
    nLag: int,
    nChainFit: int,
    is_parallel: bool = False
) -> np.ndarray:
    """
    Find omega correlation matrix for all station pairs.

    For each pair of stations, solves for the Gaussian correlation that
    produces the observed dry/wet occurrence correlation.

    Parameters
    ----------
    cor_obs : ndarray, shape (n_stations, n_stations)
        Observed occurrence correlations between stations
    Qtrans_mat : ndarray, shape (n_stations, 2^nLag)
        Normal quantiles for transition probabilities
    mat_comb : ndarray, shape (2^nLag, nLag)
        Matrix of all wet/dry combinations
    nLag : int
        Number of lag days
    nChainFit : int
        Number of chain steps for fitting
    is_parallel : bool, optional
        Whether to use parallel computation (currently sequential)

    Returns
    -------
    ndarray, shape (n_stations, n_stations)
        Spatial correlation matrix omega for occurrences
    """
    p = cor_obs.shape[0]

    # All station pairs (i < j)
    pairs = list(combinations(range(p), 2))
    n_pairs = len(pairs)

    # Find omega for each pair
    omega_pairs = []
    for i, j in pairs:
        omega_ij = find_omega(
            cor_obs[i, j],
            Qtrans_mat[[i, j], :],
            mat_comb,
            nLag,
            nChainFit
        )
        omega_pairs.append(omega_ij)

    # Fill symmetric matrix
    mat_omega = np.zeros((p, p))
    np.fill_diagonal(mat_omega, 1.0)

    for pair_idx, (i, j) in enumerate(pairs):
        mat_omega[i, j] = omega_pairs[pair_idx]
        mat_omega[j, i] = omega_pairs[pair_idx]

    return mat_omega


def find_omega(
    rho_emp: float,
    Qtrans_mat: np.ndarray,
    mat_comb: np.ndarray,
    nLag: int,
    nChainFit: int
) -> float:
    """
    Root-finding for spatial correlation parameter omega.

    Finds the Gaussian correlation omega such that cor_emp_occ(omega, ...)
    equals the target empirical correlation rho_emp.

    Parameters
    ----------
    rho_emp : float
        Target empirical correlation between occurrences
    Qtrans_mat : ndarray, shape (2, 2^nLag)
        Transition quantiles for two stations
    mat_comb : ndarray, shape (2^nLag, nLag)
        Matrix of wet/dry combinations
    nLag : int
        Number of lag days
    nChainFit : int
        Number of chain steps

    Returns
    -------
    float
        Estimated omega (Gaussian correlation)
    """
    # Test boundary values
    f_inf = cor_emp_occ(-1.0, Qtrans_mat, mat_comb, nLag, nChainFit) - rho_emp
    f_sup = cor_emp_occ(1.0, Qtrans_mat, mat_comb, nLag, nChainFit) - rho_emp

    # If maximum omega cannot reach target correlation, return 1
    if f_sup <= 0:
        return 1.0
    # If minimum omega cannot reach target correlation, return 0
    elif f_inf >= 0:
        return 0.0
    else:
        # Negative correlations are not physically plausible
        if rho_emp < 0:
            return 0.0

        # Use Brent's method to find root
        def objective(w):
            return cor_emp_occ(w, Qtrans_mat, mat_comb, nLag, nChainFit) - rho_emp

        try:
            omega: float = brentq(objective, rho_emp, 1.0, xtol=1e-3, full_output=False)  # type: ignore[assignment]
        except ValueError:
            # If brentq fails, return boundary value
            omega = 1.0 if f_sup > 0 else 0.0

        return omega


def cor_emp_occ(
    w: float,
    Qtrans_mat: np.ndarray,
    mat_comb: np.ndarray,
    nLag: int,
    nChainFit: int,
    myseed: int = 1
) -> float:
    """
    Simulate occurrence correlation for given omega.

    Generates bivariate Markov chains with correlation parameter omega and
    computes the resulting empirical correlation between occurrences.

    Parameters
    ----------
    w : float
        Gaussian correlation parameter (omega), in [-1, 1]
    Qtrans_mat : ndarray, shape (2, 2^nLag)
        Transition quantiles for two stations
    mat_comb : ndarray, shape (2^nLag, nLag)
        Matrix of wet/dry combinations
    nLag : int
        Number of lag days
    nChainFit : int
        Number of chain steps to simulate
    myseed : int, optional
        Random seed (default: 1)

    Returns
    -------
    float
        Empirical Pearson correlation between simulated occurrences
    """
    np.random.seed(myseed)

    # Generate bivariate normal with correlation w
    cov_matrix = np.array([[1.0, w], [w, 1.0]])
    rndNorm = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=cov_matrix,
        size=nChainFit + 100
    )

    # Compute correlation
    cor_emp = cor_markov_chain(
        rndNorm=rndNorm,
        QtransMat=Qtrans_mat,
        matcomb=mat_comb,
        nChainFit=nChainFit,
        nLag=nLag
    )

    return cor_emp


# =============================================================================
# Fitting Functions for Intensity Process
# =============================================================================


def get_M0(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    th: float,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int,
    nLag: int,
    infer_mat_omega_out: Optional[Dict] = None,
    is_parallel: bool = False
) -> np.ndarray:
    """
    Estimate the Gaussian covariance matrix M0 for precipitation intensities.

    For each station pair (i, j), computes the observed Pearson correlation
    of wet-day intensities on jointly-wet days, then calls find_zeta to find
    the latent Gaussian correlation that reproduces it via simulation matching.
    Assembles pairwise values into M0 and ensures positive-definiteness.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, n_stations)
        Precipitation matrix
    is_period : ndarray, shape (n_days,)
        Boolean mask for the 3-month fitting window
    th : float
        Wet/dry threshold
    parMargin : ndarray, shape (n_stations, 3)
        Marginal parameters per station
    typeMargin : str
        Type of margin ('EGPD' or 'mixExp')
    nChainFit : int
        Number of chain steps for simulation matching
    nLag : int
        Number of lag days for Markov chain
    infer_mat_omega_out : dict, optional
        Output from infer_mat_omega (used for generating occurrence patterns)
    is_parallel : bool, optional
        Whether to use parallel computation

    Returns
    -------
    ndarray, shape (n_stations, n_stations)
        Positive-definite Gaussian covariance matrix M0
    """
    P_period = P_mat[is_period, :]
    p = P_period.shape[1]
    M0 = np.eye(p)

    # Observed occurrence pattern for simulation matching
    Xt_obs = (P_period > th).astype(float)

    for i, j in combinations(range(p), 2):
        # Observed correlation on jointly-wet days
        wet_both = (Xt_obs[:, i] == 1) & (Xt_obs[:, j] == 1)
        n_joint = np.sum(wet_both)
        if n_joint > 10:
            cor_ij = np.corrcoef(P_period[wet_both, i], P_period[wet_both, j])[0, 1]
            if np.isnan(cor_ij):
                cor_ij = 0.0
        else:
            cor_ij = 0.0

        if abs(cor_ij) < 1e-6:
            M0[i, j] = M0[j, i] = 0.0
            continue

        # Use simulated occurrence pattern from omega (if available)
        # or fall back to observed occurrence pattern
        Xt_pair = Xt_obs[:min(nChainFit, len(Xt_obs)), [i, j]]

        zeta = find_zeta(
            eta_emp=cor_ij,
            nChainFit=len(Xt_pair),
            Xt=Xt_pair,
            parMargin=parMargin[[i, j], :],
            typeMargin=typeMargin
        )
        M0[i, j] = M0[j, i] = zeta

    # Ensure positive-definiteness
    M0 = modify_cor_matrix(M0)

    return M0


def find_zeta(
    eta_emp: float,
    nChainFit: int,
    Xt: np.ndarray,
    parMargin: np.ndarray,
    typeMargin: str
) -> float:
    """
    Root-finding for intensity spatial correlation parameter zeta.

    Parameters
    ----------
    eta_emp : float
        Target empirical intensity correlation
    nChainFit : int
        Number of chain steps
    Xt : ndarray, shape (nChainFit, 2)
        Occurrence pattern for two stations
    parMargin : ndarray, shape (2, 3)
        Marginal parameters for two stations
    typeMargin : str
        Type of margin

    Returns
    -------
    float
        Estimated zeta (intensity correlation)
    """
    # Test boundary values
    f_inf = cor_emp_int(-1.0, nChainFit, Xt, parMargin, typeMargin) - eta_emp
    f_sup = cor_emp_int(1.0, nChainFit, Xt, parMargin, typeMargin) - eta_emp

    if f_sup <= 0:
        return 1.0
    elif f_inf >= 0:
        return 0.0
    else:
        if eta_emp < 0:
            return 0.0

        def objective(z):
            return cor_emp_int(z, nChainFit, Xt, parMargin, typeMargin) - eta_emp

        try:
            zeta: float = brentq(objective, eta_emp, 1.0, xtol=1e-3, full_output=False)  # type: ignore[assignment]
        except ValueError:
            zeta = 1.0 if f_sup > 0 else 0.0

        return zeta


def cor_emp_int(
    zeta: float,
    nChainFit: int,
    Xt: np.ndarray,
    parMargin: np.ndarray,
    typeMargin: str
) -> float:
    """
    Simulate intensity correlation for given zeta.

    Parameters
    ----------
    zeta : float
        Intensity correlation parameter
    nChainFit : int
        Number of chain steps
    Xt : ndarray, shape (nChainFit, 2)
        Occurrence indicator for two stations
    parMargin : ndarray, shape (2, 3)
        Marginal parameters
    typeMargin : str
        Type of margin

    Returns
    -------
    float
        Empirical correlation between simulated intensities
    """
    # Generate correlated Gaussian variates for the two stations
    cov_matrix = np.array([[1.0, zeta], [zeta, 1.0]])
    Yt_Gau = np.random.multivariate_normal(
        mean=[0.0, 0.0],
        cov=cov_matrix,
        size=nChainFit
    )

    # Transform to uniform [0,1]
    Yt_Pr = stats.norm.cdf(Yt_Gau)

    # Transform to precipitation amounts
    Yt = np.zeros((nChainFit, 2))
    for st in range(2):
        Yt[:, st] = unif_to_prec(Yt_Pr[:, st], parMargin[st, :], typeMargin)

    # Mask with occurrences (multiply by Xt)
    Yt_masked = Yt * Xt

    # Compute correlation on wet days only
    wet_idx = (Xt[:, 0] == 1) & (Xt[:, 1] == 1)
    if np.sum(wet_idx) > 1:
        cor_int = np.corrcoef(Yt_masked[wet_idx, 0], Yt_masked[wet_idx, 1])[0, 1]
        if np.isnan(cor_int):
            cor_int = 0.0
    else:
        cor_int = 0.0

    return cor_int


def get_vec_autocor(
    vec_ar1_obs: np.ndarray,
    Xt: np.ndarray,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int,
    is_parallel: bool = False
) -> np.ndarray:
    """
    Find AR(1) autocorrelation parameters.

    For each station, solves for the autocorrelation coefficient that produces
    the observed lag-1 autocorrelation of precipitation amounts.

    Parameters
    ----------
    vec_ar1_obs : ndarray, shape (n_stations,)
        Observed lag-1 autocorrelations
    Xt : ndarray, shape (nChainFit, n_stations)
        Occurrence patterns
    parMargin : ndarray, shape (n_stations, 3)
        Marginal parameters
    typeMargin : str
        Type of margin
    nChainFit : int
        Number of chain steps
    is_parallel : bool, optional
        Whether to use parallel computation

    Returns
    -------
    ndarray, shape (n_stations,)
        AR(1) parameters (rho values)
    """
    p = len(vec_ar1_obs)
    vec_rho = np.zeros(p)

    for st in range(p):
        vec_rho[st] = find_autocor(
            vec_ar1_obs[st], nChainFit, Xt[:, st], parMargin[st, :], typeMargin
        )

    return vec_rho


def find_autocor(
    autocor_emp: float,
    nChainFit: int,
    Xt: np.ndarray,
    parMargin: np.ndarray,
    typeMargin: str
) -> float:
    """
    Root-finding for lag-1 autocorrelation parameter.

    Parameters
    ----------
    autocor_emp : float
        Target empirical autocorrelation
    nChainFit : int
        Number of chain steps
    Xt : ndarray, shape (nChainFit,)
        Occurrence pattern for one station
    parMargin : ndarray, shape (3,)
        Marginal parameters
    typeMargin : str
        Type of margin

    Returns
    -------
    float
        Estimated AR(1) parameter (rho)
    """
    # Test boundary values
    f_inf = autocor_emp_int(-0.99, nChainFit, Xt, parMargin, typeMargin) - autocor_emp
    f_sup = autocor_emp_int(0.99, nChainFit, Xt, parMargin, typeMargin) - autocor_emp

    if f_sup <= 0:
        return 0.99
    elif f_inf >= 0:
        return -0.99
    else:
        def objective(rho):
            return autocor_emp_int(rho, nChainFit, Xt, parMargin, typeMargin) - autocor_emp

        try:
            rho: float = brentq(objective, -0.99, 0.99, xtol=1e-3, full_output=False)  # type: ignore[assignment]
        except ValueError:
            rho = 0.0

        return rho


def autocor_emp_int(
    rho: float,
    nChainFit: int,
    Xt: np.ndarray,
    parMargin: np.ndarray,
    typeMargin: str
) -> float:
    """
    Simulate lag-1 autocorrelation of intensities for given AR(1) parameter.

    Parameters
    ----------
    rho : float
        AR(1) parameter (autocorrelation)
    nChainFit : int
        Number of chain steps
    Xt : ndarray, shape (nChainFit,)
        Occurrence pattern
    parMargin : ndarray, shape (3,)
        Marginal parameters
    typeMargin : str
        Type of margin

    Returns
    -------
    float
        Simulated lag-1 autocorrelation
    """
    # Simulate AR(1) process: Z[t] = rho*Z[t-1] + e[t]
    # where e ~ N(0, 1-rho^2)
    sd_e = np.sqrt(1.0 - rho ** 2)
    Yt_Gau = np.zeros(nChainFit)
    Yt_Gau[0] = np.random.randn()

    for t in range(1, nChainFit):
        Yt_Gau[t] = rho * Yt_Gau[t - 1] + sd_e * np.random.randn()

    # Transform to uniform and then to precipitation
    Yt_Pr = stats.norm.cdf(Yt_Gau)
    Yt = unif_to_prec(Yt_Pr, parMargin, typeMargin)

    # Mask with occurrences
    Yt_masked = Yt * Xt

    # Compute lag-1 autocorrelation on wet days
    wet_idx = Xt == 1
    if np.sum(wet_idx) > 2:
        wet_vals = Yt_masked[wet_idx]
        if len(wet_vals) > 2:
            # Compute autocorrelation at lag 1
            autocor = np.corrcoef(wet_vals[:-1], wet_vals[1:])[0, 1]
            if np.isnan(autocor):
                autocor = 0.0
        else:
            autocor = 0.0
    else:
        autocor = 0.0

    return autocor


def infer_autocor_amount(
    P_mat: np.ndarray,
    pr_state: List[pd.DataFrame],
    is_period: np.ndarray,
    nLag: int,
    th: float,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int,
    is_MAR: bool,
    is_parallel: bool = False
) -> Optional[Dict]:
    """
    Estimate spatial dependence parameters for single-station case.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, 1)
        Precipitation matrix (single station)
    pr_state : list
        Transition probabilities (not used for single station)
    is_period : ndarray
        Boolean mask for period
    nLag : int
        Number of lag days
    th : float
        Wet/dry threshold
    parMargin : ndarray, shape (1, 3)
        Marginal parameters
    typeMargin : str
        Type of margin
    nChainFit : int
        Number of chain steps
    is_MAR : bool
        Whether to fit MAR(1) model
    is_parallel : bool, optional
        Whether to use parallel computation

    Returns
    -------
    dict
        Dictionary with fitted parameters for intensity process
    """
    if is_MAR:
        return fit_MAR1_amount(P_mat, is_period, th, parMargin, typeMargin,
                               nChainFit=nChainFit, nLag=nLag)
    else:
        # No spatial dependence for single station
        return None


def infer_dep_amount(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    infer_mat_omega_out: Optional[Dict],
    nLag: int,
    th: float,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int,
    is_MAR: bool,
    copulaInt: str,
    is_parallel: bool = False
) -> Dict:
    """
    Estimate spatial dependence parameters for precipitation intensities.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, n_stations)
        Precipitation matrix (n_stations > 1)
    is_period : ndarray
        Boolean mask for period
    infer_mat_omega_out : dict
        Output from infer_mat_omega
    nLag : int
        Number of lag days
    th : float
        Wet/dry threshold
    parMargin : ndarray, shape (n_stations, 3)
        Marginal parameters
    typeMargin : str
        Type of margin
    nChainFit : int
        Number of chain steps
    is_MAR : bool
        Whether to fit MAR(1) model
    copulaInt : str
        Type of copula ('Gaussian' or 'Student')
    is_parallel : bool, optional
        Whether to use parallel computation

    Returns
    -------
    dict
        Dictionary with fitted intensity dependence parameters
    """
    if is_MAR:
        return fit_MAR1_amount(P_mat, is_period, th, parMargin, typeMargin,
                               nChainFit=nChainFit, nLag=nLag)
    else:
        return fit_copula_amount(
            P_mat, is_period, nLag, th, parMargin, typeMargin, nChainFit, copulaInt
        )


def fit_copula_amount(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    nLag: int,
    th: float,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int,
    copulaInt: str
) -> Dict:
    """
    Fit copula parameters for precipitation intensities (no temporal dependence).

    Estimates the spatial dependence structure via simulation matching:
    for each station pair, finds the latent Gaussian correlation that
    reproduces the observed wet-day intensity correlation. Optionally
    estimates Student-t degrees of freedom for upper tail dependence.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, n_stations)
        Precipitation matrix
    is_period : ndarray
        Boolean mask for 3-month fitting window
    nLag : int
        Number of lag days
    th : float
        Wet/dry threshold
    parMargin : ndarray, shape (n_stations, 3)
        Marginal parameters
    typeMargin : str
        Type of margin ('EGPD' or 'mixExp')
    nChainFit : int
        Number of chain steps for simulation matching
    copulaInt : str
        Type of copula ('Gaussian' or 'Student')

    Returns
    -------
    dict
        Keys: 'M0', 'A', 'covZ', 'sdZ', 'corZ', and optionally 'df'
    """
    p = P_mat.shape[1]

    # Estimate M0 via simulation matching
    M0 = get_M0(P_mat, is_period, th, parMargin, typeMargin, nChainFit, nLag)

    sdZ = np.sqrt(np.diag(M0))
    corZ = M0.copy()

    result = {
        'M0': M0,
        'A': np.zeros((p, p)),
        'covZ': M0,
        'sdZ': sdZ,
        'corZ': corZ
    }

    # Estimate Student-t degrees of freedom if requested
    if copulaInt == 'Student':
        P_period = P_mat[is_period, :]
        # Extract wet-day data for df estimation
        wet_all = np.all(P_period > th, axis=1)
        P_wet = P_period[wet_all, :]
        if P_wet.shape[0] > 20:
            df = get_df_student(P_wet, corZ)
            result['df'] = df

    return result


def fit_MAR1_amount(
    P_mat: np.ndarray,
    is_period: np.ndarray,
    th: float,
    parMargin: np.ndarray,
    typeMargin: str,
    nChainFit: int = 10000,
    nLag: int = 2
) -> Dict:
    """
    Fit MAR(1) model for precipitation intensities.

    Estimates both spatial dependence (M0) and temporal dependence (A):
    Z_t = A @ Z_{t-1} + epsilon_t,  Cov(epsilon) = M0 - A @ M0 @ A^T

    A is diagonal with per-station lag-1 autocorrelation coefficients
    found by simulation matching via get_vec_autocor.

    Parameters
    ----------
    P_mat : ndarray, shape (n_days, n_stations)
        Precipitation matrix
    is_period : ndarray
        Boolean mask for 3-month fitting window
    th : float
        Wet/dry threshold
    parMargin : ndarray, shape (n_stations, 3)
        Marginal parameters
    typeMargin : str
        Type of margin ('EGPD' or 'mixExp')
    nChainFit : int, optional
        Number of chain steps for simulation matching (default: 10000)
    nLag : int, optional
        Number of lag days for Markov chain (default: 2)

    Returns
    -------
    dict
        Keys: 'M0', 'A', 'covZ', 'sdZ', 'corZ'
    """
    P_period = P_mat[is_period, :]
    p = P_period.shape[1]

    # Step 1: Estimate spatial covariance M0
    M0 = get_M0(P_mat, is_period, th, parMargin, typeMargin, nChainFit, nLag)

    # Step 2: Compute observed lag-1 autocorrelations per station
    Xt_obs = (P_period > th).astype(float)
    vec_ar1_obs = np.zeros(p)
    for st in range(p):
        wet_vals = P_period[Xt_obs[:, st] == 1, st]
        if len(wet_vals) > 2:
            vec_ar1_obs[st] = np.corrcoef(wet_vals[:-1], wet_vals[1:])[0, 1]
            if np.isnan(vec_ar1_obs[st]):
                vec_ar1_obs[st] = 0.0

    # Step 3: Find latent AR(1) coefficients via simulation matching
    Xt_fit = Xt_obs[:min(nChainFit, len(Xt_obs)), :]
    vec_rho = get_vec_autocor(
        vec_ar1_obs=vec_ar1_obs,
        Xt=Xt_fit,
        parMargin=parMargin,
        typeMargin=typeMargin,
        nChainFit=len(Xt_fit)
    )

    # Step 4: Assemble diagonal A matrix
    A = np.diag(vec_rho)

    # Step 5: Innovation covariance: Cov(epsilon) = M0 - A @ M0 @ A^T
    covZ = M0 - A @ M0 @ A.T

    # Ensure positive-definiteness
    covZ = modify_cor_matrix(covZ)

    sdZ = np.sqrt(np.diag(M0))
    corZ = M0.copy()

    return {
        'M0': M0,
        'A': A,
        'covZ': covZ,
        'sdZ': sdZ,
        'corZ': corZ
    }


# =============================================================================
# Main Fitting Function
# =============================================================================


def _fit_one_month(args: Tuple) -> Tuple:
    """Fit all GWEX parameters for a single month.

    Designed as a module-level function so it is picklable for
    multiprocessing.Pool.  Receives and returns plain tuples to
    avoid serialisation issues with complex objects.
    """
    import warnings
    (iMonth, P_mat, vec_month, vec_month_char, p,
     parMargin_month, listOption) = args

    m_char = vec_month_char[iMonth]

    # Get 3-month fitting window
    period_m = get_period_fitting_month(m_char)
    is_3month_period = np.isin(vec_month, period_m)
    is_month = vec_month == (iMonth + 1)

    # --- Wet/dry transition probabilities ---
    pr_state = lag_trans_proba_matrix(
        P_mat, is_month, listOption['th'], listOption['nLag']
    )
    pr_state_list = [pr_state[i] for i in range(p)]

    # --- Marginal distributions ---
    if parMargin_month is None:
        parMargin_month = fit_margin_cdf(
            P_mat, is_month, listOption['th'], listOption['typeMargin']
        )

    # --- Spatial process for occurrences ---
    if p == 1:
        infer_mat_omega_out = None
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer_mat_omega_out = infer_mat_omega(
                P_mat,
                is_3month_period,
                listOption['th'],
                listOption['nLag'],
                pr_state_list,
                listOption['nChainFit'],
                listOption['isParallel']
            )

    # --- Spatial process for intensities ---
    if p == 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer_dep_amount_out = infer_autocor_amount(
                P_mat,
                pr_state_list,
                is_3month_period,
                listOption['nLag'],
                listOption['th'],
                parMargin_month,
                listOption['typeMargin'],
                listOption['nChainFit'],
                listOption['isMAR'],
                listOption['isParallel']
            )
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            infer_dep_amount_out = infer_dep_amount(
                P_mat,
                is_3month_period,
                infer_mat_omega_out,
                listOption['nLag'],
                listOption['th'],
                parMargin_month,
                listOption['typeMargin'],
                listOption['nChainFit'],
                listOption['isMAR'],
                listOption['copulaInt'],
                listOption['isParallel']
            )

    return (iMonth, pr_state_list, parMargin_month,
            infer_mat_omega_out, infer_dep_amount_out)


def fit_GWex_prec(
    objGwexObs: "GwexObs",
    parMargin: Optional[List[np.ndarray]] = None,
    listOption: Optional[Dict] = None,
    n_fit_workers: int = 1,
) -> Dict:
    """
    Main function to fit the GWEX precipitation model.

    Estimates all parameters for the GWEX model including:
    - Wet/dry transition probabilities
    - Marginal distributions of precipitation amounts
    - Spatial correlation structure for occurrences (omega)
    - Spatial/temporal dependence of intensities

    Parameters
    ----------
    objGwexObs : dict
        Observation object with attributes:
        - 'obs': ndarray of shape (n_days, n_stations), precipitation data
        - 'date': array of dates
    parMargin : list of ndarray, optional
        Pre-fitted marginal parameters for each month. If None, will be estimated.
        Each element should be ndarray of shape (n_stations, 3).
    listOption : dict, optional
        Options for fitting (see get_list_option)
    n_fit_workers : int, optional
        Number of parallel workers for month-level fitting.  1 = sequential
        (default).  Up to 12 workers can be used (one per month).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'listOption': fitted options
        - 'listPar': list of estimated parameters
    """
    # Get/check options
    listOption = get_list_option(listOption)

    # ========== Retrieve observations and dates ==========
    obs_data = objGwexObs.obs
    obs_date = objGwexObs.date

    if listOption.get('is3Damount', False):
        # Aggregate to 3-day precipitation
        P_1D = obs_data
        P_mat = agg_matrix(P_1D, 3)

        # Adjust dates to 3-day periods
        n = P_1D.shape[0]
        vec_dates = obs_date[::3][:n // 3]
        day_scale = 3
    else:
        P_mat = obs_data
        vec_dates = obs_date
        day_scale = 1

    # ========== Process dates and periods ==========
    # Extract months from dates (assume datetime-like)
    if hasattr(vec_dates[0], 'month'):
        vec_month = np.array([d.month for d in vec_dates])
    else:
        # If dates are strings, try to parse
        vec_month = np.array([int(str(d).split('-')[1]) for d in vec_dates])

    vec_month_char = get_list_month()
    p = P_mat.shape[1]

    # ========== Check parMargin ==========
    if parMargin is not None:
        if not isinstance(parMargin, list):
            raise ValueError("parMargin must be a list")
        if len(parMargin) != 12:
            raise ValueError("parMargin must have 12 elements (one per month)")
        for iM in range(12):
            if parMargin[iM].shape != (p, 3):
                raise ValueError(f"parMargin[{iM}] has wrong shape")

    # ========== Build per-month argument tuples ==========
    month_args = []
    for iMonth in range(12):
        pm = parMargin[iMonth] if parMargin is not None else None
        month_args.append((
            iMonth, P_mat, vec_month, vec_month_char, p,
            pm, listOption,
        ))

    # ========== Fit months (parallel or sequential) ==========
    n_fit_workers = max(1, min(n_fit_workers, 12))

    if n_fit_workers > 1:
        import multiprocessing as _mp
        print(f"  [gwex fit] fitting 12 months with {n_fit_workers} parallel workers ...",
              flush=True)
        with _mp.Pool(n_fit_workers) as pool:
            results = []
            for result in pool.imap_unordered(_fit_one_month, month_args):
                iM = result[0]
                m_char = vec_month_char[iM]
                print(f"  [gwex fit] month {iM+1:02d}/12 ({m_char}) done", flush=True)
                results.append(result)
        # Sort by month index
        results.sort(key=lambda x: x[0])
    else:
        results = []
        for args in month_args:
            iMonth = args[0]
            m_char = vec_month_char[iMonth]
            print(f"  [gwex fit] month {iMonth+1:02d}/12 ({m_char}) ...", flush=True)
            results.append(_fit_one_month(args))

    # ========== Unpack results ==========
    list_pr_state = [None] * 12
    list_parMargin = [None] * 12
    list_mat_omega = [None] * 12
    list_par_dep_amount = [None] * 12

    for (iMonth, pr_state_list, parMargin_month,
         infer_mat_omega_out, infer_dep_amount_out) in results:
        list_pr_state[iMonth] = pr_state_list
        list_parMargin[iMonth] = parMargin_month
        list_mat_omega[iMonth] = infer_mat_omega_out
        list_par_dep_amount[iMonth] = infer_dep_amount_out

    # ========== Prepare output ==========
    listPar = {
        'parOcc': {
            'list_pr_state': list_pr_state,
            'list_mat_omega': list_mat_omega
        },
        'parInt': {
            'cor_int': list_par_dep_amount,
            'parMargin': list_parMargin
        },
        'p': p
    }

    return {
        'listOption': listOption,
        'listPar': listPar
    }


# =============================================================================
# Simulation Functions
# =============================================================================


def sim_GWex_occ(
    objGwexFit: Dict,
    vecMonth: np.ndarray
) -> np.ndarray:
    """
    Simulate precipitation occurrences (wet/dry states).

    Generates a time series of binary wet/dry states for all stations using
    multivariate Markov chains with spatial dependence.

    Parameters
    ----------
    objGwexFit : dict
        Fitted model object with parameters
    vecMonth : ndarray, shape (n_days,)
        Vector of month indices (1-12) for each day

    Returns
    -------
    ndarray, shape (n_days, n_stations)
        Binary occurrence matrix (1=wet, 0=dry)
    """
    p = objGwexFit['listPar']['p']
    n = len(vecMonth)
    nLag = objGwexFit['listOption']['nLag']

    # Initialize output and random normals
    Xt = np.zeros((n, p), dtype=float)
    rndNorm = np.zeros((n, p), dtype=float)

    # Number of possible combinations
    n_comb = 2 ** nLag

    # Transition quantile array
    Qtrans = np.zeros((n, p, n_comb))

    # Parameters for occurrence process
    parOcc = objGwexFit['listPar']['parOcc']

    # Matrix of combinations from first month
    lag_cols = [f't{i}' for i in range(-nLag, 0)]
    mat_comb = parOcc['list_pr_state'][0][0][lag_cols].values.astype(bool)

    # Fill transition quantiles and generate random normals
    for t in range(n):
        iMonth = vecMonth[t] - 1  # Convert to 0-indexed

        # Get transition probabilities for this month
        Ptrans_list = [
            parOcc['list_pr_state'][iMonth][i_st]['P'].values
            for i_st in range(p)
        ]
        Qtrans_list = [stats.norm.ppf(Ptrans) for Ptrans in Ptrans_list]
        Qtrans_mat = np.array(Qtrans_list)

        for i_st in range(p):
            for i_comb in range(n_comb):
                Qtrans[t, i_st, i_comb] = Qtrans_mat[i_st, i_comb]

        # Generate multivariate Gaussian with spatial correlation
        if p == 1:
            rndNorm[t, :] = np.random.randn(1)
        else:
            mat_omega = parOcc['list_mat_omega'][iMonth]['mat_omega']
            rndNorm[t, :] = np.random.multivariate_normal(
                mean=np.zeros(p),
                cov=mat_omega
            )

    # Simulate occurrences for each station
    for st in range(p):
        Xt[:, st] = sim_precip_occurrences(
            nLag=nLag,
            matcomb=mat_comb,
            Qtrans=Qtrans[:, st, :],
            rndNorm=rndNorm[:, st]
        )

    return Xt


def sim_GWex_Yt_Pr(
    objGwexFit: Dict,
    vecMonth: np.ndarray
) -> np.ndarray:
    """
    Simulate dependent uniform variates for precipitation intensities.

    Generates uniform random variates [0,1] that represent the dependence
    structure between precipitation intensities at different sites and times.

    Parameters
    ----------
    objGwexFit : dict
        Fitted model object
    vecMonth : ndarray, shape (n_days,)
        Month indices

    Returns
    -------
    ndarray, shape (n_days, n_stations)
        Uniform random variates [0,1]
    """
    is_MAR = objGwexFit['listOption']['isMAR']
    copulaInt = objGwexFit['listOption']['copulaInt']
    p = objGwexFit['listPar']['p']
    n = len(vecMonth)

    # Prepare output matrix
    Yt_Gau = np.zeros((n, p))

    # Initialise par so it is always bound before the loop body uses it
    par: Dict = _sim_GWex_Yt_Pr_get_param(objGwexFit, vecMonth[0] - 1)

    # First time step: simulate from marginal distribution
    if p == 1:
        Yt_Gau[0, :] = np.random.randn(1)
    else:
        Yt_Gau[0, :] = _sim_Zt_Spatial(par, copulaInt, p)

    # Subsequent time steps
    for t in range(1, n):
        iMonth_prev = vecMonth[t - 1] - 1
        iMonth = vecMonth[t] - 1

        # Update parameters if month changes
        if iMonth != iMonth_prev:
            par = _sim_GWex_Yt_Pr_get_param(objGwexFit, iMonth)

        if is_MAR:
            if p == 1:
                # AR(1): Y(t) = A*Y(t-1) + e
                sd_e = par['sdZ'] if 'sdZ' in par else 1.0
                Yt_Gau[t, :] = par['A'] * Yt_Gau[t - 1, :] + sd_e * np.random.randn(1)
            else:
                # MAR(1)
                Yt_Gau[t, :] = _sim_Zt_MAR(par, copulaInt, Yt_Gau[t - 1, :], p)
        else:
            if p == 1:
                Yt_Gau[t, :] = np.random.randn(1)
            else:
                Yt_Gau[t, :] = _sim_Zt_Spatial(par, copulaInt, p)

    # Transform Gaussian to uniform via CDF
    return stats.norm.cdf(Yt_Gau)


def sim_GWex_Yt(
    objGwexFit: Dict,
    vecMonth: np.ndarray,
    Yt_Pr: np.ndarray
) -> np.ndarray:
    """
    Inverse PIT: transform uniform variates to precipitation amounts.

    Parameters
    ----------
    objGwexFit : dict
        Fitted model object
    vecMonth : ndarray, shape (n_days,)
        Month indices
    Yt_Pr : ndarray, shape (n_days, n_stations)
        Uniform variates [0,1] representing dependence

    Returns
    -------
    ndarray, shape (n_days, n_stations)
        Simulated precipitation intensities
    """
    p = objGwexFit['listPar']['p']
    n = Yt_Pr.shape[0]

    Yt = np.zeros((n, p))
    parMargin = objGwexFit['listPar']['parInt']['parMargin']
    typeMargin = objGwexFit['listOption']['typeMargin']

    # Process each month
    for iMonth in range(12):
        is_period = vecMonth == (iMonth + 1)
        n_class = np.sum(is_period)

        if n_class > 0:
            # Get indices for this month
            idx = np.where(is_period)[0]

            # For each station
            for st in range(p):
                params = parMargin[iMonth][st, :]
                for i, t in enumerate(idx):
                    Yt[t, st] = unif_to_prec(Yt_Pr[t, st], params, typeMargin)

    return Yt


def mask_GWex_Yt(
    Xt: np.ndarray,
    Yt: np.ndarray
) -> np.ndarray:
    """
    Mask intensities with occurrences.

    Sets precipitation to zero where occurrences are dry.

    Parameters
    ----------
    Xt : ndarray, shape (n_days, n_stations)
        Binary occurrence matrix
    Yt : ndarray, shape (n_days, n_stations)
        Intensity values

    Returns
    -------
    ndarray, shape (n_days, n_stations)
        Masked precipitation (Yt where Xt==1, else 0)
    """
    return Yt * Xt


def sim_GWex_prec_1it(
    objGwexFit: Dict,
    vecDates: np.ndarray,
    myseed: int,
    objGwexObs: Optional[Dict] = None,
    prob_class: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Simulate one complete iteration of precipitation.

    Generates one full realization of daily precipitation for all stations
    and dates, optionally disaggregating from 3-day to 1-day values.

    Parameters
    ----------
    objGwexFit : dict
        Fitted model
    vecDates : ndarray
        Sequence of dates
    myseed : int
        Random seed
    objGwexObs : dict, optional
        Observed data (needed for 3-day disaggregation)
    prob_class : ndarray, optional
        Classification probabilities for disaggregation

    Returns
    -------
    ndarray, shape (n_days, n_stations)
        Simulated daily precipitation
    """
    np.random.seed(myseed)

    is_3D_amount = objGwexFit['listOption'].get('is3Damount', False)

    # Adjust dates if needed
    if is_3D_amount:
        n_orig = len(vecDates)
        n = int(np.ceil(n_orig / 3) * 3)
        vecDates = np.concatenate([vecDates, vecDates[-3:]])  # Pad to multiple of 3
        vecDates = vecDates[::3][:n // 3]

    # Extract month information
    if hasattr(vecDates[0], 'month'):
        vecMonth = np.array([d.month for d in vecDates])
    else:
        vecMonth = np.array([int(str(d).split('-')[1]) for d in vecDates])

    # Simulate occurrences
    Xt = sim_GWex_occ(objGwexFit, vecMonth)

    # Simulate dependent uniform variates
    Yt_Pr = sim_GWex_Yt_Pr(objGwexFit, vecMonth)

    # Transform to precipitation amounts
    Yt = sim_GWex_Yt(objGwexFit, vecMonth, Yt_Pr)

    # Disaggregate or mask
    if is_3D_amount:
        # Requires observed data
        if objGwexObs is None:
            raise ValueError("objGwexObs required for 3-day disaggregation")
        if prob_class is None:
            prob_class = np.array([0.25, 0.5, 0.75])

        # Disaggregate 3-day to 1-day
        result = disag_3day_gwex_prec(
            Yobs=objGwexObs['obs'],
            Y3obs=agg_matrix(objGwexObs['obs'], 3),
            mObs=np.array([int(str(d).split('-')[1]) for d in objGwexObs['date'][::3]]),
            cObs=np.ones(agg_matrix(objGwexObs['obs'], 3).shape[0]),  # Placeholder
            Y3sim=Yt,
            mSim=vecMonth,
            cSim=np.ones(len(vecMonth)),  # Placeholder
            nLagScore=1
        )
        Pt = result['Ysim']
    else:
        # Simple masking
        Pt = mask_GWex_Yt(Xt, Yt)

    return Pt


# =============================================================================
# Helper Functions
# =============================================================================


def _sim_GWex_Yt_Pr_get_param(
    objGwexFit: Dict,
    iMonth: int
) -> Dict:
    """
    Get parameters for intensity simulation at a given month.

    Parameters
    ----------
    objGwexFit : dict
        Fitted model
    iMonth : int
        Month index (0-indexed)

    Returns
    -------
    dict
        Parameters including M0, A, corZ, sdZ, etc.
    """
    parInt = objGwexFit['listPar']['parInt']
    cor_int = parInt['cor_int'][iMonth]

    if cor_int is None:
        # Single station case
        return {'sdZ': 1.0, 'A': 0.0}

    return cor_int


def _sim_Zt_Spatial(
    par: Dict,
    copulaInt: str,
    p: int
) -> np.ndarray:
    """
    Simulate spatial Gaussian variates.

    Parameters
    ----------
    par : dict
        Parameters including corZ
    copulaInt : str
        Copula type
    p : int
        Number of stations

    Returns
    -------
    ndarray, shape (p,)
        Simulated Gaussian variates
    """
    if copulaInt == 'Gaussian':
        corZ = par.get('corZ', np.eye(p))
        return np.random.multivariate_normal(mean=np.zeros(p), cov=corZ)
    elif copulaInt == 'Student':
        # Student-t copula (approximate with high-df t)
        corZ = par.get('corZ', np.eye(p))
        df = par.get('df', 10)
        return stats.multivariate_t.rvs(loc=np.zeros(p), shape=corZ, df=df, size=1)
    else:
        # Default to Gaussian
        corZ = par.get('corZ', np.eye(p))
        return np.random.multivariate_normal(mean=np.zeros(p), cov=corZ)


def _sim_Zt_MAR(
    par: Dict,
    copulaInt: str,
    Yt_prev: np.ndarray,
    p: int
) -> np.ndarray:
    """
    Simulate multivariate autoregressive process.

    Parameters
    ----------
    par : dict
        Parameters including A, covZ
    copulaInt : str
        Copula type
    Yt_prev : ndarray, shape (p,)
        Previous step values
    p : int
        Number of stations

    Returns
    -------
    ndarray, shape (p,)
        Simulated values for current step
    """
    A = par.get('A', np.zeros((p, p)))
    covZ = par.get('covZ', np.eye(p))
    sdZ = par.get('sdZ', np.ones(p))

    # MAR(1): Y(t) = A @ Y(t-1) + e, where cov(e) = covZ - A covZ A^T.
    # When (A, covZ) are inconsistent (common with short MCMC chains),
    # cov_e can have negative eigenvalues.  Rather than patching cov_e
    # directly (which distorts the covariance structure), shrink A toward
    # zero until cov_e is PSD.  This conservatively reduces temporal
    # dependence when the model can't support it, preserving the fitted
    # spatial correlation structure in covZ.
    cov_e = covZ - A @ covZ @ A.T
    cov_e = 0.5 * (cov_e + cov_e.T)
    min_eig = np.min(np.linalg.eigvalsh(cov_e))

    if min_eig < 1e-6:
        # Binary search for largest shrinkage factor α ∈ [0, 1] such that
        # covZ - (α·A) covZ (α·A)^T is PSD.  α=0 always works (cov_e = covZ).
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            A_mid = mid * A
            ce = covZ - A_mid @ covZ @ A_mid.T
            ce = 0.5 * (ce + ce.T)
            if np.min(np.linalg.eigvalsh(ce)) >= 1e-6:
                lo = mid
            else:
                hi = mid
        A = lo * A
        cov_e = covZ - A @ covZ @ A.T
        cov_e = 0.5 * (cov_e + cov_e.T)

    mean_ar = A @ Yt_prev
    e = np.random.multivariate_normal(mean=np.zeros(p), cov=cov_e)

    return mean_ar + e


def disag_3D_to_1D(
    Yobs: np.ndarray,
    Y3obs: np.ndarray,
    mObs: np.ndarray,
    Y3sim: np.ndarray,
    mSim: np.ndarray,
    prob_class: np.ndarray
) -> Dict:
    """
    Disaggregate 3-day simulated precipitation to daily values.

    Parameters
    ----------
    Yobs : ndarray, shape (n_days, n_stations)
        Observed daily precipitation
    Y3obs : ndarray, shape (n_periods, n_stations)
        Observed 3-day aggregated precipitation
    mObs : ndarray, shape (n_periods,)
        Month/season for observed periods
    Y3sim : ndarray, shape (n_sim_periods, n_stations)
        Simulated 3-day precipitation
    mSim : ndarray, shape (n_sim_periods,)
        Month/season for simulated periods
    prob_class : ndarray
        Classification probabilities

    Returns
    -------
    dict
        Dictionary with disaggregated daily values and codes
    """
    # Classify periods into 4 classes
    n_obs = Y3obs.shape[0]
    n_sim = Y3sim.shape[0]
    n_stations = Y3obs.shape[1]

    # Simple classification: low, moderate, high, extreme
    class_obs = np.ones(n_obs, dtype=int)
    class_sim = np.ones(n_sim, dtype=int)

    # Call disaggregation function
    result = disag_3day_gwex_prec(
        Yobs=Yobs,
        Y3obs=Y3obs,
        mObs=mObs,
        cObs=class_obs,
        Y3sim=Y3sim,
        mSim=mSim,
        cSim=class_sim,
        nLagScore=1
    )

    return result
