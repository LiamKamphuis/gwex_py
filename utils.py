"""
Utility and helper functions for GWEX package.

Translated from R GWexPrec_lib.r by Guillaume Evin.
Python translation provides statistical utilities for precipitation analysis,
including dry/wet day frequencies, transition probabilities, correlation
matrices, and CDF calculations.
"""

import numpy as np
import pandas as pd
from itertools import product, combinations
from scipy import stats


def dry_day_frequency(mat_prec, th):
    """
    Calculate the proportion of dry days per station.

    A day is considered dry if precipitation is at or below the threshold.

    Parameters
    ----------
    mat_prec : ndarray
        Array of shape (n_days, n_stations) containing precipitation values.
    th : float
        Threshold above which a day is considered wet (e.g., 0.2 mm).

    Returns
    -------
    ndarray
        Vector of shape (n_stations,) with proportion of dry days per station.
    """
    # Proportion of days <= threshold (dry days) for each station (column)
    return np.nanmean(mat_prec <= th, axis=0)


def wet_day_frequency(mat_prec, th):
    """
    Calculate the proportion of wet days per station.

    A day is considered wet if precipitation is above the threshold.

    Parameters
    ----------
    mat_prec : ndarray
        Array of shape (n_days, n_stations) containing precipitation values.
    th : float
        Threshold above which a day is considered wet (e.g., 0.2 mm).

    Returns
    -------
    ndarray
        Vector of shape (n_stations,) with proportion of wet days per station.
    """
    # Proportion of days > threshold (wet days) for each station (column)
    return np.nanmean(mat_prec > th, axis=0)


def lag_trans_proba_vector(vec_prec, is_period, th, nlag):
    """
    Estimate transition probabilities for wet/dry states over nlag lags.

    Computes conditional probabilities Pr(today=wet | previous_nlag_states)
    for a single station.

    Parameters
    ----------
    vec_prec : ndarray
        Vector of shape (n_days,) with precipitation for one station.
    is_period : ndarray
        Boolean vector of shape (n_days,) indicating days in the 3-month period.
    th : float
        Threshold above which a day is considered wet.
    nlag : int
        Number of lag days to consider.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns t-nlag, ..., t-1 (boolean states),
        and P (conditional probability). Shape is (2^nlag, nlag+1).
    """
    ndays = nlag + 1

    # Get indices of days in this period, excluding those too close to the
    # end of the record (where the nlag look-ahead would exceed array bounds)
    ind_is_period = np.where(is_period)[0]
    ind_is_period = ind_is_period[ind_is_period + nlag < len(vec_prec)]

    # Build matrix of wet/dry indicators for current and previous nlag days
    iswetlag = np.zeros((len(ind_is_period), ndays), dtype=bool)
    for ilag in range(ndays):
        iswetlag[:, ilag] = vec_prec[ind_is_period + ilag] > th

    # Remove rows with NaN values
    nz = ~np.any(np.isnan(vec_prec[ind_is_period[:, np.newaxis] + np.arange(ndays)]), axis=1)
    x_nz = iswetlag[nz, :]

    # Generate all possible combinations of wet/dry states (2^ndays combinations)
    all_combos = list(product([False, True], repeat=ndays))

    # Create DataFrame with all combinations and compute probabilities
    comb_data = []
    for combo in all_combos:
        # Check how many rows match this combination
        matches = np.all(x_nz == combo, axis=1)
        prob = np.mean(matches) if len(x_nz) > 0 else 0.0
        row = list(combo) + [prob]
        comb_data.append(row)

    col_names = [f't{i}' for i in range(-nlag, 1)] + ['P']
    comb_pr = pd.DataFrame(comb_data, columns=col_names)

    # Compute conditional probabilities: Pr(t0=wet | t-nlag,...,t-1)
    # Split into FALSE and TRUE for current day (t0, last column before P)
    n_pr = len(comb_pr)
    # Cast to float ndarray explicitly so Pylance doesn't infer ExtensionArray
    pr_f = np.asarray(comb_pr['P'].iloc[:n_pr//2].values, dtype=float)
    pr_t = np.asarray(comb_pr['P'].iloc[n_pr//2:].values, dtype=float)

    # Conditional probability: P(t0=1 | past)
    pr_cond = pr_t / (pr_f + pr_t)
    pr_cond[pr_f + pr_t == 0] = 0.0

    # Return only the rows where t0=TRUE with past states and conditional probability
    result_cols = [f't{i}' for i in range(-nlag, 0)] + ['P']
    result = comb_pr.iloc[n_pr//2:, :].copy()
    result['P'] = pr_cond

    return result[result_cols].reset_index(drop=True)


def lag_trans_proba_matrix(mat_prec, is_period, th, nlag):
    """
    Estimate transition probabilities for all stations.

    Parameters
    ----------
    mat_prec : ndarray
        Array of shape (n_days, n_stations) with precipitation.
    is_period : ndarray
        Boolean vector of shape (n_days,) indicating days in the 3-month period.
    th : float
        Threshold above which a day is considered wet.
    nlag : int
        Number of lag days.

    Returns
    -------
    dict
        Dictionary with one entry per station, where each value is a DataFrame
        of transition probabilities (output from lag_trans_proba_vector).
    """
    n_stations = mat_prec.shape[1]
    result = {}

    for i in range(n_stations):
        result[i] = lag_trans_proba_vector(mat_prec[:, i], is_period, th, nlag)

    return result


def modify_cor_matrix(cor_matrix):
    """
    Fix non-positive-definite correlation matrix.

    Uses eigenvalue decomposition: replaces negative eigenvalues with small
    positive values (1e-10), then reconstructs and re-normalizes.

    References
    ----------
    Rousseeuw, P. J. and G. Molenberghs. 1993. Transformation of non positive
    semidefinite correlation matrices. Communications in Statistics: Theory
    and Methods 22(4):965-984.

    Rebonato, R., & Jackel, P. (2000). The most general methodology to create
    a valid correlation matrix for risk management and option pricing purposes.
    J. Risk, 2(2), 17-26.

    Parameters
    ----------
    cor_matrix : ndarray
        Possibly non-positive-definite correlation matrix.

    Returns
    -------
    ndarray
        Positive-definite correlation matrix.
    """
    # Eigendecomposition
    eig_val, eig_vec = np.linalg.eigh(cor_matrix)

    # Check for negative eigenvalues
    is_neg_eig = eig_val < 0

    if np.any(is_neg_eig):
        # Replace negative eigenvalues with small positive value
        eig_val[is_neg_eig] = 1e-10
        eig_diag = np.diag(eig_val)

        # Reconstruct correlation matrix: Q @ Lambda @ Q^T
        cor_recon = eig_vec @ eig_diag @ eig_vec.T

        # Normalize by diagonal to get correlation matrix (diagonal = 1)
        diag_sqrt = np.sqrt(np.diag(cor_recon))
        cor_matrix_out = cor_recon / np.outer(diag_sqrt, diag_sqrt)
    else:
        cor_matrix_out = cor_matrix.copy()

    # Ensure diagonal is exactly 1 (numerical precision)
    np.fill_diagonal(cor_matrix_out, 1.0)

    return cor_matrix_out


def get_df_student(P, Sig, max_df=20):
    """
    Estimate degrees of freedom for multivariate Student-t distribution.

    Finds the degrees of freedom that maximizes the log-likelihood of the
    multivariate Student-t distribution fitted to precipitation data.

    References
    ----------
    McNeil et al. (2005) "Quantitative Risk Management"

    Parameters
    ----------
    P : ndarray
        Array of shape (n_days, n_stations) of positive precipitation values.
        Zero/missing values should be marked as NaN.
    Sig : ndarray
        Correlation matrix of shape (n_stations, n_stations).
    max_df : int, optional
        Maximum degrees of freedom to test (default=20).

    Returns
    -------
    int
        Estimated degrees of freedom.
    """
    # Transform to probability integral transform (empirical CDF)
    U = get_emp_cdf_matrix(P)

    # Compute likelihood for each df from 1 to max_df
    vec_lk = np.full(max_df, np.nan)

    for df in range(1, max_df + 1):
        # Transform uniform to Student-t quantiles
        t_data = stats.t.ppf(U, df)

        # Log-likelihood: multivariate Student-t minus marginal Student-t
        try:
            mv_logpdf = stats.multivariate_t.logpdf(t_data, loc=np.zeros(P.shape[1]),
                                                    shape=Sig, df=df)
            marg_logpdf = np.sum(stats.t.logpdf(t_data, df), axis=1)
            lk = mv_logpdf - marg_logpdf
        except:
            lk = np.full(P.shape[0], np.nan)

        # Sum log-likelihoods (ignoring NaNs)
        nz = ~np.isnan(lk)
        if np.any(nz):
            vec_lk[df - 1] = np.sum(lk[nz])
        else:
            vec_lk[df - 1] = np.nan

    # Find df that maximizes likelihood
    df_hat = np.nanargmax(vec_lk) + 1

    # If likelihood is finite, return df_hat; otherwise return max_df + 10
    if np.isfinite(vec_lk[df_hat - 1]):
        return df_hat
    else:
        return max_df + 10


def get_emp_cdf_matrix(X):
    """
    Compute empirical CDF values using Gringorten plotting position.

    The Gringorten plotting position (a=0.44) is optimized for Gumbel
    distribution. For positive precipitation values, computes empirical
    cumulative probabilities. NaN for zero/missing values.

    Parameters
    ----------
    X : ndarray
        Array of shape (n_days, n_stations) of positive precipitation.
        Zero/missing values are NaN.

    Returns
    -------
    ndarray
        Array of same shape as X with empirical CDF values (or NaN).
    """
    n_days, n_stations = X.shape
    Y = np.full((n_days, n_stations), np.nan)

    # Process each station (column)
    for i in range(n_stations):
        x_i = X[:, i]
        # Identify non-NaN values
        nz = ~np.isnan(x_i)

        if np.any(nz):
            x_valid = x_i[nz]
            # Compute ranks (1-indexed)
            ranks = stats.rankdata(x_valid)
            n = len(x_valid)
            # Gringorten plotting position: (rank - 0.44) / (n + 0.12)
            cdf_vals = (ranks - 0.44) / (n + 0.12)
            # Assign back to output array
            Y[nz, i] = cdf_vals

    return Y


def get_list_month():
    """
    Get list of 12 month abbreviations.

    Returns
    -------
    list
        List of 3-letter month abbreviations.
    """
    return ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
            'JUL', 'AOU', 'SEP', 'OCT', 'NOV', 'DEC']


def get_period_fitting_month(m_char):
    """
    Get the 3-month fitting window for a given month.

    Returns the 3 month indices centered on the input month
    (previous, current, next).

    Parameters
    ----------
    m_char : str
        3-letter month abbreviation (e.g., 'JAN').

    Returns
    -------
    ndarray
        Array of 3 month indices (1-12) for the fitting window.
    """
    list_m = get_list_month()

    # Find index of input month (0-indexed)
    try:
        i_m = list_m.index(m_char)
    except ValueError:
        raise ValueError(f"Invalid month: {m_char}")

    # Create extended list with December at start and January at end
    vec_m_ext = [12] + list(range(1, 13)) + [1]

    # Return 3-month window
    return np.array(vec_m_ext[i_m : i_m + 3])


def month2season(vec_month):
    """
    Convert month numbers to season indices.

    Season mapping: Dec/Jan/Feb=1, Mar/Apr/May=2, Jun/Jul/Aug=3, Sep/Oct/Nov=4

    Parameters
    ----------
    vec_month : array-like
        Vector of month numbers (1-12).

    Returns
    -------
    ndarray
        Vector of season indices (1-4).
    """
    # Season mapping: month -> season
    season_map = np.array([1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1])

    # Convert to numpy array if needed
    vec_month = np.asarray(vec_month)

    # Map months (1-12) to seasons (1-indexed)
    return season_map[vec_month - 1]


def agg_matrix(mat, k, average=False):
    """
    Temporally aggregate a matrix by summing over k consecutive rows.

    Parameters
    ----------
    mat : ndarray
        Array of shape (n_rows, n_cols) to aggregate.
    k : int
        Aggregation period (number of rows to sum).
    average : bool, optional
        If True, return averages instead of sums (default=False).

    Returns
    -------
    ndarray
        Aggregated array of shape (n_rows // k, n_cols).
    """
    # Number of complete aggregation windows
    n = int(np.floor(mat.shape[0] / k))
    n_cols = mat.shape[1]
    mat_agg = np.zeros((n, n_cols))

    # Aggregate by summing k consecutive rows
    for i in range(k):
        mat_agg += mat[i::k][:n, :]

    # Average if requested
    if average:
        mat_agg = mat_agg / k

    return mat_agg


def get_list_option(list_option):
    """
    Get default options and validate user-provided options.

    Validates and sets default values for all GWEX options including
    precipitation threshold, lag order, margin type, copula type, etc.

    Parameters
    ----------
    list_option : dict or None
        Dictionary with option keys and values. Can be None or empty.

    Returns
    -------
    dict
        Dictionary with all options set (defaults + user overrides).

    Raises
    ------
    ValueError
        If any option value is invalid.
    """
    if list_option is None:
        list_option = {}
    else:
        list_option = dict(list_option)  # Copy to avoid modifying input

    # ===== Options for occurrence =====

    # Wet day threshold
    if 'th' in list_option:
        th = list_option['th']
        if not (isinstance(th, (int, float)) and th >= 0):
            raise ValueError('th must be numeric and >= 0')
    else:
        th = 0.2
        list_option['th'] = th

    # Number of lags
    if 'nLag' in list_option:
        nlag = list_option['nLag']
        if nlag not in [1, 2, 3, 4, 5]:
            raise ValueError('nLag must be between 1 and 5')
    else:
        nlag = 2
        list_option['nLag'] = nlag

    # ===== Options for amount =====

    # Margin type
    if 'typeMargin' in list_option:
        type_margin = list_option['typeMargin']
        if type_margin not in ['mixExp', 'EGPD']:
            raise ValueError("typeMargin must be 'mixExp' or 'EGPD'")
    else:
        type_margin = 'mixExp'
        list_option['typeMargin'] = type_margin

    # Copula type
    if 'copulaInt' in list_option:
        copula_int = list_option['copulaInt']
        if copula_int not in ['Gaussian', 'Student']:
            raise ValueError("copulaInt must be 'Gaussian' or 'Student'")
    else:
        copula_int = 'Gaussian'
        list_option['copulaInt'] = copula_int

    # Multivariate Autoregressive
    if 'isMAR' in list_option:
        is_mar = list_option['isMAR']
        if not isinstance(is_mar, (bool, np.bool_)):
            raise ValueError('isMAR must be boolean')
    else:
        is_mar = False
        list_option['isMAR'] = is_mar

    # 3D amount
    if 'is3Damount' in list_option:
        is_3d_amount = list_option['is3Damount']
        if not isinstance(is_3d_amount, (bool, np.bool_)):
            raise ValueError('is3Damount must be boolean')
    else:
        is_3d_amount = False
        list_option['is3Damount'] = is_3d_amount

    # Number of chain iterations for fitting
    if 'nChainFit' in list_option:
        n_chain_fit = list_option['nChainFit']
        if not isinstance(n_chain_fit, (int, np.integer)):
            raise ValueError('nChainFit must be an integer')
    else:
        n_chain_fit = 100000
        list_option['nChainFit'] = n_chain_fit

    # Number of clusters/parallel workers
    if 'nCluster' in list_option:
        n_cluster = list_option['nCluster']
        if not isinstance(n_cluster, (int, np.integer)):
            raise ValueError('nCluster must be an integer')
    else:
        n_cluster = 1
        list_option['nCluster'] = n_cluster

    # Parallel computation flag
    is_parallel = n_cluster > 1
    list_option['isParallel'] = is_parallel

    return list_option


def joint_proba_occ(P, th):
    """
    Compute joint occurrence probabilities for all station pairs.

    For each pair of stations (i, j), computes the four joint probabilities:
    - p00: Pr(dry at i AND dry at j)
    - p01: Pr(dry at i AND wet at j)
    - p10: Pr(wet at i AND dry at j)
    - p11: Pr(wet at i AND wet at j)

    Parameters
    ----------
    P : ndarray
        Array of shape (n_days, n_stations) with precipitation.
    th : float
        Wet/dry threshold.

    Returns
    -------
    dict
        Dictionary with keys 'p00', 'p01', 'p10', 'p11', each containing
        an array of shape (n_stations, n_stations) with joint probabilities.
    """
    p = P.shape[1]

    # Initialize output matrices (diagonal = 1)
    p00 = np.ones((p, p))
    p01 = np.ones((p, p))
    p10 = np.ones((p, p))
    p11 = np.ones((p, p))

    # Compute for all pairs (i < j)
    for i in range(p - 1):
        ri = P[:, i]
        for j in range(i + 1, p):
            rj = P[:, j]

            # Valid data (non-NaN for both stations)
            nz = (~np.isnan(ri)) & (~np.isnan(rj))

            if np.sum(nz) == 0:
                raise ValueError(
                    f"fitGwexModel fails: too many missing values at columns "
                    f"{i} and {j} (no simultaneous data available for this "
                    f"month and pair of stations)"
                )

            ri_valid = ri[nz]
            rj_valid = rj[nz]

            # Dry-dry
            p00_ij = np.mean((ri_valid <= th) & (rj_valid <= th))
            p00[i, j] = p00_ij
            p00[j, i] = p00_ij

            # Dry-wet
            p01_ij = np.mean((ri_valid <= th) & (rj_valid > th))
            p01[i, j] = p01_ij
            p10[j, i] = p01_ij

            # Wet-dry
            p10_ij = np.mean((ri_valid > th) & (rj_valid <= th))
            p10[i, j] = p10_ij
            p01[j, i] = p10_ij

            # Wet-wet
            p11_ij = np.mean((ri_valid > th) & (rj_valid > th))
            p11[i, j] = p11_ij
            p11[j, i] = p11_ij

    return {'p00': p00, 'p01': p01, 'p10': p10, 'p11': p11}


def cor_obs_occ(pi00, pi0, pi1):
    """
    Compute observed correlations between dry/wet occurrences.

    Uses the formula from Mhanna et al. (2012) Eq. 6:
    cor(i,j) = (pi00[i,j] - pi0[i]*pi0[j]) / sqrt(pi0[i]*pi1[i]*pi0[j]*pi1[j])

    References
    ----------
    Mhanna, Muamaraldin, and Willy Bauwens. "A Stochastic Space-Time Model
    for the Generation of Daily Rainfall in the Gaza Strip." International
    Journal of Climatology 32, no. 7 (June 15, 2012): 1098–1112.
    doi:10.1002/joc.2305.

    Parameters
    ----------
    pi00 : ndarray
        Array of shape (n_stations, n_stations) with Pr(dry at i AND dry at j).
    pi0 : ndarray
        Vector of shape (n_stations,) with Pr(dry at station).
    pi1 : ndarray
        Vector of shape (n_stations,) with Pr(wet at station).

    Returns
    -------
    ndarray
        Array of shape (n_stations, n_stations) with occurrence correlations.
    """
    p = pi00.shape[0]

    # Initialize with 1s (diagonal and undefined values)
    cor_obs = np.ones((p, p))

    # Compute correlation for all pairs (i < j)
    for i in range(p - 1):
        for j in range(i + 1, p):
            # Standard deviation of occurrence at each station
            sig_i = np.sqrt(pi0[i] * pi1[i])
            sig_j = np.sqrt(pi0[j] * pi1[j])

            # Eq. 6 from Mhanna (2012)
            cor_ij = (pi00[i, j] - pi0[i] * pi0[j]) / (sig_i * sig_j)

            # Assign to both [i,j] and [j,i]
            cor_obs[i, j] = cor_ij
            cor_obs[j, i] = cor_ij

    return cor_obs
