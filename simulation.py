"""
GWEX precipitation occurrence and disaggregation simulation functions.

Translation from C++ (Rcpp) to Python/NumPy of the GWEX package's
toolsSimDisag.cpp module for simulating precipitation occurrences
via Markov chains and disaggregating 3-day precipitation to daily values.
"""

import numpy as np
from typing import Tuple, Dict, List


def find_row(m: np.ndarray, v: np.ndarray) -> int:
    """
    Find the index of the row in matrix m that equals vector v.

    Parameters
    ----------
    m : ndarray
        2D array of possible combinations (n_comb x nLag)
    v : ndarray
        1D vector to search for in rows of m

    Returns
    -------
    int
        Index of matching row, or -1 if not found
    """
    for i in range(m.shape[0]):
        if np.array_equal(m[i, :], v):
            return i
    return -1


def sim_precip_occurrences(
    nLag: int,
    matcomb: np.ndarray,
    Qtrans: np.ndarray,
    rndNorm: np.ndarray
) -> np.ndarray:
    """
    Simulate precipitation occurrences using a Markov chain with time-varying quantiles.

    Generates a binary sequence of wet (1) / dry (0) occurrences where the transition
    probability depends on the previous nLag states.

    Parameters
    ----------
    nLag : int
        Number of lag days to condition on
    matcomb : ndarray, shape (n_comb, nLag)
        Matrix of all possible wet/dry combinations
    Qtrans : ndarray, shape (n, n_comb)
        Time-varying transition quantiles. Qtrans[t, row_idx] is the quantile
        threshold at time t for combination row_idx
    rndNorm : ndarray, shape (n,)
        Random normal deviates for the entire simulation period

    Returns
    -------
    ndarray, shape (n,)
        Binary occurrence sequence (0=dry, 1=wet)
    """
    n = len(rndNorm)
    Xt = np.zeros(n, dtype=float)

    # Fill first nLag time steps with random 0/1 with probability 0.5
    Xt[:nLag] = (np.random.rand(nLag) < 0.5).astype(float)

    # Simulate the chain
    for t in range(nLag, n):
        # Get previous nLag occurrences
        comb = Xt[t - nLag:t]

        # Find matching row in matcomb
        row = find_row(matcomb, comb)

        # Compare random normal to quantile threshold
        Xt[t] = float(rndNorm[t] <= Qtrans[t, row])

    return Xt


def sim_precip_occurrences_4_fitting(
    nLag: int,
    nChainFit: int,
    matcomb: np.ndarray,
    Qtrans_vec: np.ndarray,
    rndNorm: np.ndarray
) -> np.ndarray:
    """
    Simulate precipitation occurrences with fixed (non-time-varying) quantiles.

    Similar to sim_precip_occurrences but uses a constant quantile vector
    and returns only the last nChainFit values (useful for model fitting).

    Parameters
    ----------
    nLag : int
        Number of lag days to condition on
    nChainFit : int
        Number of final chain steps to return
    matcomb : ndarray, shape (n_comb, nLag)
        Matrix of all possible wet/dry combinations
    Qtrans_vec : ndarray, shape (n_comb,)
        Fixed transition quantiles for each combination
    rndNorm : ndarray, shape (n,)
        Random normal deviates for the entire simulation period

    Returns
    -------
    ndarray, shape (nChainFit,)
        Last nChainFit values of the binary occurrence sequence
    """
    n = len(rndNorm)
    Xt = np.zeros(n, dtype=float)

    # Fill first nLag time steps with random 0/1 with probability 0.5
    Xt[:nLag] = (np.random.rand(nLag) < 0.5).astype(float)

    # Simulate the chain
    for t in range(nLag, n):
        # Get previous nLag occurrences
        comb = Xt[t - nLag:t]

        # Find matching row in matcomb
        row = find_row(matcomb, comb)

        # Compare random normal to quantile threshold
        Xt[t] = float(rndNorm[t] <= Qtrans_vec[row])

    # Return only last nChainFit values
    return Xt[n - nChainFit:n]


def pearson_rho(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two vectors.

    Parameters
    ----------
    x : ndarray
        First vector
    y : ndarray
        Second vector

    Returns
    -------
    float
        Pearson correlation coefficient
    """
    # Using NumPy's corrcoef is simpler and more robust
    corr_matrix = np.corrcoef(x, y)
    return corr_matrix[0, 1]


def cor_markov_chain(
    rndNorm: np.ndarray,
    QtransMat: np.ndarray,
    matcomb: np.ndarray,
    nChainFit: int,
    nLag: int
) -> float:
    """
    Simulate two-station Markov chain and compute correlation between occurrences.

    Generates occurrence sequences for two stations independently using a shared
    matcomb but station-specific quantile matrices, then returns their correlation.

    Parameters
    ----------
    rndNorm : ndarray, shape (n, 2)
        Bivariate normal random deviates
    QtransMat : ndarray, shape (2, n_comb)
        Transition quantiles for each station
    matcomb : ndarray, shape (n_comb, nLag)
        Matrix of all possible wet/dry combinations
    nChainFit : int
        Number of final chain steps to use for correlation
    nLag : int
        Number of lag days

    Returns
    -------
    float
        Pearson correlation between the two occurrence sequences
    """
    n = rndNorm.shape[0]
    Xt = np.zeros((n, 2), dtype=float)

    # Initialize first nLag time steps: 1 if rndNorm < 0, else 0
    Xt[:nLag, 0] = (rndNorm[:nLag, 0] < 0).astype(float)
    Xt[:nLag, 1] = (rndNorm[:nLag, 1] < 0).astype(float)

    # Simulate the chain for both stations
    for t in range(nLag, n):
        for st in range(2):
            # Get previous nLag occurrences for this station
            comb = Xt[t - nLag:t, st]

            # Find matching row in matcomb
            iComb = find_row(matcomb, comb)

            # Compare random normal to quantile threshold
            Xt[t, st] = float(rndNorm[t, st] <= QtransMat[st, iComb])

    # Compute correlation on last nChainFit values
    series_0 = Xt[n - nChainFit:n, 0]
    series_1 = Xt[n - nChainFit:n, 1]

    return pearson_rho(series_0, series_1)


def _matrix_subcol(X: np.ndarray, i0: int, i1: int, icol: int) -> np.ndarray:
    """
    Extract a subvector from a specific column of a matrix.

    Parameters
    ----------
    X : ndarray
        2D array
    i0 : int
        Starting row (inclusive)
    i1 : int
        Ending row (inclusive)
    icol : int
        Column index

    Returns
    -------
    ndarray
        1D subvector X[i0:i1+1, icol]
    """
    return X[i0:i1 + 1, icol]


def getrmsei(
    Yobs: np.ndarray,
    Y3obs: np.ndarray,
    mObs: np.ndarray,
    cObs: np.ndarray,
    Y3sim: np.ndarray,
    mSim: np.ndarray,
    cSim: np.ndarray,
    nLagScore: int,
    i: int,
    Ysimilag: np.ndarray
) -> np.ndarray:
    """
    Compute RMSE-based scores for disaggregation field matching.

    For a given simulated 3-day period, computes a similarity score with each
    observed 3-day period. The score includes the current 3-day period and
    (optionally) previous daily values for temporal continuity.

    Parameters
    ----------
    Yobs : ndarray, shape (nTobs*3, nStat)
        Daily observed precipitation
    Y3obs : ndarray, shape (nTobs, nStat)
        3-day aggregated observed precipitation
    mObs : ndarray, shape (nTobs,)
        Month index for each observed 3-day period
    cObs : ndarray, shape (nTobs,)
        Precipitation class for each observed 3-day period
    Y3sim : ndarray, shape (nTsim, nStat)
        3-day aggregated simulated precipitation
    mSim : ndarray, shape (nTsim,)
        Month index for each simulated 3-day period
    cSim : ndarray, shape (nTsim,)
        Precipitation class for each simulated 3-day period
    nLagScore : int
        Number of lag days to include in score (usually 1)
    i : int
        Index of current simulated 3-day period
    Ysimilag : ndarray, shape (3, nStat)
        Simulated daily precipitation for 3 days before period i

    Returns
    -------
    ndarray, shape (nTobs,)
        Score for each observed 3-day period (lower is better)
    """
    nTobs = Y3obs.shape[0]
    nStat = Y3obs.shape[1]
    NA_VAL = -9999.0

    rmseI = np.full(nTobs, np.inf)

    if i < nLagScore:
        # For first time steps: no previous lag information available
        for j in range(nTobs):
            # Check if month and class match
            not_same_class = (mSim[i] != mObs[j]) or (cSim[i] != cObs[j])

            # Check for missing values
            if np.any(Y3obs[j, :] == NA_VAL) or not_same_class:
                rmseI[j] = 1e30
            else:
                # Compute adimensionalized (normalized) difference
                Y3sim_sum = np.sum(Y3sim[i, :])
                Y3obs_sum = np.sum(Y3obs[j, :])

                if Y3sim_sum == 0:
                    adim_sim = Y3sim[i, :]
                else:
                    adim_sim = Y3sim[i, :] / Y3sim_sum

                if Y3obs_sum == 0:
                    adim_obs = Y3obs[j, :]
                else:
                    adim_obs = Y3obs[j, :] / Y3obs_sum

                rmseI[j] = np.sum(np.abs(adim_sim - adim_obs))
    else:
        # For subsequent time steps: include lag information
        # Discard first nLagScore observations (insufficient historical data)
        rmseI[:nLagScore] = 2e30

        for j in range(nLagScore, nTobs):
            # Check if month and class match
            not_same_class = (mSim[i] != mObs[j]) or (cSim[i] != cObs[j])

            # Check for missing values
            if np.any(Y3obs[j, :] == NA_VAL) or not_same_class:
                rmseI[j] = 3e30
            else:
                # Compute current 3-day period score
                Y3sim_sum = np.sum(Y3sim[i, :])
                Y3obs_sum = np.sum(Y3obs[j, :])

                if Y3sim_sum == 0:
                    adim_sim = Y3sim[i, :]
                else:
                    adim_sim = Y3sim[i, :] / Y3sim_sum

                if Y3obs_sum == 0:
                    adim_obs = Y3obs[j, :]
                else:
                    adim_obs = Y3obs[j, :] / Y3obs_sum

                rmseIJ = np.sum(np.abs(adim_sim - adim_obs))

                # Add lag scores from previous days
                # jDay is the daily index just before the current 3-day period
                jDay = j * 3 - 1

                for iLag in range(nLagScore):
                    obs_lag = Yobs[jDay - iLag, :]

                    if np.any(obs_lag == NA_VAL):
                        rmseIJ = 4e30
                        break

                    # Adimensionalize lag values
                    sim_lag = Ysimilag[2 - iLag, :]
                    sim_lag_sum = np.sum(sim_lag)
                    obs_lag_sum = np.sum(obs_lag)

                    if sim_lag_sum == 0:
                        adim_sim = sim_lag
                    else:
                        adim_sim = sim_lag / sim_lag_sum

                    if obs_lag_sum == 0:
                        adim_obs = obs_lag
                    else:
                        adim_obs = obs_lag / obs_lag_sum

                    rmseIJ = rmseIJ + np.sum(np.abs(adim_sim - adim_obs))

                rmseI[j] = rmseIJ

    return rmseI


def disag_3day_gwex_prec(
    Yobs: np.ndarray,
    Y3obs: np.ndarray,
    mObs: np.ndarray,
    cObs: np.ndarray,
    Y3sim: np.ndarray,
    mSim: np.ndarray,
    cSim: np.ndarray,
    nLagScore: int = 1
) -> Dict:
    """
    Disaggregate 3-day simulated precipitation to daily values using observed patterns.

    For each simulated 3-day period, finds the closest observed 3-day fields
    (by month, class, and structure similarity) and uses their daily patterns
    to disaggregate the simulated 3-day total.

    Parameters
    ----------
    Yobs : ndarray, shape (nTobs*3, nStat)
        Daily observed precipitation
    Y3obs : ndarray, shape (nTobs, nStat)
        3-day aggregated observed precipitation
    mObs : ndarray, shape (nTobs,)
        Month index (1-12) for each observed 3-day period
    cObs : ndarray, shape (nTobs,)
        Precipitation class (1-4) for each observed 3-day period
    Y3sim : ndarray, shape (nTsim, nStat)
        3-day aggregated simulated precipitation
    mSim : ndarray, shape (nTsim,)
        Month index (1-12) for each simulated 3-day period
    cSim : ndarray, shape (nTsim,)
        Precipitation class (1-4) for each simulated 3-day period
    nLagScore : int
        Number of lag days for matching similarity (default: 1)

    Returns
    -------
    dict
        'Ysim': ndarray, shape (nTsim*3, nStat) - disaggregated daily precipitation
        'codeDisag': ndarray, shape (nTsim, nStat) - code indicating disaggregation method:
                    -1000: no precipitation (all zeros)
                     1-10: matched to observed field (1-10 indicates which best match)
                     2000: fell back to random selection from best matches
                    +10000: disaggregated amounts exceed observed maximum
    """
    nTobs = Y3obs.shape[0]
    nStat = Y3obs.shape[1]
    nTsim = Y3sim.shape[0]
    nBestField = 10
    NA_VAL = -9999.0

    # Initialize output arrays
    Ysim = np.zeros((nTsim * 3, nStat), dtype=float)
    codeDisag = np.full((nTsim, nStat), NA_VAL, dtype=float)

    # Process each simulated 3-day period
    for i in range(nTsim):
        # Daily index just before this 3-day period
        iDay = i * 3 - 1

        # Prepare lag information if available
        Ysimilag = np.zeros((3, nStat), dtype=float)
        if i > 0:
            for ilag in range(3):
                Ysimilag[ilag, :] = Ysim[iDay + ilag - 2, :]

        # Compute RMSE scores for matching
        rmseI = getrmsei(Yobs, Y3obs, mObs, cObs, Y3sim, mSim, cSim, nLagScore, i, Ysimilag)

        # Find 10 best matching observed fields (argsort returns indices sorted ascending)
        best_indices = np.argsort(rmseI)
        indBestFieldI = best_indices[:nBestField]

        # Process each station
        for k in range(nStat):
            codeDisag[i, k] = NA_VAL

            # Case 1: No precipitation to disaggregate
            if Y3sim[i, k] == 0:
                codeDisag[i, k] = -1000
                for ii in range(iDay + 1, iDay + 4):
                    Ysim[ii, k] = 0.0
            else:
                # Case 2: Loop through best matching fields
                found_match = False
                for j in range(nBestField):
                    jBest = indBestFieldI[j]
                    jDay = jBest * 3 - 1

                    # Check if there's observed precipitation in this field
                    if Y3obs[jBest, k] > 0:
                        codeDisag[i, k] = j + 1

                        # Extract the 3 observed daily values
                        Yobs3D = _matrix_subcol(Yobs, jDay + 1, jDay + 3, k)

                        # Disaggregate by rescaling observed daily pattern
                        obs_sum = np.sum(Yobs3D)
                        for ii in range(iDay + 1, iDay + 4):
                            Ysim[ii, k] = Yobs3D[ii - iDay - 1] * Y3sim[i, k] / obs_sum

                        # Check if disaggregated values exceed observed maximum
                        disag_vals = _matrix_subcol(Ysim, iDay + 1, iDay + 3, k)
                        if np.any(disag_vals > np.max(Yobs[:, k])):
                            codeDisag[i, k] = codeDisag[i, k] + 10000

                        found_match = True
                        break

                # Case 3: If no structural match found, use random selection from best matches
                if not found_match:
                    # Compute simple amount-based scores
                    simple_rmse = np.full(nTobs, np.inf)

                    if i < nLagScore:
                        # First time steps: only match on current amount
                        for j in range(nTobs):
                            not_same_class = (mSim[i] != mObs[j]) or (cSim[i] != cObs[j])
                            if (Y3obs[j, k] == NA_VAL) or not_same_class:
                                simple_rmse[j] = 1e30
                            else:
                                simple_rmse[j] = np.abs(Y3sim[i, k] - Y3obs[j, k])
                    else:
                        # Subsequent steps: include lag information
                        simple_rmse[:nLagScore] = 1e30

                        for j in range(nLagScore, nTobs):
                            not_same_class = (mSim[i] != mObs[j]) or (cSim[i] != cObs[j])
                            if ((Y3obs[j, k] == NA_VAL) or (Y3obs[j, k] == 0) or not_same_class):
                                simple_rmse[j] = 1e30
                            else:
                                rmseIJ = np.abs(Y3sim[i, k] - Y3obs[j, k])
                                jDay = j * 3 - 1

                                # Check lag values
                                lag_vals = _matrix_subcol(Yobs, jDay - nLagScore + 1, jDay, k)
                                if np.any(lag_vals == NA_VAL):
                                    rmseIJ = 1e30
                                else:
                                    for iLag in range(nLagScore):
                                        rmseIJ = rmseIJ + np.abs(Ysim[iDay - iLag, k] -
                                                                  Yobs[jDay - iLag, k])

                                simple_rmse[j] = rmseIJ

                    # Get 10 best matches and randomly select one
                    best_simple = np.argsort(simple_rmse)[:nBestField]
                    rnd_idx = int(np.floor(np.random.rand() * 10))
                    j3Day = best_simple[rnd_idx]
                    jDay = j3Day * 3 - 1

                    codeDisag[i, k] = 2000
                    Yobs3D = _matrix_subcol(Yobs, jDay + 1, jDay + 3, k)
                    obs_sum = np.sum(Yobs3D)
                    for ii in range(iDay + 1, iDay + 4):
                        Ysim[ii, k] = Yobs3D[ii - iDay - 1] * Y3sim[i, k] / obs_sum

                    # Check if disaggregated values exceed observed maximum
                    disag_vals = _matrix_subcol(Ysim, iDay + 1, iDay + 3, k)
                    if np.any(disag_vals > np.max(Yobs[:, k])):
                        codeDisag[i, k] = codeDisag[i, k] + 10000

    return {
        'codeDisag': codeDisag,
        'Ysim': Ysim
    }
