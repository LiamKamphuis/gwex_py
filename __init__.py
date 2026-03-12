"""
GWEX-Py: Multi-Site Stochastic Model for Daily Precipitation
=============================================================

Python translation of the GWEX R package by Guillaume Evin (INRAE).

This package provides tools for fitting and simulating multi-site daily
precipitation using:
- Markov chain occurrence models with spatial correlation
- EGPD or mixture-of-exponentials marginal distributions
- Gaussian or Student copulas for spatial dependence
- Optional MAR(1) temporal dependence
- Optional 3-day disaggregation

References
----------
- Evin, G., A.-C. Favre, and B. Hingray. 2018. "Stochastic Generation of
  Multi-Site Daily Precipitation Focusing on Extreme Events." Hydrol. Earth
  Syst. Sci. 22(1): 655-72.
- Wilks, D.S. (1998) "Multisite generalization of a daily stochastic
  precipitation generation model", J Hydrol, 210: 178-191.

Usage
-----
>>> from gwex_py import GwexObs, fit_gwex_model, sim_gwex_model
>>> import numpy as np
>>>
>>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
>>> obs = GwexObs(variable='Prec', date=dates, obs=precip_matrix)
>>> fit = fit_gwex_model(obs, list_option={'th': 0.5})
>>> sim = sim_gwex_model(fit, nb_rep=100, d_start=dates[0], d_end=dates[-1])
"""

__version__ = "1.1.0"

from .core import GwexObs, GwexFit, GwexSim, fit_gwex_model, sim_gwex_model
from .distributions import (
    cdf_egpd_gi,
    pdf_egpd_gi,
    ppf_egpd_gi,
    rvs_egpd_gi,
    egpd_gi_fit_pwm,
    fit_margin_cdf,
    unif_to_prec,
)
from .utils import (
    get_list_option,
    modify_cor_matrix,
    dry_day_frequency,
    wet_day_frequency,
    lag_trans_proba_matrix,
)

__all__ = [
    # Core classes
    "GwexObs",
    "GwexFit",
    "GwexSim",
    # Main interface
    "fit_gwex_model",
    "sim_gwex_model",
    # Distribution functions
    "cdf_egpd_gi",
    "pdf_egpd_gi",
    "ppf_egpd_gi",
    "rvs_egpd_gi",
    "egpd_gi_fit_pwm",
    "fit_margin_cdf",
    "unif_to_prec",
    # Utilities
    "get_list_option",
    "modify_cor_matrix",
    "dry_day_frequency",
    "wet_day_frequency",
    "lag_trans_proba_matrix",
]
