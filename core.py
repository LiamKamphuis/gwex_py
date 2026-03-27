"""
GWex Core Module - Class definitions and main interface

Translates R's S4 classes (Gwex, GwexObs, GwexFit, GwexSim) to Python dataclasses.
Provides main interface functions for fitting and simulating GWEX models.

Author: Claude (translated from Guillaume Evin's R code)
Version: v1.1.0
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


GWEX_VERSION = "v1.1.0"


@dataclass
class GwexObs:
    """
    Observation container for GWEX model.

    Holds observed time series data (precipitation or temperature) at multiple
    stations over a period of time.

    Attributes:
        variable: Type of variable ('Prec' or 'Temp')
        date: numpy array of datetime64[D] with shape (nTime,)
        obs: numpy array of observations with shape (nTime, nStations)
        version: GWEX package version

    Examples:
        >>> import numpy as np
        >>> from datetime import datetime
        >>>
        >>> # Create date range for 2005-2014
        >>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
        >>> # Create sample precipitation data: 3652 days x 5 stations
        >>> obs = np.random.rand(len(dates), 5) * 10
        >>>
        >>> # Create GwexObs for precipitation
        >>> gwex_obs = GwexObs(variable='Prec', date=dates, obs=obs)
        >>> print(gwex_obs.n_stations)
        5
    """
    variable: str
    date: np.ndarray
    obs: np.ndarray
    version: str = field(default=GWEX_VERSION)

    def __post_init__(self):
        """Validate inputs after initialization."""
        # Validate variable type
        if self.variable not in ('Prec', 'Temp'):
            raise ValueError("variable must be 'Prec' or 'Temp'")

        # Validate date is 1D
        if self.date.ndim != 1:
            raise ValueError("date must be a 1-dimensional array")

        # Validate date dtype
        if not np.issubdtype(self.date.dtype, np.datetime64):
            raise ValueError("date must be an array of datetime64")

        # Validate obs is 2D
        if self.obs.ndim != 2:
            raise ValueError("obs must be a 2-dimensional array (nTime x nStations)")

        # Validate shapes match
        if self.obs.shape[0] != len(self.date):
            raise ValueError(
                f"obs must have the same number of rows ({self.obs.shape[0]}) "
                f"as the date array ({len(self.date)})"
            )

        # Ensure obs is float array
        self.obs = np.asarray(self.obs, dtype=np.float64)

        # Ensure date is datetime64[D]
        if self.date.dtype != np.dtype('datetime64[D]'):
            self.date = self.date.astype('datetime64[D]')

    @property
    def n_stations(self) -> int:
        """Get the number of stations."""
        return self.obs.shape[1]

    @property
    def n_times(self) -> int:
        """Get the number of time steps."""
        return self.obs.shape[0]

    def __repr__(self) -> str:
        """String representation."""
        date_min = self.date[0]
        date_max = self.date[-1]
        return (
            f"GwexObs(variable='{self.variable}', "
            f"period={date_min} -> {date_max}, "
            f"n_stations={self.n_stations}, n_times={self.n_times})"
        )


@dataclass
class GwexFit:
    """
    Fitted GWEX model container.

    Holds the fitted parameters and options for a GWEX model after fitting
    to observed data.

    Attributes:
        variable: Type of variable ('Prec' or 'Temp')
        fit: Dictionary containing fitted model results with keys:
            - 'listOption': dict of fitting options
            - 'listPar': dict of fitted parameters with subkeys:
              - 'parOcc': occurrence parameters (for Prec)
              - 'parInt': intensity parameters
        p: Number of stations
        version: GWEX package version

    Examples:
        >>> from core import GwexObs, fit_gwex_model
        >>> import numpy as np
        >>>
        >>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
        >>> obs = np.random.rand(len(dates), 3) * 10
        >>> gwex_obs = GwexObs(variable='Prec', date=dates, obs=obs)
        >>>
        >>> # Fit model (requires fit_GWex_prec from precipitation module)
        >>> fit = fit_gwex_model(gwex_obs, list_option={'th': 0.5})
        >>> print(fit.n_stations)
        3
    """
    variable: str
    fit: Dict[str, Any]
    p: int
    version: str = field(default=GWEX_VERSION)

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.variable not in ('Prec', 'Temp'):
            raise ValueError("variable must be 'Prec' or 'Temp'")

        if not isinstance(self.fit, dict):
            raise ValueError("fit must be a dictionary")

        if self.p <= 0:
            raise ValueError("p (number of stations) must be positive")

        # Validate expected keys in fit dictionary
        expected_keys = {'listOption', 'listPar'}
        if not expected_keys.issubset(set(self.fit.keys())):
            raise ValueError(
                f"fit dictionary must contain keys {expected_keys}, "
                f"got {set(self.fit.keys())}"
            )

    @property
    def n_stations(self) -> int:
        """Get the number of stations."""
        return self.p

    def __repr__(self) -> str:
        """String representation."""
        opt = self.fit.get('listOption', {})
        return (
            f"GwexFit(variable='{self.variable}', n_stations={self.n_stations}, "
            f"options={list(opt.keys())})"
        )


@dataclass
class GwexSim:
    """
    Simulation results container.

    Holds the simulated time series generated from a fitted GWEX model.

    Attributes:
        variable: Type of variable ('Prec' or 'Temp')
        list_option: Dictionary of model options used for simulation
        date: numpy array of datetime64[D] with shape (nTime,)
        sim: numpy array of simulations with shape (nTime, nStations, nRep)
        version: GWEX package version

    Examples:
        >>> from core import GwexObs, fit_gwex_model, sim_gwex_model
        >>> import numpy as np
        >>>
        >>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
        >>> obs = np.random.rand(len(dates), 3) * 10
        >>> gwex_obs = GwexObs(variable='Prec', date=dates, obs=obs)
        >>>
        >>> fit = fit_gwex_model(gwex_obs, list_option={'th': 0.5})
        >>> sim = sim_gwex_model(
        ...     fit, nb_rep=2,
        ...     d_start=dates[0], d_end=dates[365]
        ... )
        >>> print(sim.n_stations, sim.n_rep)
        3 2
    """
    variable: str
    list_option: Dict[str, Any]
    date: np.ndarray
    sim: np.ndarray
    version: str = field(default=GWEX_VERSION)

    def __post_init__(self):
        """Validate inputs after initialization."""
        if self.variable not in ('Prec', 'Temp'):
            raise ValueError("variable must be 'Prec' or 'Temp'")

        # Validate date is 1D
        if self.date.ndim != 1:
            raise ValueError("date must be a 1-dimensional array")

        if not np.issubdtype(self.date.dtype, np.datetime64):
            raise ValueError("date must be an array of datetime64")

        # Validate sim is 3D
        if self.sim.ndim != 3:
            raise ValueError("sim must be a 3-dimensional array (nTime x nStations x nRep)")

        # Validate shapes match
        if self.sim.shape[0] != len(self.date):
            raise ValueError(
                f"sim first dimension ({self.sim.shape[0]}) must match "
                f"date length ({len(self.date)})"
            )

        # Ensure sim is float array
        self.sim = np.asarray(self.sim, dtype=np.float64)

        # Ensure date is datetime64[D]
        if self.date.dtype != np.dtype('datetime64[D]'):
            self.date = self.date.astype('datetime64[D]')

    @property
    def n_stations(self) -> int:
        """Get the number of stations."""
        return self.sim.shape[1]

    @property
    def n_rep(self) -> int:
        """Get the number of replications/scenarios."""
        return self.sim.shape[2]

    @property
    def n_times(self) -> int:
        """Get the number of time steps."""
        return self.sim.shape[0]

    def __repr__(self) -> str:
        """String representation."""
        date_min = self.date[0]
        date_max = self.date[-1]
        return (
            f"GwexSim(variable='{self.variable}', "
            f"period={date_min} -> {date_max}, "
            f"n_stations={self.n_stations}, n_rep={self.n_rep}, "
            f"n_times={self.n_times})"
        )


def fit_gwex_model(
    obs: GwexObs,
    par_margin: Optional[List[np.ndarray]] = None,
    list_option: Optional[Dict[str, Any]] = None,
    n_fit_workers: int = 1,
) -> GwexFit:
    """
    Fit a GWEX model to observations.

    Fits either a precipitation or temperature GWEX model depending on the
    variable type in the observations.

    Parameters:
        obs: GwexObs object containing observations
        par_margin: (optional, for Prec only) list of 12 arrays (one per month),
            each of shape (nStations x 3), containing pre-estimated parameters
            of marginal distributions (EGPD or Mixture of Exponentials)
        list_option: (optional) dictionary of fitting options. For precipitation:
            - 'th': threshold value in mm (default: 0.2)
            - 'nLag': order of Markov chain (default: 2)
            - 'typeMargin': 'EGPD' or 'mixExp' (default: 'mixExp')
            - 'copulaInt': 'Gaussian' or 'Student' (default: 'Gaussian')
            - 'isMAR': apply MAR(1) model (default: False)
            - 'is3Damount': apply 3D-amount model (default: False)
            - 'nChainFit': length of runs for fitting (default: 100000)
            - 'nCluster': number of clusters for parallel computation

            For temperature:
            - 'hasTrend': fit linear trend (default: False)
            - 'objGwexPrec': optional GwexObs with precipitation data
            - 'th': threshold for wet/dry classification (default: 0.2)
            - 'typeMargin': 'SGED' or 'Gaussian' (default: 'SGED')
            - 'depStation': 'MAR1' or 'Gaussian' (default: 'MAR1')

    Returns:
        GwexFit object with fitted parameters and options

    Raises:
        TypeError: if obs is not a GwexObs object
        ValueError: if variable type is unsupported

    Examples:
        >>> import numpy as np
        >>> from core import GwexObs, fit_gwex_model
        >>>
        >>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
        >>> obs_prec = np.random.rand(len(dates), 5) * 10
        >>> gwex_obs = GwexObs(variable='Prec', date=dates, obs=obs_prec)
        >>>
        >>> # Fit with custom threshold
        >>> fit = fit_gwex_model(gwex_obs, list_option={'th': 0.5})
        >>> print(f"Fitted {fit.n_stations} stations")
        Fitted 5 stations
    """
    # Validate input
    if not isinstance(obs, GwexObs):
        raise TypeError("obs must be a GwexObs object")

    type_var = obs.variable
    p = obs.n_stations

    print("Fit generator")

    # Import fitting functions from sibling modules
    if type_var == 'Prec':
        from .precipitation import fit_GWex_prec
        fit_result = fit_GWex_prec(obs, par_margin, list_option,
                                   n_fit_workers=n_fit_workers)
    elif type_var == 'Temp':
        raise NotImplementedError(
            "Temperature model is not yet implemented in gwex_py. "
            "Only precipitation ('Prec') is currently supported."
        )
    else:
        raise ValueError(f"Unsupported variable type: {type_var}")

    # Create and return GwexFit object
    return GwexFit(variable=type_var, fit=fit_result, p=p)


def sim_gwex_model(
    fit: GwexFit,
    nb_rep: int = 10,
    d_start: Optional[np.datetime64] = None,
    d_end: Optional[np.datetime64] = None,
    obs: Optional[GwexObs] = None,
    prob_class: Optional[np.ndarray] = None,
    sim_prec: Optional['GwexSim'] = None,
    use_seed: bool = False,
) -> GwexSim:
    """
    Simulate from a fitted GWEX model.

    Generates stochastic scenarios from a fitted GWEX model. For temperature
    models conditional on precipitation, provide sim_prec (or set up through
    list_option in fit).

    Parameters:
        fit: GwexFit object with fitted model parameters
        nb_rep: number of simulation replications/scenarios (default: 10)
        d_start: starting date for simulation (datetime64)
        d_end: ending date for simulation (datetime64)
        obs: optional GwexObs with observations (for disaggregation or
            needed for conditional models)
        prob_class: vector of probability classes for intensity classification
            (default: [0.5, 0.75, 0.9, 0.99])
        sim_prec: optional GwexSim with precipitation simulations (for
            temperature conditional on precipitation)
        use_seed: if True, control random seed per replication (default: False)

    Returns:
        GwexSim object containing simulated time series

    Raises:
        TypeError: if fit is not a GwexFit object
        ValueError: if required arguments are missing

    Examples:
        >>> import numpy as np
        >>> from core import GwexObs, fit_gwex_model, sim_gwex_model
        >>>
        >>> dates = np.arange('2005-01-01', '2015-01-01', dtype='datetime64[D]')
        >>> obs = np.random.rand(len(dates), 3) * 10
        >>> gwex_obs = GwexObs(variable='Prec', date=dates, obs=obs)
        >>>
        >>> fit = fit_gwex_model(gwex_obs, list_option={'th': 0.5})
        >>>
        >>> # Simulate 2 scenarios for the first year
        >>> sim = sim_gwex_model(
        ...     fit, nb_rep=2,
        ...     d_start=dates[0], d_end=dates[365]
        ... )
        >>> print(f"Simulated shape: {sim.sim.shape}")
        Simulated shape: (365, 3, 2)
    """
    # Validate input
    if not isinstance(fit, GwexFit):
        raise TypeError("fit must be a GwexFit object")

    type_var = fit.variable
    p = fit.n_stations

    # Resolve d_start to a definite datetime64[D] (avoids Pylance Optional narrowing issues)
    if d_start is None:
        d_start_dt: np.datetime64 = np.datetime64('1900-01-01', 'D')
    elif isinstance(d_start, np.datetime64):
        d_start_dt = d_start.astype('datetime64[D]')
    else:
        d_start_dt = np.datetime64(str(d_start), 'D')

    if d_end is None:
        d_end_dt: np.datetime64 = np.datetime64('1999-12-31', 'D')
    elif isinstance(d_end, np.datetime64):
        d_end_dt = d_end.astype('datetime64[D]')
    else:
        d_end_dt = np.datetime64(str(d_end), 'D')

    # Create date vector
    vec_dates = np.arange(d_start_dt, d_end_dt + np.timedelta64(1, 'D'), dtype='datetime64[D]')

    # Set default probability classes
    if prob_class is None:
        prob_class = np.array([0.5, 0.75, 0.9, 0.99])

    # Handle conditional temperature models
    if type_var == 'Temp':
        list_option = fit.fit['listOption']
        cond_prec = list_option.get('condPrec', False)

        if cond_prec:
            if sim_prec is None or not isinstance(sim_prec, GwexSim):
                raise ValueError(
                    "For temperature models conditional on precipitation, "
                    "sim_prec (GwexSim object with precipitation simulations) is required"
                )
            vec_dates = sim_prec.date
            sim_prec_data = sim_prec.sim
            nb_rep = sim_prec.n_rep
        else:
            sim_prec_data = None
    else:
        sim_prec_data = None

    n = len(vec_dates)

    # Initialize output array
    print("Generate scenarios")
    sim_out = np.zeros((n, p, nb_rep), dtype=np.float64)

    # Sequential loop through replications (parallel version deferred)
    for i_sim in range(nb_rep):
        # Set seed if requested
        if use_seed:
            seed = i_sim
        else:
            seed = None

        # Call simulation function based on variable type
        if type_var == 'Prec':
            from .precipitation import sim_GWex_prec_1it
            # precipitation.py expects raw dicts, not dataclasses:
            #   objGwexFit -> fit.fit (the inner dict with 'listOption' and 'listPar')
            #   objGwexObs -> {'obs': ..., 'date': ...} or None
            #   myseed     -> int (None becomes 0 = unseeded equivalent)
            obs_dict = {'obs': obs.obs, 'date': obs.date} if obs is not None else None
            sim_out[:, :, i_sim] = sim_GWex_prec_1it(
                fit.fit,
                vec_dates,
                myseed=seed if seed is not None else 0,
                objGwexObs=obs_dict,
                prob_class=prob_class,
            )
        elif type_var == 'Temp':
            raise NotImplementedError(
                "Temperature simulation is not yet implemented in gwex_py. "
                "Only precipitation ('Prec') is currently supported."
            )

    # Create and return GwexSim object
    return GwexSim(
        variable=type_var,
        list_option=fit.fit['listOption'],
        date=vec_dates,
        sim=sim_out
    )
