"""
load_ideam_data.py  [DEPRECATED]
---------------------------------
This module is deprecated.  Use the integration layer instead:

    from src.gwex_integration.data_bridge import build_gwex_obs
    obs = build_gwex_obs("A_valley_floor")   # returns GwexObs directly

The replacement supports:
  - Automatic CSV discovery (no hardcoded file lists)
  - Region-based and ad-hoc station selection
  - QC-screened data via src/rfa/data_loading
  - Extensibility for new gauges without code changes

This file is retained only for backward compatibility and will be
removed in a future version.

Original description:
Loads IDEAM gauge CSVs from the Aburrá Valley project and assembles them
into the T×S numpy array + date vector needed by gwex_py.

Two configurations are provided:
  - OPTION_1: 4 long-running stations, 1971–2025 (~54 years)
  - OPTION_2: 5 active stations, 1991–2025 (~35 years)  ← recommended start

Usage (deprecated):
    from load_ideam_data import load_gwex_inputs
    obs_array, dates, station_names = load_gwex_inputs(option=2)

    from gwex_py import GwexObs
    gwex_obs = GwexObs(obs=obs_array, date=dates, type_var="Prec")
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "aburra-valley-hydrology" / "data" / "processed" / "gauges" / "ideam"

# ---------------------------------------------------------------------------
# Station configurations
# ---------------------------------------------------------------------------

# Option 1: 4 long-running stations, common window 1971-01-01 – 2025-08-04
#   Best for extreme-tail estimation (longest record)
OPTION_1_STATIONS = [
    "27010800_la_meseta_san_pedro_19710101_20251231_all.csv",
    "27010810_santa_elena_19710101_20250804_all.csv",
    "27015090_tulio_ospina_19610101_20251231_all.csv",
    "27015330_aeropuerto_olaya_herrera_19610101_20250823_all.csv",
]
OPTION_1_START = "1971-01-01"
OPTION_1_END   = "2025-08-04"   # limited by santa_elena

# Option 2: 5 active stations, common window 1991-01-01 – 2025-08-04
#   Best for spatial coverage (includes northern Astilleros gauge)
OPTION_2_STATIONS = [
    "27010800_la_meseta_san_pedro_19710101_20251231_all.csv",
    "27010810_santa_elena_19710101_20250804_all.csv",
    "27011110_astilleros_19910101_20251231_all.csv",
    "27015090_tulio_ospina_19610101_20251231_all.csv",
    "27015330_aeropuerto_olaya_herrera_19610101_20250823_all.csv",
]
OPTION_2_START = "1991-01-01"
OPTION_2_END   = "2025-08-04"   # limited by santa_elena

# ---------------------------------------------------------------------------
# Historic-only stations (useful for L-moments RFA but NOT for GWEX —
# no overlap with the active network)
# ---------------------------------------------------------------------------
HISTORIC_STATIONS = {
    "chorrillos":       "27010350_chorrillos_19610101_19871231_all.csv",
    "villahermosa":     "27010450_villahermosa_plant_19610101_19871231_all.csv",
    "ayura":            "27010930_ayura_19610101_19851231_all.csv",
}


def _load_station(csv_path: Path, full_date_index: pd.DatetimeIndex) -> pd.Series:
    """
    Load one station CSV, reindex to the full daily date range, and fill
    absent rows with NaN.  Rows that exist but have value 0.0 are genuine
    dry days and are kept as-is.
    """
    df = pd.read_csv(csv_path, parse_dates=["Fecha"], index_col="Fecha")
    series = df["precipitation_mm"].reindex(full_date_index)  # NaN for absent rows
    return series


def load_gwex_inputs(
    option: int = 2,
    th: float = 0.2,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Assemble T×S precipitation array and aligned date vector for gwex_py.

    Parameters
    ----------
    option : int
        1 → 4-station config (1971–2025, ~54 years)
        2 → 5-station config (1991–2025, ~35 years)  [default]
    th : float
        Wet-day threshold in mm (used for reporting only; gwex_py applies
        its own threshold internally from options["th"]).
    verbose : bool
        Print data summary when True.

    Returns
    -------
    obs_array : np.ndarray, shape (T, S)
        Daily precipitation in mm.  NaN where data is absent.
    dates : np.ndarray of datetime64[D], shape (T,)
        One date per row of obs_array.
    station_names : list of str
        Human-readable station labels, one per column of obs_array.
    """
    if option == 1:
        files  = OPTION_1_STATIONS
        start  = OPTION_1_START
        end    = OPTION_1_END
    elif option == 2:
        files  = OPTION_2_STATIONS
        start  = OPTION_2_START
        end    = OPTION_2_END
    else:
        raise ValueError("option must be 1 or 2")

    full_index = pd.date_range(start=start, end=end, freq="D")

    series_list    = []
    station_names  = []

    for fname in files:
        path = DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Cannot find {path}. Check DATA_DIR.")
        s = _load_station(path, full_index)
        series_list.append(s)
        # extract a clean label from the filename
        parts = fname.split("_")
        label = "_".join(parts[1:-3]) if len(parts) > 4 else parts[0]
        station_names.append(label)

    df_aligned = pd.concat(series_list, axis=1)
    df_aligned.columns = station_names

    obs_array = df_aligned.values.astype(float)   # shape (T, S)
    dates     = full_index.values.astype("datetime64[D]")

    if verbose:
        T, S = obs_array.shape
        print(f"\nGWEX input summary (Option {option})")
        print(f"  Period       : {start} → {end}")
        print(f"  Shape        : {T} days × {S} stations")
        print(f"  Years        : {T/365.25:.1f}")
        print()
        print(f"  {'Station':<30} {'NaN days':>9} {'NaN %':>7} {'Wet days':>9} {'Max mm':>8}")
        print("  " + "-"*68)
        for i, col in enumerate(station_names):
            col_data = obs_array[:, i]
            n_nan  = int(np.isnan(col_data).sum())
            n_wet  = int((col_data >= th).sum())
            maxval = float(np.nanmax(col_data))
            print(f"  {col:<30} {n_nan:>9} {100*n_nan/T:>6.1f}% {n_wet:>9} {maxval:>8.1f}")

        total_nan = np.isnan(obs_array).sum()
        total_cells = obs_array.size
        print(f"\n  Overall NaN  : {total_nan} / {total_cells} cells ({100*total_nan/total_cells:.1f}%)")
        print(f"\n  ✓ Ready for GwexObs(obs=obs_array, date=dates, type_var='Prec')")

    return obs_array, dates, station_names


def load_historic_station(name: str) -> pd.Series:
    """
    Load one of the historic-only stations as a plain pandas Series (daily).
    Useful for L-moments RFA or record-period frequency analysis.

    Parameters
    ----------
    name : str
        One of: 'chorrillos', 'villahermosa', 'ayura'
    """
    if name not in HISTORIC_STATIONS:
        raise ValueError(f"name must be one of {list(HISTORIC_STATIONS)}")
    path = DATA_DIR / HISTORIC_STATIONS[name]
    df   = pd.read_csv(path, parse_dates=["Fecha"], index_col="Fecha")
    # reindex to full contiguous daily range (fill gaps with NaN)
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    return df["precipitation_mm"].reindex(full_index)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    obs, dates, names = load_gwex_inputs(option=2, verbose=True)

    print("\n--- Building GwexObs ---")
    try:
        from gwex_py import GwexObs
        gwex_obs = GwexObs(obs=obs, date=dates, type_var="Prec")
        print(f"  GwexObs created: {gwex_obs.obs.shape[0]} days × {gwex_obs.obs.shape[1]} stations")
    except ImportError:
        print("  (gwex_py not on sys.path — run from the gwex_py parent directory)")

    print("\n--- Historic stations ---")
    for name in HISTORIC_STATIONS:
        s = load_historic_station(name)
        print(f"  {name}: {len(s)} days, {s.notna().sum()} non-NaN, max={s.max():.1f} mm")
