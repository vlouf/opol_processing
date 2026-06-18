"""
Temperature profiles for OceanPOL radar processing (Py-ART edition).

Retrieves the vertical temperature profile from ERA5/ERA5T reanalysis and, when
reanalysis is unavailable, derives a standard-atmosphere profile anchored on the
radar-detected bright band (melting layer). Profiles are returned as
(height [m, ASL], temperature [K]) and interpolated onto the radar gate
altitudes (``radar.gate_altitude``).

Ported from the oceanpol_kit reference (commit-tracked); adapted from a pyodim
sweep list to a single Py-ART ``Radar`` object.

@title: temperature
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    find_era5_temperature_file
    read_era5_temperature
    standard_temperature_profile
    detect_brightband_height
    get_volume_temperature_profile
    interp_temperature
"""
import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from scipy import interpolate


def find_era5_temperature_file(date: pd.Timestamp) -> Optional[str]:
    """
    Locate the ERA5 temperature file for a date on NCI's rt52 archive.

    Final ERA5 lives under ``era5/`` as monthly files
    (``t_era5_oper_pl_YYYYMM...``). Recent dates that have not yet been
    finalised are only available as the preliminary ERA5T product under
    ``era5t/`` as daily files (``t_era5t_oper_pl_YYYYMMDD-YYYYMMDD.nc``).
    Final data is preferred; ERA5T is the fallback for recent dates.

    Parameters
    ----------
    date : pd.Timestamp
        Date for extraction.

    Returns
    -------
    Optional[str]
        Path to the first matching file, or None if none is found.
    """
    y, m, d = date.year, date.month, date.day
    patterns = [
        f"/g/data/rt52/era5/pressure-levels/reanalysis/t/{y}/t_era5_oper_pl_{y}{m:02}*.nc",
        f"/g/data/rt52/era5t/pressure-levels/reanalysis/t/{y}/t_era5t_oper_pl_{y}{m:02}{d:02}*.nc",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_era5_temperature(date: pd.Timestamp, longitude: float, latitude: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the temperature profile from ERA5 data for a given date, longitude,
    and latitude.

    Parameters
    ----------
    date : pd.Timestamp
        Date for extraction.
    longitude : float
        Radar longitude.
    latitude : float
        Radar latitude.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - z: A 1D array of heights in meters (ascending).
        - temperature: A 1D array of temperatures in Kelvin.
    """
    era5_file = find_era5_temperature_file(date)
    if era5_file is None:
        raise FileNotFoundError(
            f"No ERA5/ERA5T temperature file found for {date:%Y-%m-%d} under /g/data/rt52."
        )

    # Get temperature. Use a context manager so the file handle is released —
    # this runs inside a long-lived multiprocessing batch where leaks add up.
    with xr.open_dataset(era5_file) as dset:
        nset = dset.sel(longitude=longitude, latitude=latitude, time=date, method="nearest")
        temp_profile = nset.t.values
        level = nset.level.values
    geo_h_profile = -2494.3 / 0.218 * np.log(level / 1013.15)
    zmin = geo_h_profile.min()
    temp_ground = temp_profile[np.argmin(geo_h_profile)] + 0.0065 * zmin

    # Extrapolate sea level temperature.
    geo_h_profile = np.append(geo_h_profile, 0)
    temp_profile = np.append(temp_profile, temp_ground)

    # interp1d requires monotonically increasing x.
    order = np.argsort(geo_h_profile)
    return geo_h_profile[order], temp_profile[order]


def standard_temperature_profile(
    z0c: float, ztop: float = 20000.0, lapse: float = 0.0065
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a temperature profile from a freezing-level height assuming a constant
    lapse rate (standard-atmosphere style).

    Parameters
    ----------
    z0c : float
        Height of the 0 degC level in metres.
    ztop : float, optional
        Top of the profile in metres (default 20000).
    lapse : float, optional
        Lapse rate in K m-1 (default 0.0065).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (heights_m ascending, temperature_K), matching read_era5_temperature.
    """
    z = np.arange(0.0, ztop + 1.0, 100.0)
    temp_c = -lapse * (z - z0c)
    return z, temp_c + 273.15


def _as_nan(data: np.ndarray) -> np.ndarray:
    """Return a float64 ndarray with masked entries replaced by NaN."""
    return np.ma.filled(np.ma.asarray(data, dtype="float64"), np.nan)


def detect_brightband_height(
    radar,
    fields: dict,
    zmin: float = 500.0,
    zmax: float = 6000.0,
    min_points: int = 50,
) -> Optional[float]:
    """
    Estimate the freezing-level (0 degC) height from the radar bright band.

    In stratiform precipitation the melting layer produces a depression in
    RHOHV together with enhanced ZDR and a reflectivity peak. This routine
    pools all gates with that signature across the volume and takes the
    RHOHV-weighted median height of the depression as the melting-layer level;
    the 0 degC level is taken a few hundred metres above it.

    Parameters
    ----------
    radar : pyart.core.Radar
        The radar volume.
    fields : dict
        Canonical->actual field name mapping (see radar_codes.resolve_fields).
    zmin, zmax : float, optional
        Height window (m) to search for the melting layer.
    min_points : int, optional
        Minimum number of qualifying gates required to trust the estimate.

    Returns
    -------
    Optional[float]
        Estimated 0 degC height in metres, or None if no reliable bright band
        is found (e.g. convective-only or clear scans).
    """
    z = _as_nan(radar.gate_altitude["data"])
    zh = _as_nan(radar.fields[fields["DBZH"]]["data"])
    rho = _as_nan(radar.fields[fields["RHOHV"]]["data"])
    zdr = _as_nan(radar.fields[fields["ZDR"]]["data"])

    good = (
        np.isfinite(z)
        & np.isfinite(zh)
        & np.isfinite(rho)
        & np.isfinite(zdr)
        & (z >= zmin)
        & (z <= zmax)
        & (zh >= 20.0)
        & (zh <= 45.0)
        & (rho >= 0.85)
        & (rho <= 0.97)
        & (zdr >= 0.5)
    )

    if not good.any():
        return None

    heights = z[good]
    weights = 0.97 - rho[good]
    if heights.size < min_points:
        return None

    order = np.argsort(heights)
    h = heights[order]
    cw = np.cumsum(weights[order])
    ml_height = float(h[np.searchsorted(cw, 0.5 * cw[-1])])

    return ml_height + 300.0


def get_volume_temperature_profile(
    date: pd.Timestamp,
    lat: float,
    lon: float,
    radar,
    fields: dict,
    default_z0c: float = 2500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a (heights_m, temperature_K) profile for the whole volume.

    Tries, in order: (1) ERA5/ERA5T reanalysis; (2) a standard-lapse profile
    anchored on the radar-detected bright band; (3) a standard-lapse profile
    anchored on a default freezing level. Never raises for missing temperature
    data, so processing can continue.

    Parameters
    ----------
    date : pd.Timestamp
        Volume date/time.
    lat, lon : float
        Radar latitude / longitude.
    radar : pyart.core.Radar
        The radar volume.
    fields : dict
        Canonical->actual field name mapping (see radar_codes.resolve_fields).
    default_z0c : float, optional
        Freezing-level height (m) used only if both ERA5 and the bright band
        are unavailable (default 2500).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (heights_m ascending, temperature_K).
    """
    try:
        return read_era5_temperature(date, lon, lat)
    except Exception as err:  # noqa: BLE001 - resilience is intentional
        print(f"WARNING: ERA5 temperature unavailable ({type(err).__name__}: {err}).")

    z0c = detect_brightband_height(radar, fields)
    if z0c is not None:
        print(f"Using radar bright-band freezing level ~{z0c:.0f} m for temperature.")
        return standard_temperature_profile(z0c)

    print(
        f"WARNING: no bright band detectable; using default freezing level "
        f"{default_z0c:.0f} m for temperature."
    )
    return standard_temperature_profile(default_z0c)


def interp_temperature(
    geo_h_profile: np.ndarray, temp_profile: np.ndarray, gate_altitude: np.ndarray
) -> np.ndarray:
    """
    Interpolate a (heights_m, temperature_K) profile onto gate altitudes,
    returning degC (masked outside the profile range).

    Parameters
    ----------
    geo_h_profile : np.ndarray
        1D ascending heights (m, ASL).
    temp_profile : np.ndarray
        1D temperatures (K) matching ``geo_h_profile``.
    gate_altitude : np.ndarray
        2D radar gate altitudes (m, ASL), e.g. ``radar.gate_altitude['data']``.

    Returns
    -------
    np.ndarray
        Masked 2D array of temperature in degC (masked outside the profile).
    """
    f_interp = interpolate.interp1d(
        geo_h_profile, temp_profile, bounds_error=False, fill_value=9999
    )
    return np.ma.masked_equal(f_interp(gate_altitude), 9999) - 273.15
