"""
Codes for estimating hydrometeor-related parameters: hydrometeor classification
(HID), rainfall rate, drop-size distribution (NW, D0) and snowfall rate.

Ported from the oceanpol_kit reference. All functions operate on plain numpy
arrays (NaN-filled, not masked); the production line wraps the results into
Py-ART field dictionaries.

@title: hydrometeors
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    compute_hid
    get_dsd_estimate
    get_rainfall_estimate
    get_snowfall_estimate
"""
from typing import Tuple

import numpy as np

from csu_radartools.csu_fhc import csu_fhc_summer


def compute_hid(
    dbz: np.ndarray,
    zdr: np.ndarray,
    kdp: np.ndarray,
    rhohv: np.ndarray,
    temperature: np.ndarray,
) -> np.ndarray:
    """
    Hydrometeor identification using the CSU summer fuzzy-logic scheme.

    Categories: 1 Drizzle; 2 Rain; 3 Ice Crystals; 4 Aggregates; 5 Wet Snow;
    6 Vertical Ice; 7 LD Graupel; 8 HD Graupel; 9 Hail; 10 Big Drops.

    Parameters
    ----------
    dbz, zdr, kdp, rhohv, temperature : np.ndarray
        2D radar fields (NaN-filled). Temperature in degC.

    Returns
    -------
    np.ndarray
        2D classification (int-valued, 0 = no data).
    """
    hid = csu_fhc_summer(
        use_temp=True,
        method="hybrid",
        dz=dbz,
        zdr=zdr,
        ldr=None,
        kdp=kdp,
        rho=rhohv,
        T=temperature,
        verbose=False,
        plot_flag=False,
        n_types=10,
        temp_factor=1,
        band="C",
    )
    hid[np.isnan(dbz)] = 0
    return hid


def get_dsd_estimate(dbz: np.ndarray, zdr: np.ndarray, temperature: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the drop-size distribution parameters NW (log10 normalised
    intercept) and D0 (median volume diameter, mm) from ZH and ZDR.

    Only valid for liquid drops (temperature >= 0); set to NaN below freezing.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (nw, d0).
    """
    d0 = np.zeros_like(zdr)
    pos = (-0.5 <= zdr) & (zdr < 1.25)
    tmp = 0.0203 * zdr**4 - 0.1488 * zdr**3 + 0.2209 * zdr**2 + 0.5571 * zdr + 0.801
    d0[pos] = tmp[pos]

    pos = (1.25 <= zdr) & (zdr < 5)
    tmp = 0.0355 * zdr**3 - 0.3021 * zdr**2 + 1.0556 * zdr + 0.6844
    d0[pos] = tmp[pos]

    nw = 10 ** (dbz / 10) / (0.056 * d0**7.319)
    nw = np.log10(nw)

    # Only valid for positive temperatures (liquid drops)
    nw[temperature < 0] = np.nan
    d0[temperature < 0] = np.nan

    return nw, d0


def get_rainfall_estimate(
    zh: np.ndarray, zdr: np.ndarray, kdp: np.ndarray, temperature: np.ndarray, southern_ocean: bool
) -> np.ndarray:
    """
    Estimate rainfall rate (mm h-1) from ZH, ZDR, KDP and temperature, using
    regime-dependent coefficients (Southern-Ocean tuned set when applicable).

    Parameters
    ----------
    zh, zdr, kdp, temperature : np.ndarray
        2D radar fields (NaN-filled). Temperature in degC.
    southern_ocean : bool
        Use the Southern-Ocean coefficient set when True.

    Returns
    -------
    np.ndarray
        2D rainfall rate.
    """
    sigma_dr = 10 ** (0.1 * zdr)
    eta_h = 10 ** (0.1 * zh)
    rainfall = np.zeros_like(zh) + np.nan

    pos = (kdp <= 0.3) & (zdr <= 0.25)
    if southern_ocean:
        tmp = 0.021 * eta_h**0.72
    else:
        tmp = 0.016 * eta_h**0.846
    rainfall[pos] = tmp[pos]

    pos = (kdp <= 0.3) & (zdr > 0.25)
    if southern_ocean:
        tmp = 0.0086 * eta_h**0.91 * sigma_dr ** (-4.21)
    else:
        tmp = 0.011 * eta_h**0.825 * sigma_dr ** (-3.055)
    rainfall[pos] = tmp[pos]

    pos = (kdp > 0.3) & (zdr <= 0.25)
    if southern_ocean:
        tmp = 30.62 * kdp**0.78
    else:
        tmp = 16.171 * kdp**0.742
    rainfall[pos] = tmp[pos]

    pos = (kdp > 0.3) & (zdr > 0.25)
    if southern_ocean:
        tmp = 45.70 * kdp ** (0.88) * sigma_dr ** (-1.67)
    else:
        tmp = 24.199 * kdp ** (0.827) * sigma_dr ** (-0.488)
    rainfall[pos] = tmp[pos]

    rainfall[np.isnan(rainfall) | (temperature <= -10)] = 0
    return rainfall


def get_snowfall_estimate(dbz_clean: np.ndarray, kdp: np.ndarray, temps: np.ndarray) -> np.ndarray:
    """
    Estimate snowfall rate from reflectivity and KDP; valid below freezing only
    (set to 0 where temperature > 0).

    Parameters
    ----------
    dbz_clean, kdp, temps : np.ndarray
        2D radar fields. Temperature in degC.

    Returns
    -------
    np.ndarray
        2D snowfall rate.
    """
    snow = 1.48 * kdp**0.615 * (10 ** (dbz_clean / 10)) ** 0.33
    snow[np.isnan(snow) | (temps > 0)] = 0
    return snow
