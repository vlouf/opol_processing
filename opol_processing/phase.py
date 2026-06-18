"""
Differential-phase processing: corrected PHIDP and specific differential phase
(KDP) via phido's ``kdp_pyart`` (Py-ART radar interface).

The non-stationary complex retrieval is used, gated by a precipitation
GateFilter so KDP is only integrated over genuine precipitation (this prevents
the radial PHIDP streaks that arise from integrating phase noise over clear
air). The legacy Bringi (csu_kdp) and Giangrande (LP solver) variants have been
removed.

@title: phase
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    get_phidp
"""
import numpy as np

import phido


# Non-stationary complex KDP retrieval configuration (phido.kdp_pyart).
NONSTATIONARY_KWARGS = dict(
    lmax=[250, 1000],   # [range, azimuth]
    nsteps=11,
    nit=40,
    eps=0.0,
    atol=1e-10,
    limit_range=True,
)


def get_phidp(radar, phidp_field, gatefilter, refl, temperature):
    """
    Calculate the corrected differential phase (PHIDP) and specific differential
    phase (KDP) with phido's non-stationary complex retrieval.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.
    phidp_field : str
        Name of the (raw) differential-phase field in ``radar.fields``.
    gatefilter : pyart.filters.GateFilter
        Precipitation gate filter; the retrieval is restricted to non-excluded
        (precipitation) gates.
    refl : np.ndarray
        2D reflectivity (dBZ), used to suppress spurious negative KDP.
    temperature : np.ndarray
        2D temperature (degC).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (phidp_corr, kdp) as NaN-filled float64 arrays (NaN outside the
        precipitation gate filter).
    """
    kdp_meta, phidp_meta = phido.kdp_pyart(
        radar,
        phidp_field,
        gatefilter=gatefilter,
        stationary=False,
        complex=True,
        **NONSTATIONARY_KWARGS,
    )

    phidp_corr = np.ma.filled(phidp_meta["data"], np.nan).astype("float64")
    kdp = np.ma.filled(kdp_meta["data"], np.nan).astype("float64")

    # Suppress spurious negative KDP in warm, light rain (NaN comparisons are
    # False, so masked/absent gates are left untouched).
    refl_a = np.ma.filled(np.ma.asarray(refl, dtype="float64"), np.nan)
    temp_a = np.ma.filled(np.ma.asarray(temperature, dtype="float64"), np.nan)
    cleanup = (kdp < 0) & (refl_a < 40) & (temp_a >= 0)
    kdp[cleanup] = 0.0

    return phidp_corr, kdp
