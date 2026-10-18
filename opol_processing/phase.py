"""
Differential-phase processing: corrected PHIDP and specific differential phase
(KDP).

Default method (``"lrose"``): the endpoint-constrained local self-consistency
KDP algorithm for LROSE (see :mod:`kdp_lrose`):

    Dixon, M., and J. Hubbert, 2026: A robust, general-purpose KDP algorithm
    for LROSE. Proc. 13th European Conference on Radar in Meteorology and
    Hydrology (ERAD 2026), Belgrade, Serbia.

Optional method (``"phido"``, not activated by default): phido's
non-stationary complex retrieval (``phido.kdp_pyart``,
https://github.com/jordanbrook/phido). phido is only imported when this method
is requested, so it is not a required dependency.

Both methods are gated by the precipitation GateFilter, so phase is only
processed over genuine precipitation (this prevents the radial PHIDP streaks
that arise from integrating phase noise over clear air). The corrected PHIDP is
returned with the system differential phase removed (starts from zero), as the
downstream Z-PHI and ZDR attenuation corrections require.

@title: phase
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    get_phidp
    field_metadata
"""
import numpy as np

from . import kdp_lrose


DEFAULT_METHOD = "lrose"
METHODS = ("lrose", "phido")

# Tunables for the LROSE retrieval (defaults: kdp_lrose.KdpParams).
LROSE_PARAMS = kdp_lrose.KdpParams()

# Non-stationary complex KDP retrieval configuration (phido.kdp_pyart).
NONSTATIONARY_KWARGS = dict(
    lmax=[250, 1000],   # [range, azimuth]
    nsteps=11,
    nit=40,
    eps=0.0,
    atol=1e-10,
    limit_range=True,
)

_REFERENCES = {
    "lrose": (
        "Dixon, M., and J. Hubbert, 2026: A robust, general-purpose KDP algorithm for LROSE. "
        "Proc. 13th European Conference on Radar in Meteorology and Hydrology (ERAD 2026), "
        "Belgrade, Serbia."
    ),
    "phido": "phido non-stationary complex KDP retrieval, https://github.com/jordanbrook/phido",
}

_COMMENTS = {
    "lrose": {
        "phidp": (
            "LROSE KdpFilt PHIDP processing (unfolding, FIR filtering) with the endpoint-constrained "
            "local self-consistency backscatter-phase removal of Dixon and Hubbert (2026); "
            "system differential phase removed; precipitation gates only."
        ),
        "kdp": (
            "Endpoint-constrained local self-consistency KDP (Dixon and Hubbert 2026), falling back to "
            "the LROSE adaptive-length slope KDP where no backscatter phase is detected; "
            "precipitation gates only."
        ),
    },
    "phido": {
        "phidp": "phido non-stationary complex retrieval; precipitation gates only.",
        "kdp": "phido non-stationary complex retrieval; precipitation gates only.",
    },
}


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"Unknown phase method {method!r}; expected one of {METHODS}.")


def field_metadata(method=DEFAULT_METHOD):
    """
    ``comment`` and ``references`` attributes for the corrected PHIDP and KDP
    output fields.

    Returns
    -------
    Tuple[dict, dict]
        (phidp_attrs, kdp_attrs).
    """
    _check_method(method)
    ref = _REFERENCES[method]
    return (
        {"comment": _COMMENTS[method]["phidp"], "references": ref},
        {"comment": _COMMENTS[method]["kdp"], "references": ref},
    )


def _phido(radar, phidp_field, gatefilter):
    try:
        import phido
    except ImportError as err:
        raise ImportError(
            "The 'phido' phase method needs phido: pip install git+https://github.com/jordanbrook/phido.git"
        ) from err

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
    return phidp_corr, kdp


def get_phidp(radar, phidp_field, gatefilter, refl, temperature, zdr=None, rhohv=None, snr=None,
              method=DEFAULT_METHOD):
    """
    Calculate the corrected differential phase (PHIDP) and specific differential
    phase (KDP).

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
        2D calibrated reflectivity (dBZ). Used by the LROSE self-consistency
        relation, and to suppress spurious negative KDP.
    temperature : np.ndarray
        2D temperature (degC).
    zdr, rhohv, snr : np.ndarray
        2D calibrated ZDR (dB), RHOHV and SNR (dB). Required by ``"lrose"``,
        ignored by ``"phido"``.
    method : {"lrose", "phido"}
        Phase-processing method. Default ``"lrose"`` (Dixon and Hubbert 2026).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (phidp_corr, kdp) as NaN-filled float64 arrays (NaN outside the
        precipitation gate filter).
    """
    _check_method(method)

    if method == "lrose":
        if zdr is None or rhohv is None or snr is None:
            raise ValueError("The 'lrose' phase method needs zdr, rhohv and snr.")
        phidp_corr, kdp, _ = kdp_lrose.retrieve_phidp_kdp(
            radar,
            radar.fields[phidp_field]["data"],
            refl,
            zdr,
            rhohv,
            snr,
            gatefilter=gatefilter,
            params=LROSE_PARAMS,
        )
    else:
        phidp_corr, kdp = _phido(radar, phidp_field, gatefilter)

    # Suppress spurious negative KDP in warm, light rain (NaN comparisons are
    # False, so masked/absent gates are left untouched).
    refl_a = np.ma.filled(np.ma.asarray(refl, dtype="float64"), np.nan)
    temp_a = np.ma.filled(np.ma.asarray(temperature, dtype="float64"), np.nan)
    cleanup = (kdp < 0) & (refl_a < 40) & (temp_a >= 0)
    kdp[cleanup] = 0.0

    return phidp_corr, kdp
