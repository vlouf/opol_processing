"""
Gate filters for OceanPOL processing.

Three role-separated gate filters are built, each reproducing an oceanpol_kit
mask using Py-ART's GateFilter idiom (they are intentionally NOT merged: the
masks serve opposite purposes and merging reintroduces clear-air velocity loss
and PHIDP streaks):

- ``do_gatefilter_opol``    : reflectivity cleaning (hydrometeor mask).
- ``do_precip_gatefilter``  : strict precipitation gate for phase processing.
- ``do_velocity_gatefilter``: permissive coherence gate for velocity dealiasing.

@title: filtering
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    range_texture
    do_gatefilter_opol
    do_precip_gatefilter
    do_velocity_gatefilter
"""
import pyart
import numpy as np
import pandas as pd


def range_texture(data: np.ndarray, winlen: int = 10) -> np.ndarray:
    """
    Windowed standard deviation along range (texture), per ray.

    Vectorised replacement for the oceanpol_kit numba ``area_std``; uses a
    centred rolling standard deviation. Masked/NaN entries are treated as NaN
    and the resulting texture is NaN-filled with 0.

    Parameters
    ----------
    data : np.ndarray
        2D array (nrays, ngates).
    winlen : int, optional
        Range window length (default 10).

    Returns
    -------
    np.ndarray
        2D texture array (nrays, ngates), float64.
    """
    arr = np.ma.filled(np.ma.asarray(data, dtype="float64"), np.nan)
    # rolling along range (axis 1): operate on the transpose then transpose back.
    tex = pd.DataFrame(arr.T).rolling(window=winlen, min_periods=1, center=True).std().to_numpy().T
    return np.nan_to_num(tex, nan=0.0)


def do_gatefilter_opol(
    radar,
    refl_name: str = "total_power",
    rhohv_name: str = "cross_correlation_ratio",
    phidp_name: str = "PHIDP",
    despeckle_size: int = 5,
) -> "pyart.filters.GateFilter":
    """
    Reflectivity-cleaning gate filter, reproducing oceanpol_kit's
    ``get_hydrometeor_mask``:

        exclude where (PHIDP texture > 60) OR (RHOHV < 0.5) OR (refl < -20),
        but rescue gates where RHOHV > 0.90; then despeckle.

    A temporary ``PHIDP_TEXTURE`` field is added and removed.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.
    refl_name : str
        Reflectivity field name (cleaning is based on total power).
    rhohv_name : str
        (Corrected) cross-correlation ratio field name.
    phidp_name : str
        Raw differential phase field name (for the texture estimate).
    despeckle_size : int, optional
        Minimum speckle size for ``despeckle_field`` (default 5).

    Returns
    -------
    pyart.filters.GateFilter
        Gate filter excluding non-hydrometeor / noise gates.
    """
    texture = range_texture(radar.fields[phidp_name]["data"], winlen=10)
    radar.add_field(
        "PHIDP_TEXTURE",
        {"data": texture, "long_name": "phidp_texture", "units": "degree"},
        replace_existing=True,
    )

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_above("PHIDP_TEXTURE", 60)
    gf.exclude_below(rhohv_name, 0.5)
    gf.exclude_below(refl_name, -20.0)
    # Rescue high-correlation gates (the pos[rhohv > 0.90] = 0 override).
    gf.include_above(rhohv_name, 0.90)

    gf = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf, size=despeckle_size)

    radar.fields.pop("PHIDP_TEXTURE", None)
    return gf


def do_precip_gatefilter(
    radar,
    refl_name: str = "corrected_reflectivity",
    rhohv_name: str = "cross_correlation_ratio",
    snr_name: str = "SNR",
    rho_min: float = 0.8,
    snr_min: float = 3.0,
) -> "pyart.filters.GateFilter":
    """
    Strict precipitation gate filter, reproducing oceanpol_kit's
    ``get_precip_mask`` (RHOHV >= 0.8, SNR >= 3 dB, finite reflectivity).

    Applied before phase processing so KDP noise does not accumulate along the
    ray over clear air.

    Returns
    -------
    pyart.filters.GateFilter
        Gate filter; non-excluded gates are genuine precipitation.
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_below(rhohv_name, rho_min)
    gf.exclude_below(snr_name, snr_min)
    gf.exclude_invalid(refl_name)
    return gf


def do_velocity_gatefilter(
    radar,
    vel_name: str = "VRAD",
    sqi_name: str = "SQI",
    th_name: str = "total_power",
    sqi_min: float = 0.4,
    texture_max: float = None,
    texture_win: int = 5,
    despeckle_size: int = 3,
) -> "pyart.filters.GateFilter":
    """
    Permissive coherence gate filter for velocity, reproducing oceanpol_kit's
    ``get_velocity_mask``.

    The discriminator is coherence and signal presence, not signal strength:
    finite velocity, finite total power (no power -> receiver noise), SQI above
    threshold, optional velocity texture, then despeckle. SNR/RHOHV are
    deliberately not used so clear-air coherent velocity is preserved.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.
    vel_name : str
        Raw (folded) Doppler velocity field name.
    sqi_name : str or None
        Signal-quality-index field name (primary discriminator). If None, a
        velocity-texture threshold of 6.0 is used instead.
    th_name : str
        Total-power field name (detection gate).
    sqi_min : float, optional
        Minimum SQI to keep a gate (default 0.4).
    texture_max : float, optional
        If set, reject gates with velocity texture above this value. Off by
        default when SQI is available; 6.0 is used automatically if SQI is not.
    texture_win : int, optional
        Range window for the texture estimate (default 5).
    despeckle_size : int, optional
        Minimum speckle size for ``despeckle_field`` (default 3).

    Returns
    -------
    pyart.filters.GateFilter
        Gate filter; non-excluded gates are coherent velocity.
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_invalid(vel_name)
    gf.exclude_invalid(th_name)  # no total power -> velocity is receiver noise
    if sqi_name is not None and sqi_name in radar.fields:
        gf.exclude_below(sqi_name, sqi_min)

    use_texture = texture_max if texture_max is not None else (6.0 if sqi_name is None else None)
    if use_texture is not None:
        texture = range_texture(radar.fields[vel_name]["data"], winlen=texture_win)
        radar.add_field(
            "VRAD_TEXTURE",
            {"data": texture, "long_name": "velocity_texture", "units": "m s-1"},
            replace_existing=True,
        )
        gf.exclude_above("VRAD_TEXTURE", use_texture)

    gf = pyart.correct.despeckle_field(radar, vel_name, gatefilter=gf, size=despeckle_size)

    radar.fields.pop("VRAD_TEXTURE", None)
    return gf
