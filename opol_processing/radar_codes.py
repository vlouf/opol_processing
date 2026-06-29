"""
Radar IO and per-field correction helpers for OceanPOL processing.

Field naming across the OceanPOL archive is resolved via ``FIELD_ALIASES``
(single-pol style ``SNR``/``VRAD``/``WRAD`` vs dual-pol ``SNRH``/``VRADH``/
``WRADH``), matching the oceanpol_kit reference. Temperature handling now lives
in ``temperature.py``.

@title: radar_codes
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    resolve_fields
    read_radar
    sort_azimuths
    check_reflectivity
    correct_rhohv
    correct_zdr
    correct_standard_name
    coverage_content_type
    set_significant_digits
    fill_missing
    unravel
"""
import pyart
import numpy as np


# Canonical field name -> ordered list of accepted ODIM aliases.
FIELD_ALIASES = {
    "DBZH": ["DBZH"],
    "TH": ["TH"],
    "RHOHV": ["RHOHV", "RHOHVH"],
    "ZDR": ["ZDR"],
    "PHIDP": ["PHIDP", "PHIDPH"],
    "SNR": ["SNR", "SNRH"],
    "VRAD": ["VRAD", "VRADH"],
    "WRAD": ["WRAD", "WRADH"],
    "SQI": ["SQI", "SQIH"],    
}

# Fields without which the file cannot be processed.
REQUIRED_FIELDS = ["DBZH", "TH", "RHOHV", "ZDR", "PHIDP", "SNR", "VRAD"]


def resolve_fields(radar, aliases: dict = FIELD_ALIASES) -> dict:
    """
    Resolve canonical field names to the variable names actually present in the
    radar, accounting for naming conventions that changed over the archive.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.
    aliases : dict, optional
        Mapping canonical name -> ordered list of accepted aliases.

    Returns
    -------
    dict
        Mapping canonical name -> actual variable name (or None if absent).

    Raises
    ------
    KeyError
        If any field in REQUIRED_FIELDS cannot be resolved.
    """
    available = set(radar.fields.keys())
    resolved = {
        canonical: next((name for name in candidates if name in available), None)
        for canonical, candidates in aliases.items()
    }

    missing = [c for c in REQUIRED_FIELDS if resolved.get(c) is None]
    if missing:
        raise KeyError(
            f"Required field(s) {missing} not found. "
            f"Available variables: {sorted(available)}. "
            f"Add the correct alias to FIELD_ALIASES."
        )

    return resolved


def read_radar(radar_file_name: str):
    """
    Read the input radar file, keeping the original ODIM quantity names
    (resolved later via ``resolve_fields``).

    Parameters
    ----------
    radar_file_name : str
        Radar file name (ODIM .hdf/.h5 or CF/Radial .nc).

    Returns
    -------
    pyart.core.Radar
        Py-ART radar structure.
    """
    if radar_file_name.endswith(".nc"):
        radar = pyart.io.read(radar_file_name)
    else:
        radar = pyart.aux_io.read_odim_h5(radar_file_name, file_field_names=True)

    return sort_azimuths(radar)


def sort_azimuths(radar):
    """
    Sort the azimuth coordinate within each sweep, accounting for wrap-around
    across the -180/+180 degree seam.

    The radar ray ordering is re-ordered per sweep so the azimuth values are
    monotonically increasing after unwrapping, and the matching ray-dependent
    fields and time coordinate are reordered in the same way.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.

    Returns
    -------
    pyart.core.Radar
        The same radar object with sorted azimuths and aligned ray data.
    """
    if getattr(radar, "nsweeps", 0) <= 0:
        return radar

    azimuth = np.asarray(radar.azimuth["data"], dtype="float64")
    elevation = np.asarray(radar.elevation["data"], dtype="float64")
    time_data = np.asarray(radar.time["data"])

    sweep_starts = np.asarray(radar.sweep_start_ray_index["data"], dtype="int64")
    sweep_stops = np.asarray(radar.sweep_end_ray_index["data"], dtype="int64")

    for sweep_idx in range(radar.nsweeps):
        start = int(sweep_starts[sweep_idx])
        stop = int(sweep_stops[sweep_idx]) + 1
        if stop <= start:
            continue

        sweep_azimuth = azimuth[start:stop]
        if sweep_azimuth.size < 2:
            continue

        valid = np.isfinite(sweep_azimuth)
        if np.count_nonzero(valid) != sweep_azimuth.size:
            valid_idx = np.flatnonzero(valid)
            order = np.argsort(np.unwrap(np.deg2rad(sweep_azimuth[valid_idx]), discont=np.pi), kind="stable")
            order_full = np.concatenate((valid_idx[order], np.flatnonzero(~valid)))
        else:
            order_full = np.argsort(np.unwrap(np.deg2rad(sweep_azimuth), discont=np.pi), kind="stable")

        if np.array_equal(order_full, np.arange(sweep_azimuth.size)):
            continue

        azimuth[start:stop] = sweep_azimuth[order_full]
        elevation[start:stop] = elevation[start:stop][order_full]
        time_data[start:stop] = time_data[start:stop][order_full]

        for field in radar.fields.values():
            data = field["data"]
            if not hasattr(data, "shape") or data.ndim < 1:
                continue
            if data.shape[0] != azimuth.size:
                continue
            field["data"][start:stop] = data[start:stop][order_full]

    radar.azimuth["data"] = azimuth
    radar.elevation["data"] = elevation
    radar.time["data"] = time_data
    return radar


def order_sweeps_by_elevation(radar):
    """
    Return a radar whose sweeps are ordered by ascending fixed elevation angle.

    The OceanPOL scan strategy changed from bottom-up to top-down. Downstream
    processing (UNRAVEL 3D, phido non-stationary, bright-band detection) and the
    published product all assume elevation increases with sweep index, so the
    volume is re-ordered when the sweeps are not already ascending.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.

    Returns
    -------
    pyart.core.Radar
        The same object if already ascending, otherwise a new, re-ordered radar
        (via ``Radar.extract_sweeps``).
    """
    angles = np.asarray(radar.fixed_angle["data"], dtype="float64")
    order = np.argsort(angles, kind="stable")
    if np.array_equal(order, np.arange(radar.nsweeps)):
        return radar
    return radar.extract_sweeps(order.tolist())


def check_reflectivity(radar, refl_field_name: str) -> bool:
    """Return True if the radar has a non-empty reflectivity field."""
    dbz = radar.fields[refl_field_name]["data"]
    if np.ma.isMaskedArray(dbz) and dbz.count() == 0:
        return False
    return True


def correct_rhohv(radar, rhohv_name: str = "RHOHV", snr_name: str = "SNR") -> np.ndarray:
    """
    Correct cross-correlation ratio (RHOHV) from noise (Schuur et al. 2003, p7
    eq 5).

    Returns
    -------
    np.ndarray
        Corrected cross-correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()

    natural_snr = 10 ** (0.1 * snr)
    natural_snr = np.ma.filled(natural_snr, -9999)
    rho_corr = rhohv * (1 + 1 / natural_snr)

    # Not allowing the corrected RHOHV to be lower than the raw rhohv.
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return rho_corr


def correct_zdr(radar, zdr_name: str = "ZDR", snr_name: str = "SNR") -> np.ndarray:
    """
    Correct differential reflectivity (ZDR) from noise (Schuur et al. 2003, p7
    eq 6).

    Returns
    -------
    np.ndarray
        Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def correct_standard_name(radar) -> None:
    """
    Restrict ``standard_name`` to entries that exist in the CF standard-name
    table (it is a protected CF keyword).
    """
    for coord in ("range", "azimuth", "elevation", "sweep_number", "fixed_angle", "sweep_mode"):
        try:
            getattr(radar, coord).pop("standard_name")
        except Exception:
            pass

    good_keys = [
        "corrected_reflectivity",
        "total_power",
        "radar_estimated_rain_rate",
        "corrected_velocity",
    ]
    for k in radar.fields.keys():
        if k not in good_keys:
            try:
                radar.fields[k].pop("standard_name")
            except Exception:
                continue

    try:
        radar.fields["velocity"]["standard_name"] = "radial_velocity_of_scatterers_away_from_instrument"
        radar.fields["velocity"]["long_name"] = "Doppler radial velocity of scatterers away from instrument"
    except KeyError:
        pass

    radar.latitude["standard_name"] = "latitude"
    radar.longitude["standard_name"] = "longitude"
    radar.altitude["standard_name"] = "altitude"

    return None


def coverage_content_type(radar) -> None:
    """Add ACDD-1.3 ``coverage_content_type`` attributes."""
    for coord in ("range", "azimuth", "elevation", "latitude", "longitude", "altitude"):
        getattr(radar, coord)["coverage_content_type"] = "coordinate"

    for aux in ("sweep_number", "fixed_angle", "sweep_mode"):
        getattr(radar, aux)["coverage_content_type"] = "auxiliaryInformation"

    for k in radar.fields.keys():
        if k == "radar_echo_classification":
            radar.fields[k]["coverage_content_type"] = "thematicClassification"
        else:
            radar.fields[k]["coverage_content_type"] = "physicalMeasurement"

    return None


def set_significant_digits(radar) -> None:
    """Set the ``_Least_significant_digit`` netCDF attribute (CF/Radial names)."""
    fieldnames = [
        ("velocity", 2),
        ("corrected_velocity", 2),
        ("total_power", 2),
        ("reflectivity", 2),
        ("corrected_reflectivity", 2),
        ("attenuation_corrected_reflectivity", 2),
        ("path_integrated_attenuation", 2),
        ("cross_correlation_ratio", 2),
        ("differential_reflectivity", 2),
        ("corrected_differential_reflectivity", 2),
        ("path_integrated_differential_attenuation", 2),
        ("differential_phase", 4),
        ("corrected_differential_phase", 4),
        ("corrected_specific_differential_phase", 4),
        ("temperature", 1),
        ("radar_estimated_rain_rate", 2),
        ("radar_estimated_snow_rate", 2),
        ("normalized_intercept_parameter", 2),
        ("median_volume_diameter", 2),
    ]
    for key, value in fieldnames:
        try:
            radar.fields[key]["_Least_significant_digit"] = value
        except KeyError:
            continue

    return None


def fill_missing(radar) -> None:
    """
    Replace masked entries with field-appropriate missing values, in place,
    right before writing the netCDF.

    Float fields are filled with NaN (``_FillValue = NaN``); the integer
    ``radar_echo_classification`` is filled with 0 (``_FillValue = 0``), since
    NaN is not representable in an integer field.
    """
    for name, fld in radar.fields.items():
        data = fld["data"]
        if name == "radar_echo_classification":
            if np.ma.isMaskedArray(data):
                data = data.filled(0)
            data = np.where(np.isfinite(np.asarray(data, dtype="float64")), data, 0)
            fld["data"] = np.asarray(data, dtype=np.int16)
            fld["_FillValue"] = np.int16(0)
        else:
            if np.ma.isMaskedArray(data):
                data = data.filled(np.nan)
            fld["data"] = np.asarray(data, dtype=np.float32)
            fld["_FillValue"] = np.float32(np.nan)

    return None


def unravel(radar, gatefilter, vel_name="VRAD", dbz_name="corrected_reflectivity"):
    """
    Unfold Doppler velocity using UNRAVEL.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure (velocity already coherence-censored).
    gatefilter : pyart.filters.GateFilter
        Velocity coherence gate filter.
    vel_name : str
        Name of the (censored) Doppler velocity field.
    dbz_name : str
        Name of the reflectivity field.

    Returns
    -------
    dict
        Py-ART field dictionary of the unfolded Doppler velocity.

    Notes
    -----
    Uses UNRAVEL's defaults (in particular the default 3D strategy), matching
    the oceanpol_kit reference call. The legacy ``strategy="long_range"`` and
    explicit ``alpha`` from the old OPOL pipeline are not used: ``long_range``
    runs extra continuity passes and roughly doubled the per-volume runtime.
    The nyquist velocity (13.3 m s-1) is OceanPOL's, i.e. the value oceanpol_kit
    reads from the ODIM metadata.
    """
    import unravel

    nyquist = 13.3
    unfvel = unravel.unravel_3D_pyart(
        radar, vel_name, dbz_name, gatefilter=gatefilter, nyquist_velocity=nyquist
    )

    vel_meta = pyart.config.get_metadata("velocity")
    vel_meta["data"] = np.ma.masked_where(gatefilter.gate_excluded, unfvel).astype(np.float32)
    vel_meta["_Least_significant_digit"] = 2
    vel_meta["_FillValue"] = np.nan
    vel_meta["comment"] = "UNRAVEL algorithm."
    vel_meta["long_name"] = "Doppler radial velocity of scatterers away from instrument"
    vel_meta["units"] = "m s-1"

    return vel_meta
