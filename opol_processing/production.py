"""
OPOL Level 1b driver.

Reads an ODIM volume, applies the oceanpol_kit science (alias resolution,
date-indexed calibration, robust ERA5/bright-band temperature, coherence-based
velocity censoring + UNRAVEL, precip-gated PHIDO phase, Z-PHI attenuation,
Southern-Ocean rainfall, DSD/snow/HID) and writes a CF/Radial netCDF with
verbose variable names.

@project: OCEANPol
@title: production
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology and Monash University

.. autosummary::
    :toctree: generated/

    _mkdir
    process_and_save
    production_line
"""
# Python Standard Library
import os
import re
import time
import uuid
import datetime
import warnings

# Other Libraries
import pyart
import cftime
import numpy as np
import pandas as pd

# Custom modules.
from . import attenuation
from . import calibration
from . import filtering
from . import hydrometeors
from . import phase
from . import radar_codes
from . import temperature


# If the number of coherent velocity gates left after noise censoring is below
# this value, UNRAVEL is skipped and the censored velocity is passed through.
MIN_UNRAVEL_GATES = 100

HID_COMMENT = (
    "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; 5: Wet Snow; "
    "6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"
)

# Final CF/Radial variables to publish; everything else is dropped before write.
KEEP_FIELDS = {
    "total_power",
    "reflectivity",
    "corrected_reflectivity",
    "attenuation_corrected_reflectivity",
    "path_integrated_attenuation",
    "differential_reflectivity",
    "corrected_differential_reflectivity",
    "path_integrated_differential_attenuation",
    "cross_correlation_ratio",
    "velocity",
    "corrected_velocity",
    "differential_phase",
    "corrected_differential_phase",
    "corrected_specific_differential_phase",
    "temperature",
    "radar_echo_classification",
    "radar_estimated_rain_rate",
    "radar_estimated_snow_rate",
    "normalized_intercept_parameter",
    "median_volume_diameter",
}


def _meta(data, **kwargs) -> dict:
    """Build a Py-ART field dictionary."""
    field = {"data": data}
    field.update(kwargs)
    return field


def _toc(label: str, t0: float, debug: bool) -> float:
    """Print the elapsed time for a step (when debug) and return a fresh tic."""
    now = time.time()
    if debug:
        print(f"  [{label:<22}] {now - t0:7.3f} s")
    return now


def _mkdir(path):
    """Make a directory, tolerating concurrent creation (multiprocessing)."""
    if os.path.exists(path):
        return None
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    return None


def production_line(radar_file_name, do_dealiasing=True, use_csu=True, debug=False):
    """
    Production line for correcting and estimating OPOL radar parameters.

    Parameters
    ----------
    radar_file_name : str
        Name of the input radar file (ODIM .hdf/.h5).
    do_dealiasing : bool
        Dealias velocity with UNRAVEL.
    use_csu : bool
        Compute HID/rainfall/DSD/snow products.
    debug : bool
        If True, print the wall-clock time taken by each processing step.

    Returns
    -------
    radar : pyart.core.Radar or None
        Processed radar (None if the file is unsuitable, e.g. < 10 sweeps).
    """
    st = time.time()
    t = st
    if debug:
        print(f"Processing {radar_file_name}")

    radar = radar_codes.read_radar(radar_file_name)
    if radar.nsweeps < 10:
        if debug:
            print(f"  Skipped: only {radar.nsweeps} sweeps (< 10).")
        return None
    t = _toc("read", t, debug)

    # Ensure sweeps are ordered by ascending elevation (the OceanPOL scan
    # strategy changed from bottom-up to top-down; downstream 3D steps assume
    # increasing elevation with sweep index).
    radar = radar_codes.order_sweeps_by_elevation(radar)
    if debug:
        print(f"  sweep elevations: {np.round(np.asarray(radar.fixed_angle['data']), 2)}")
    t = _toc("order sweeps", t, debug)

    # Resolve field names once for the whole volume (raises if a required field
    # is missing).
    fields = radar_codes.resolve_fields(radar)
    dbz_name = fields["DBZH"]
    th_name = fields["TH"]
    zdr_name = fields["ZDR"]
    rho_name = fields["RHOHV"]
    phidp_name = fields["PHIDP"]
    snr_name = fields["SNR"]
    vel_name = fields["VRAD"]
    sqi_name = fields["SQI"]

    # Correct time units (signal processing sometimes drops a space).
    if "since " not in radar.time["units"]:
        radar.time["units"] = radar.time["units"].replace("since", "since ")
    rdate = pd.Timestamp(cftime.num2pydate(radar.time["data"][0], radar.time["units"]))

    if not radar_codes.check_reflectivity(radar, dbz_name):
        raise TypeError(f"Reflectivity field is empty in {radar_file_name}.")

    # Date-indexed calibration offsets (dB).
    cal_offset, zdr_offset = calibration.get_calib_offset(rdate)

    r = radar.range["data"]
    lat = float(np.asarray(radar.latitude["data"]).ravel()[0])
    lon = float(np.asarray(radar.longitude["data"]).ravel()[0])
    southern_ocean = lat < -40
    t = _toc("resolve+calib", t, debug)

    # --- RHOHV noise correction ---
    rho_corr = radar_codes.correct_rhohv(radar, rhohv_name=rho_name, snr_name=snr_name)
    radar.add_field(
        "cross_correlation_ratio",
        _meta(rho_corr, long_name="Corrected cross correlation ratio", units="1"),
        replace_existing=True,
    )
    t = _toc("rhohv", t, debug)

    # --- Temperature profile (ERA5 -> bright band -> default) ---
    geo_h, temp_k = temperature.get_volume_temperature_profile(rdate, lat, lon, radar, fields)
    temps = temperature.interp_temperature(geo_h, temp_k, radar.gate_altitude["data"])
    radar.add_field(
        "temperature",
        _meta(
            temps.astype(np.float32),
            long_name="Temperature at gate",
            units="degrees Celsius",
            comment=f"Profile date: {rdate:%Y/%m/%d}",
        ),
        replace_existing=True,
    )
    t = _toc("temperature", t, debug)

    # --- Reflectivity cleaning (hydrometeor gate filter) ---
    gf = filtering.do_gatefilter(radar, vel_name=vel_name, sqi_name=sqi_name, th_name=th_name)
    th = radar.fields[th_name]["data"]
    # Numba speckle filter on the masked total power (replaces pyart despeckle).
    refl = np.ascontiguousarray(np.ma.filled(np.ma.masked_where(gf.gate_excluded, th), np.nan), dtype="float64")
    mask_i8 = np.ascontiguousarray(gf.gate_excluded.astype(np.int8))
    dbz_clean = filtering.speckle_filter(refl, mask_i8).astype(np.float32) + cal_offset
    dbz_clean = np.ma.masked_invalid(dbz_clean)
    radar.add_field(
        "corrected_reflectivity",
        _meta(
            dbz_clean,
            long_name="Corrected reflectivity",
            units="dBZ",
            standard_name="corrected_equivalent_reflectivity_factor",
            comment="Cleaned and calibrated; attenuation NOT applied (see attenuation_corrected_reflectivity).",
        ),
        replace_existing=True,
    )
    t = _toc("refl cleaning", t, debug)

    # --- PHIDP / KDP (precip-gated PHIDO) ---
    precip_gf = filtering.do_precip_gatefilter(
        radar, refl_name="corrected_reflectivity", rhohv_name="cross_correlation_ratio", snr_name=snr_name
    )
    refl_for_phi = np.ma.filled(dbz_clean, np.nan)
    phidp_corr, kdp = phase.get_phidp(radar, phidp_name, precip_gf, refl_for_phi, temps)
    radar.add_field(
        "corrected_differential_phase",
        _meta(phidp_corr.astype(np.float32), long_name="Corrected differential phase (PHIDO)", units="degree"),
        replace_existing=True,
    )
    radar.add_field(
        "corrected_specific_differential_phase",
        _meta(kdp.astype(np.float32), long_name="Corrected specific differential phase (PHIDO)", units="degree/km"),
        replace_existing=True,
    )
    t = _toc("phidp/kdp (phido)", t, debug)

    # --- Velocity dealiasing (coherence censoring + UNRAVEL) ---
    if do_dealiasing and vel_name in radar.fields:        
        n_coherent = int(np.count_nonzero(~gf.gate_excluded))
        censored = np.ma.masked_where(gf.gate_excluded, radar.fields[vel_name]["data"])
        if debug:
            print(f"  coherent velocity gates: {n_coherent}")

        if n_coherent < MIN_UNRAVEL_GATES:
            print(f"Skipping UNRAVEL: {n_coherent} coherent velocity gates (< {MIN_UNRAVEL_GATES}).")
            vel_meta = pyart.config.get_metadata("velocity")
            vel_meta["data"] = censored.astype(np.float32)
            vel_meta["units"] = "m s-1"
            vel_meta["comment"] = f"UNRAVEL skipped ({n_coherent} coherent gates)."
        else:
            radar.add_field("VEL_CENSORED", _meta(censored, units="m s-1"), replace_existing=True)
            vel_meta = radar_codes.unravel(radar, gf, vel_name="VEL_CENSORED", dbz_name="corrected_reflectivity")
            radar.fields.pop("VEL_CENSORED", None)

        radar.add_field("corrected_velocity", vel_meta, replace_existing=True)
    t = _toc("dealiasing (unravel)", t, debug)

    # --- ZH attenuation (Z-PHI) ---
    pia = attenuation.correct_attenuation(r, dbz_clean, phidp_corr, temps)
    radar.add_field(
        "path_integrated_attenuation",
        _meta(pia.astype(np.float32), long_name="Path integrated attenuation", units="dB"),
        replace_existing=True,
    )
    dbz_atten = np.ma.masked_invalid((dbz_clean + pia).astype(np.float32))
    radar.add_field(
        "attenuation_corrected_reflectivity",
        _meta(
            dbz_atten,
            long_name="Attenuation corrected reflectivity",
            units="dBZ",
            comment="Corrected reflectivity with Z-PHI path-integrated attenuation added.",
        ),
        replace_existing=True,
    )
    t = _toc("attenuation (zh)", t, debug)

    # --- ZDR noise + differential attenuation correction ---
    radar.fields[zdr_name]["data"] = radar.fields[zdr_name]["data"] + zdr_offset
    corr_zdr = radar_codes.correct_zdr(radar, zdr_name=zdr_name, snr_name=snr_name)
    corr_zdr = np.ma.masked_where(np.ma.getmaskarray(dbz_clean), corr_zdr)
    radar.add_field(
        "corrected_differential_reflectivity",
        _meta(corr_zdr.astype(np.float32), long_name="Corrected differential reflectivity", units="dB"),
        replace_existing=True,
    )
    zdr_atten = attenuation.correct_attenuation_zdr(radar, gf, phidp_name="corrected_differential_phase")
    radar.add_field("path_integrated_differential_attenuation", zdr_atten, replace_existing=True)
    t = _toc("zdr (+ diff atten)", t, debug)

    # --- Retrieval products ---
    if use_csu:
        dbz_arr = np.ma.filled(dbz_clean, np.nan)
        zdr_arr = np.ma.filled(corr_zdr, np.nan)
        kdp_arr = np.asarray(kdp)
        rho_arr = np.asarray(rho_corr)
        t_arr = np.ma.filled(temps, np.nan)

        hid = hydrometeors.compute_hid(dbz_arr, zdr_arr, kdp_arr, rho_arr, t_arr)
        radar.add_field(
            "radar_echo_classification",
            _meta(
                np.ma.masked_equal(hid.astype(np.int16), 0),
                long_name="Hydrometeor classification",
                units="1",
                comments=HID_COMMENT,
            ),
            replace_existing=True,
        )

        rain = hydrometeors.get_rainfall_estimate(dbz_arr, zdr_arr, kdp_arr, t_arr, southern_ocean)
        radar.add_field(
            "radar_estimated_rain_rate",
            _meta(
                rain.astype(np.float32),
                long_name="Rainfall rate",
                units="mm h-1",
                standard_name="rainfall_rate",
                comment="Southern-Ocean coefficient set." if southern_ocean else "Tropical coefficient set.",
            ),
            replace_existing=True,
        )

        nw, d0 = hydrometeors.get_dsd_estimate(dbz_arr, zdr_arr, t_arr)
        radar.add_field(
            "normalized_intercept_parameter",
            _meta(nw.astype(np.float32), long_name="Normalized intercept parameter (log10 Nw)", units="log10(mm-1 m-3)"),
            replace_existing=True,
        )
        radar.add_field(
            "median_volume_diameter",
            _meta(d0.astype(np.float32), long_name="Median volume diameter D0", units="mm"),
            replace_existing=True,
        )

        snow = hydrometeors.get_snowfall_estimate(dbz_arr, kdp_arr, t_arr)
        radar.add_field(
            "radar_estimated_snow_rate",
            _meta(snow.astype(np.float32), long_name="Snowfall rate", units="mm h-1"),
            replace_existing=True,
        )
        t = _toc("products (hid/rain/dsd)", t, debug)

    # --- Rename surviving raw ODIM fields to CF/Radial verbose names ---
    rename = {
        th_name: "total_power",
        dbz_name: "reflectivity",
        zdr_name: "differential_reflectivity",
        vel_name: "velocity",
        phidp_name: "differential_phase",
    }
    for old, new in rename.items():
        if old in radar.fields and old != new:
            radar.add_field(new, radar.fields.pop(old), replace_existing=True)

    # --- Drop everything not in the published set ---
    for key in list(radar.fields.keys()):
        if key not in KEEP_FIELDS:
            radar.fields.pop(key, None)

    # --- Finalise metadata, fill missing values, return ---
    radar_codes.set_significant_digits(radar)
    radar_codes.correct_standard_name(radar)
    radar_codes.coverage_content_type(radar)
    radar_codes.fill_missing(radar)
    t = _toc("finalise", t, debug)

    if debug:
        print(f"  [{'TOTAL production_line':<22}] {time.time() - st:7.3f} s")

    return radar


def process_and_save(radar_file_name, outpath, do_dealiasing=True, use_csu=True, debug=False):
    """
    Run the production line and write the CF/Radial netCDF output.

    Parameters
    ----------
    radar_file_name : str
        Name of the input radar file.
    outpath : str
        Root path for saving output data.
    do_dealiasing : bool
        Dealias velocity with UNRAVEL.
    use_csu : bool
        Compute HID/rainfall/DSD/snow products.
    debug : bool
        If True, print per-step and write timings.
    """
    today = datetime.datetime.utcnow()

    datestr = re.findall("[0-9]{8}", os.path.basename(radar_file_name))[0]
    outpath_ppi = os.path.join(outpath, "ppi", datestr)
    _mkdir(outpath_ppi)

    base = os.path.basename(radar_file_name)
    outfilename = re.sub(r"\.(hdf|h5|nc)$", "", base) + ".cfradial.nc"
    outfilename = os.path.join(outpath_ppi, outfilename)
    if os.path.isfile(outfilename):
        print(f"Output file {outfilename} already exists.")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = production_line(radar_file_name, do_dealiasing=do_dealiasing, use_csu=use_csu, debug=debug)
        if radar is None:
            print(f"{radar_file_name} has not been processed. Check logs.")
            return None

    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    radar_end_date = cftime.num2pydate(radar.time["data"][-1], radar.time["units"])

    latitude = radar.gate_latitude["data"]
    longitude = radar.gate_longitude["data"]
    maxlon, minlon = longitude.max(), longitude.min()
    maxlat, minlat = latitude.max(), latitude.min()

    unique_id = str(uuid.uuid4())
    metadata = {
        "Conventions": "CF-1.6, ACDD-1.3",
        "country": "Australia",
        "creator_email": "CPOL-support@bom.gov.au",
        "creator_name": "Commonwealth of Australia, Bureau of Meteorology, Science and Innovation, Research, Weather and Environmental Prediction, Radar Science and Nowcasting",
        "creator_url": "https://bom365.sharepoint.com/sites/SI_WEP_RSAN",
        "date_created": today.isoformat(),
        "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
        "geospatial_lat_max": f"{maxlat:0.6}",
        "geospatial_lat_min": f"{minlat:0.6}",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_max": f"{maxlon:0.6}",
        "geospatial_lon_min": f"{minlon:0.6}",
        "geospatial_lon_units": "degrees_east",
        "history": "created by Valentin Louf on gadi.nci.org.au at " + today.isoformat() + " using Py-ART",
        "id": unique_id,
        "institution": "Bureau of Meteorology",
        "instrument": "radar",
        "instrument_name": "OPOL",
        "instrument_type": "radar",
        "keywords": "R/V INVESTIGATOR, POLARIMETRIC RADAR, C-BAND RADAR",
        "keywords_vocabulary": "NASA Global Change Master Directory (GCMD) Science Keywords",
        "license": "CC BY-NC-SA 4.0",
        "naming_authority": "au.gov.bom",
        "origin_altitude": radar.altitude["data"][0],
        "origin_latitude": radar.latitude["data"][0],
        "origin_longitude": radar.longitude["data"][0],
        "platform_is_mobile": "true",
        "processing_level": "L2",
        "project": "OPOL",
        "publisher_name": "NCI",
        "publisher_url": "nci.gov.au",
        "product_version": f"v{today.year}.{today.month:02}",
        "site_name": "RV Investigator",
        "source": "radar",
        "standard_name_vocabulary": "CF Standard Name Table v71",
        "summary": "Volumetric scan from OPOL dual-polarization Doppler radar (RV Investigator)",
        "time_coverage_start": radar_start_date.isoformat(),
        "time_coverage_end": radar_end_date.isoformat(),
        "time_coverage_duration": "P06M",
        "time_coverage_resolution": "PT06M",
        "title": "radar PPI volume from OPOL",
        "uuid": unique_id,
    }
    radar.metadata = metadata

    tw = time.time()
    pyart.io.write_cfradial(outfilename, radar, format="NETCDF4")
    if debug:
        print(f"  [{'write_cfradial':<22}] {time.time() - tw:7.3f} s")

    return None
