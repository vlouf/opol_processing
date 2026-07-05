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

    production_line
    process_and_save
"""
# Python Standard Library
import os
import sys
import time
import uuid
import datetime
import warnings
import json

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
from . import utils

# --- Drop low elevations (≤ 0.6°) and truncate to 150 km ---
MIN_ELEVATION_DEG = 0.6
MAX_RANGE_M = 150_000.0

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
    # "path_integrated_attenuation",
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
    "signal_to_noise_ratio",
    "signal_quality_index",
    "spectrum_width",
}


def _get_rss_mb():
    """Return (rss_mb, peak_rss_mb) when available, else (None, None)."""
    rss_mb = None
    peak_rss_mb = None

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            rss_kb = float(parts[1])
                            rss_mb = rss_kb / 1024.0
                        break
        except (OSError, ValueError):
            pass

    try:
        resource_mod = __import__("resource")
        ru_maxrss = float(resource_mod.getrusage(resource_mod.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            peak_rss_mb = ru_maxrss / (1024.0 * 1024.0)
        else:
            peak_rss_mb = ru_maxrss / 1024.0
    except (ImportError, AttributeError, OSError, ValueError):
        pass

    return rss_mb, peak_rss_mb


def _log_memory(stage, radar=None, extra=""):
    """Print lightweight memory diagnostics for ad-hoc HPC profiling."""
    rss_mb, peak_rss_mb = _get_rss_mb()
    dims = ""
    if radar is not None:
        dims = " nsweeps=%s nrays=%s ngates=%s" % (radar.nsweeps, radar.nrays, radar.ngates)
    rss_txt = "n/a" if rss_mb is None else "%.1f" % rss_mb
    peak_txt = "n/a" if peak_rss_mb is None else "%.1f" % peak_rss_mb
    print(
        "[mem] pid=%s stage=%s rss_mb=%s peak_rss_mb=%s%s%s"
        % (os.getpid(), stage, rss_txt, peak_txt, dims, (" " + extra) if extra else "")
    )


def _append_memory_profile_row(
    stage,
    file_name,
    rss_mb,
    peak_rss_mb,
    radar=None,
    extra="",
):
    """Append one JSONL memory profile row to a shared log path when configured."""
    log_path = os.environ.get("OPOL_MEMLOG_PATH")
    if not log_path:
        return

    payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pid": os.getpid(),
        "stage": stage,
        "file": file_name,
        "rss_mb": rss_mb,
        "peak_rss_mb": peak_rss_mb,
        "extra": extra,
    }
    if radar is not None:
        payload["nsweeps"] = radar.nsweeps
        payload["nrays"] = radar.nrays
        payload["ngates"] = radar.ngates

    line = json.dumps(payload, separators=(",", ":")) + "\n"
    try:
        fd = os.open(log_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def _profile_memory(stage, radar=None, file_name="", extra=""):
    """Emit memory profile output to stdout and optional JSONL log file."""
    rss_mb, peak_rss_mb = _get_rss_mb()
    rss_txt = "n/a" if rss_mb is None else "%.1f" % rss_mb
    peak_txt = "n/a" if peak_rss_mb is None else "%.1f" % peak_rss_mb

    dims = ""
    if radar is not None:
        dims = " nsweeps=%s nrays=%s ngates=%s" % (radar.nsweeps, radar.nrays, radar.ngates)

    print(
        "[mem] pid=%s stage=%s file=%s rss_mb=%s peak_rss_mb=%s%s%s"
        % (
            os.getpid(),
            stage,
            file_name or "-",
            rss_txt,
            peak_txt,
            dims,
            (" " + extra) if extra else "",
        ),
        flush=True,
    )
    _append_memory_profile_row(
        stage=stage,
        file_name=file_name,
        rss_mb=rss_mb,
        peak_rss_mb=peak_rss_mb,
        radar=radar,
        extra=extra,
    )


def production_line(radar_file_name, do_dealiasing=True, use_csu=True, debug=False, profile_memory=False):
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
    basename = os.path.basename(radar_file_name)
    if debug:
        print(f"Processing {radar_file_name}")
    if profile_memory:
        _profile_memory("start", file_name=basename)

    radar = radar_codes.read_radar(radar_file_name)
    if profile_memory:
        _profile_memory("after-read", radar=radar, file_name=basename)
    if radar.nsweeps < 10:
        if debug:
            print(f"  Skipped: only {radar.nsweeps} sweeps (< 10).")
        return None
    t = utils.toc("read", t, debug)

    # Ensure sweeps are ordered by ascending elevation (the OceanPOL scan
    # strategy changed from bottom-up to top-down; downstream 3D steps assume
    # increasing elevation with sweep index).
    radar = radar_codes.order_sweeps_by_elevation(radar)
    radar = radar_codes.sort_azimuths(radar)
    if debug:
        print(f"  sweep elevations: {np.round(np.asarray(radar.fixed_angle['data']), 2)}")
    t = utils.toc("order sweeps", t, debug)

    # Decimate rays from 0.5° to 1° azimuth resolution if needed
    # (Must be done AFTER sweep ordering, as ordering uses sweep indices)
    if radar.nrays > 2:
        az_diff = np.abs(np.diff(radar.azimuth['data'][:2]))
        if az_diff[0] < 0.75:  # Likely 0.5° resolution
            radar = utils.decimate_rays_to_1degree(radar, debug=debug)
            if debug:
                print(f"  Decimated rays from 0.5° to 1° azimuth resolution")
            t = utils.toc("decimate rays", t, debug)
        
    ind_sweeps = np.where(radar.fixed_angle["data"] > MIN_ELEVATION_DEG)[0].tolist()
    if len(ind_sweeps) < radar.nsweeps:
        if debug:
            dropped = radar.nsweeps - len(ind_sweeps)
            print(f"  Dropping {dropped} sweep(s) \u2264 {MIN_ELEVATION_DEG}\u00b0; {len(ind_sweeps)} remaining")
        radar = radar.extract_sweeps(ind_sweeps)

    ind_rng = np.where(radar.range["data"] <= MAX_RANGE_M)[0]
    if ind_rng.size < radar.ngates:
        radar.range["data"] = radar.range["data"][ind_rng]
        radar.ngates = ind_rng.size
        radar.init_gate_x_y_z()
        radar.init_gate_longitude_latitude()
        radar.init_gate_altitude()
        for _f in list(radar.fields):
            radar.fields[_f]["data"] = radar.fields[_f]["data"][:, ind_rng]
        if debug:
            print(f"  Range truncated to {MAX_RANGE_M / 1000:.0f} km ({ind_rng.size} gates)")
    t = utils.toc("domain trim", t, debug)
    if profile_memory:
        _profile_memory("after-domain-trim", radar=radar, file_name=basename)

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
    t = utils.toc("resolve+calib", t, debug)

    # --- RHOHV noise correction ---
    rho_corr = radar_codes.correct_rhohv(radar, rhohv_name=rho_name, snr_name=snr_name)
    radar.add_field(
        "cross_correlation_ratio",
        utils.meta(rho_corr, long_name="Corrected cross correlation ratio", units="1"),
        replace_existing=True,
    )
    t = utils.toc("rhohv", t, debug)

    # --- Temperature profile (ERA5 -> bright band -> default) ---
    geo_h, temp_k = temperature.get_volume_temperature_profile(rdate, lat, lon, radar, fields)
    temps = temperature.interp_temperature(geo_h, temp_k, radar.gate_altitude["data"])
    radar.add_field(
        "temperature",
        utils.meta(
            temps.astype(np.float32),
            long_name="Temperature at gate",
            units="degrees Celsius",
            comment=f"Profile date: {rdate:%Y/%m/%d}",
        ),
        replace_existing=True,
    )
    t = utils.toc("temperature", t, debug)
    if profile_memory:
        _profile_memory("after-temperature", radar=radar, file_name=basename)

    # --- Reflectivity cleaning (hydrometeor gate filter) ---
    gf = filtering.do_gatefilter_opol(
        radar, refl_name=th_name, rhohv_name="cross_correlation_ratio", phidp_name=phidp_name, zdr_name=zdr_name
    )
    th = radar.fields[th_name]["data"]
    # Numba speckle filter on the masked total power (replaces pyart despeckle).
    refl = np.ascontiguousarray(np.ma.filled(np.ma.masked_where(gf.gate_excluded, th), np.nan), dtype="float64")
    mask_i8 = np.ascontiguousarray(gf.gate_excluded.astype(np.int8))
    dbz_clean = filtering.speckle_filter(refl, mask_i8).astype(np.float32) + cal_offset
    dbz_clean = np.ma.masked_invalid(dbz_clean)
    radar.add_field(
        "corrected_reflectivity",
        utils.meta(
            dbz_clean,
            long_name="Corrected reflectivity",
            units="dBZ",
            standard_name="corrected_equivalent_reflectivity_factor",
            comment="Cleaned and calibrated; attenuation NOT applied (see attenuation_corrected_reflectivity).",
        ),
        replace_existing=True,
    )
    t = utils.toc("refl cleaning", t, debug)
    if profile_memory:
        _profile_memory("after-refl-cleaning", radar=radar, file_name=basename)

    # --- PHIDP / KDP (precip-gated PHIDO) ---
    precip_gf = filtering.do_precip_gatefilter(
        radar, refl_name="corrected_reflectivity", rhohv_name="cross_correlation_ratio", snr_name=snr_name
    )
    refl_for_phi = np.ma.filled(dbz_clean, np.nan)
    if profile_memory:
        _profile_memory("before-phido", radar=radar, file_name=basename)
    phidp_corr, kdp = phase.get_phidp(radar, phidp_name, precip_gf, refl_for_phi, temps)
    radar.add_field(
        "corrected_differential_phase",
        utils.meta(phidp_corr.astype(np.float32), long_name="Corrected differential phase (PHIDO)", units="degree"),
        replace_existing=True,
    )
    radar.add_field(
        "corrected_specific_differential_phase",
        utils.meta(kdp.astype(np.float32), long_name="Corrected specific differential phase (PHIDO)", units="degree/km"),
        replace_existing=True,
    )
    t = utils.toc("phidp/kdp (phido)", t, debug)
    if profile_memory:
        _profile_memory("after-phido", radar=radar, file_name=basename)

    # --- Velocity dealiasing (coherence censoring + UNRAVEL) ---
    if do_dealiasing and vel_name in radar.fields:
        vgf = filtering.do_velocity_gatefilter(radar, vel_name=vel_name, sqi_name=sqi_name, th_name=th_name)
        n_coherent = int(np.count_nonzero(~vgf.gate_excluded))
        censored = np.ma.masked_where(vgf.gate_excluded, radar.fields[vel_name]["data"])

        if n_coherent < MIN_UNRAVEL_GATES:
            print(f"Skipping UNRAVEL: {n_coherent} coherent velocity gates (< {MIN_UNRAVEL_GATES}).")
            vel_meta = pyart.config.get_metadata("velocity")
            vel_meta["data"] = censored.astype(np.float32)
            vel_meta["units"] = "m s-1"
            vel_meta["comment"] = f"UNRAVEL skipped ({n_coherent} coherent gates)."
        else:
            radar.add_field("VEL_CENSORED", utils.meta(censored, units="m s-1"), replace_existing=True)
            if profile_memory:
                _profile_memory("before-unravel", radar=radar, file_name=basename, extra="coherent=%s" % n_coherent)
            vel_meta = radar_codes.unravel(radar, vgf, vel_name="VEL_CENSORED", dbz_name="corrected_reflectivity")
            radar.fields.pop("VEL_CENSORED", None)
            if profile_memory:
                _profile_memory("after-unravel", radar=radar, file_name=basename, extra="coherent=%s" % n_coherent)

        radar.add_field("corrected_velocity", vel_meta, replace_existing=True)
    t = utils.toc("dealiasing (unravel)", t, debug)

    # --- ZH attenuation (Z-PHI) ---
    pia = attenuation.correct_attenuation(r, dbz_clean, phidp_corr, temps)
    radar.add_field(
        "path_integrated_attenuation",
        utils.meta(pia.astype(np.float32), long_name="Path integrated attenuation", units="dB"),
        replace_existing=True,
    )
    dbz_atten = np.ma.masked_invalid((dbz_clean + pia).astype(np.float32))
    radar.add_field(
        "attenuation_corrected_reflectivity",
        utils.meta(
            dbz_atten,
            long_name="Attenuation corrected reflectivity",
            units="dBZ",
            comment="Corrected reflectivity with Z-PHI path-integrated attenuation added.",
        ),
        replace_existing=True,
    )
    t = utils.toc("attenuation (zh)", t, debug)

    # --- ZDR noise + differential attenuation correction ---
    radar.fields[zdr_name]["data"] = radar.fields[zdr_name]["data"] + zdr_offset
    corr_zdr = radar_codes.correct_zdr(radar, zdr_name=zdr_name, snr_name=snr_name)
    corr_zdr = np.ma.masked_where(np.ma.getmaskarray(dbz_clean), corr_zdr)
    radar.add_field(
        "corrected_differential_reflectivity",
        utils.meta(corr_zdr.astype(np.float32), long_name="Corrected differential reflectivity", units="dB"),
        replace_existing=True,
    )
    zdr_atten = attenuation.correct_attenuation_zdr(radar, gf, phidp_name="corrected_differential_phase")
    radar.add_field("path_integrated_differential_attenuation", zdr_atten, replace_existing=True)
    t = utils.toc("zdr (+ diff atten)", t, debug)
    if profile_memory:
        _profile_memory("after-zdr", radar=radar, file_name=basename)

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
            utils.meta(
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
            utils.meta(
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
            utils.meta(nw.astype(np.float32), long_name="Normalized intercept parameter (log10 Nw)", units="log10(mm-1 m-3)"),
            replace_existing=True,
        )
        radar.add_field(
            "median_volume_diameter",
            utils.meta(d0.astype(np.float32), long_name="Median volume diameter D0", units="mm"),
            replace_existing=True,
        )

        snow = hydrometeors.get_snowfall_estimate(dbz_arr, kdp_arr, t_arr)
        radar.add_field(
            "radar_estimated_snow_rate",
            utils.meta(snow.astype(np.float32), long_name="Snowfall rate", units="mm h-1"),
            replace_existing=True,
        )
        t = utils.toc("products (hid/rain/dsd)", t, debug)
        if profile_memory:
            nbytes_mib = (
                dbz_arr.nbytes + zdr_arr.nbytes + kdp_arr.nbytes + rho_arr.nbytes + t_arr.nbytes
            ) / (1024.0 * 1024.0)
            _profile_memory(
                "after-products",
                radar=radar,
                file_name=basename,
                extra="temp_arrays_mib=%.1f" % nbytes_mib,
            )

    # --- Rename surviving raw ODIM fields to CF/Radial verbose names ---
    rename = {
        th_name: "total_power",
        dbz_name: "reflectivity",
        zdr_name: "differential_reflectivity",
        vel_name: "velocity",
        phidp_name: "differential_phase",
        snr_name: "signal_to_noise_ratio",
        fields["WRAD"]: "spectrum_width",
        sqi_name: "signal_quality_index",
    }
    for old, new in rename.items():
        if old in radar.fields and old != new:
            radar.add_field(new, radar.fields.pop(old), replace_existing=True)

    # --- Drop everything not in the published set ---
    for key in list(radar.fields.keys()):
        if key not in KEEP_FIELDS:
            radar.fields.pop(key, None)

    # --- Apply TH mask to raw moments (fields whose noise is not informative) ---
    # cross_correlation_ratio, differential_phase, signal_to_noise_ratio,
    # signal_quality_index and temperature are intentionally left unmasked.
    th_mask = np.ma.getmaskarray(dbz_clean)
    for _raw in ("reflectivity", "differential_reflectivity", "velocity", "spectrum_width"):
        if _raw in radar.fields:
            radar.fields[_raw]["data"] = np.ma.masked_where(th_mask, radar.fields[_raw]["data"])

    # --- Finalise metadata, fill missing values, return ---
    radar_codes.set_significant_digits(radar)
    radar_codes.correct_standard_name(radar)
    radar_codes.coverage_content_type(radar)
    radar_codes.fill_missing(radar)
    t = utils.toc("finalise", t, debug)
    if profile_memory:
        _profile_memory("after-finalise", radar=radar, file_name=basename)

    if debug:
        print(f"  [{'TOTAL production_line':<22}] {time.time() - st:7.3f} s")

    return radar


def process_and_save(
    radar_file_name,
    output_filename,
    do_dealiasing=True,
    use_csu=True,
    debug=False,
    do_return=False,
    exist_ok=False,
    profile_memory=False,
):
    """
    Run the production line and write the CF/Radial netCDF output.

    Parameters
    ----------
    radar_file_name : str
        Name of the input radar file.
    output_filename : str
        Full path to output netCDF file.
    do_dealiasing : bool
        Dealias velocity with UNRAVEL.
    use_csu : bool
        Compute HID/rainfall/DSD/snow products.
    debug : bool
        If True, print per-step and write timings.
    do_return : bool
        If True, return the processed radar object (for testing).
    exist_ok : bool
        If False, raise error if output file already exists. If True, overwrite.
    """
    if os.path.isfile(output_filename) and not exist_ok:
        raise FileExistsError(f"Output file {output_filename} already exists. Use exist_ok=True to overwrite.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = production_line(
            radar_file_name,
            do_dealiasing=do_dealiasing,
            use_csu=use_csu,
            debug=debug,
            profile_memory=profile_memory,
        )
        if radar is None:
            print(f"{radar_file_name} has not been processed. Check logs.")
            return None

    today = datetime.datetime.now(datetime.timezone.utc)
    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    radar_end_date = cftime.num2pydate(radar.time["data"][-1], radar.time["units"])

    latitude = radar.gate_latitude["data"]
    longitude = radar.gate_longitude["data"]
    maxlon, minlon = longitude.max(), longitude.min()
    maxlat, minlat = latitude.max(), latitude.min()

    unique_id = str(uuid.uuid4())
    metadata = {
        "Conventions": "CF-1.7, CF-Radial-1.4, ACDD-1.3",
        "country": "Australia",
        "creator_email": "valentin.louf@bom.gov.au",
        "creator_name": "Commonwealth of Australia, Bureau of Meteorology, Science and Innovation, Research, Weather and Environmental Prediction, Radar Science and Nowcasting",
        "creator_url": "https://bom365.sharepoint.com/sites/SI_WEP_RSAN",
        "date_created": today.isoformat(),
        "georefs_applied": "1",
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
        "primary_axis": "axis_z",
        "processing_level": "L2",
        "project": "OPOL",
        "publisher_name": "NCI",
        "publisher_url": "nci.gov.au",
        "product_version": f"v{today.year}.{today.month:02}",
        "ray_times_increase": "true" if np.all(np.diff(radar.time["data"]) >= 0) else "false",
        "site_name": "RV Investigator",
        "source": "radar",
        "standard_name_vocabulary": "CF Standard Name Table v71",
        "summary": "Volumetric scan from OPOL dual-polarization Doppler radar (RV Investigator)",
        "time_coverage_start": radar_start_date.isoformat(),
        "time_coverage_end": radar_end_date.isoformat(),
        "time_coverage_duration": "PT05M",
        "time_coverage_resolution": "PT05M",
        "title": "radar PPI volume from OPOL",
        "uuid": unique_id,
    }
    radar.metadata = metadata

    tw = time.time()
    if profile_memory:
        _profile_memory("before-write", radar=radar, file_name=os.path.basename(radar_file_name))
    utils.write_compressed_cfradial(radar, output_filename)
    if profile_memory:
        _profile_memory("after-write", radar=radar, file_name=os.path.basename(radar_file_name))
    if debug:
        elapsed = time.time() - tw
        print(f"  [{'write_cfradial':<22}] {elapsed:7.3f} s")

    if do_return:
        return radar
    return None
