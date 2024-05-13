"""
OPOL Level 1b driver.

@project: OCEANPol
@title: production
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology and Monash University
@date: 09/06/2023

.. autosummary::
    :toctree: generated/

    _mkdir
    process_and_save
    production_line
"""
# Python Standard Library
import os
import re
import copy
import uuid
import datetime
import warnings

# Other Libraries
import pyart
import cftime
import numpy as np

# Custom modules.
from . import attenuation
from . import filtering
from . import hydrometeors
from . import phase
from . import radar_codes


def _mkdir(dir):
    """
    Make directory. Might seem redundant but you might have concurrency issue
    when dealing with multiprocessing.
    """
    if os.path.exists(dir):
        return None

    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

    return None


def process_and_save(radar_file_name, outpath, do_dealiasing=True, use_unravel=True):
    """
    Call processing function and write data.

    Parameters:
    ===========
    radar_file_name: str
        Name of the input radar file.
    outpath: str
        Path for saving output data.
    do_dealiasing: bool
        Dealias velocity.
    use_unravel: bool
        Use of UNRAVEL for dealiasing the velocity
    """
    today = datetime.datetime.utcnow()

    voyage_directory = radar_file_name.split("/")[-3]
    datestr = re.findall("[0-9]{8}", os.path.basename(radar_file_name))[0]
    # Create output directories.
    _mkdir(outpath)
    outpath_ppi = os.path.join(outpath, "ppi")
    _mkdir(outpath_ppi)
    outpath_ppi = os.path.join(outpath_ppi, datestr)
    _mkdir(outpath_ppi)

    # Generate output file name.
    outfilename = os.path.basename(radar_file_name).replace(".hdf", ".cfradial.nc")
    outfilename = os.path.join(outpath_ppi, outfilename)
    # Check if output file already exists.
    if os.path.isfile(outfilename):
        print(f"Output file {outfilename} already exists.")
        return None

    # Business start here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        radar = production_line(radar_file_name, do_dealiasing=do_dealiasing, use_unravel=use_unravel)
        if radar is None:
            print(f"{radar_file_name} has not been processed. Check logs.")
            return None

    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    radar_end_date = cftime.num2pydate(radar.time["data"][-1], radar.time["units"])

    # Lat/lon informations
    latitude = radar.gate_latitude["data"]
    longitude = radar.gate_longitude["data"]
    maxlon = longitude.max()
    minlon = longitude.min()
    maxlat = latitude.max()
    minlat = latitude.min()
    origin_altitude = radar.altitude["data"][0]
    origin_latitude = radar.latitude["data"][0]
    origin_longitude = radar.longitude["data"][0]

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
        "origin_altitude": origin_altitude,
        "origin_latitude": origin_latitude,
        "origin_longitude": origin_longitude,
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

    # Write results
    pyart.io.write_cfradial(outfilename, radar, format="NETCDF4")

    return None


def production_line(radar_file_name, do_dealiasing=True, use_unravel=True):
    """
    Production line for correcting and estimating OPOL data radar parameters.
    The naming convention for these parameters is assumed to be DBZ, ZDR, VEL,
    PHIDP, KDP, SNR, RHOHV, and NCP. KDP, NCP, and SNR are optional and can be
    recalculated.

    Parameters:
    ===========
    radar_file_name: str
        Name of the input radar file.
    do_dealiasing: bool
        Dealias velocity.
    use_unravel: bool
        Use of UNRAVEL for dealiasing the velocity

    Returns:
    ========
    radar: Object
        Py-ART radar structure.

    PLAN:
    =====
    01/ Read input radar file.
    02/ Check if radar file OK (no problem with azimuth and reflectivity).
    03/ Get radar date.
    04/ Check if NCP field exists (creating a fake one if it doesn't)
    05/ Check if RHOHV field exists (creating a fake one if it doesn't)
    06/ Compute SNR and temperature using radiosoundings.
    07/ Correct RHOHV using Ryzhkov algorithm.
    08/ Create gatefilter (remove noise and incorrect data).
    09/ Correct ZDR using Ryzhkov algorithm.
    10/ Compute Giangrande's PHIDP using pyart.
    11/ Unfold velocity.
    12/ Compute attenuation for ZH
    13/ Compute attenuation for ZDR
    16/ Removing fake/temporary fieds.
    17/ Rename fields to pyart standard names.
    """
    FIELDS_NAMES = [
        ("VEL", "velocity"),
        ("VEL_UNFOLDED", "corrected_velocity"),
        ("TH", "total_power"),
        ("DBZ", "corrected_reflectivity"),
        ("DBZH", "corrected_reflectivity"),
        ("DBZ_CORR_ORIG", "corrected_reflectivity_edge"),
        ("RHOHV_CORR", "cross_correlation_ratio"),
        ("ZDR", "differential_reflectivity"),
        ("ZDR_CORR", "corrected_differential_reflectivity"),
        ("PHIDP", "differential_phase"),
        ("PHIDP_BRINGI", "bringi_differential_phase"),
        ("PHIDP_GG", "giangrande_differential_phase"),
        ("PHIDP_VAL", "corrected_differential_phase"),
        ("KDP", "specific_differential_phase"),
        ("KDP_BRINGI", "bringi_specific_differential_phase"),
        ("KDP_GG", "giangrande_specific_differential_phase"),
        ("KDP_VAL", "corrected_specific_differential_phase"),
        ("WIDTH", "spectrum_width"),
        ("SNR", "signal_to_noise_ratio"),
        ("NCP", "normalized_coherent_power"),
        ("DBZV", "reflectivity_v"),
        ("WRADV", "spectrum_width_v"),
        ("SNRV", "signal_to_noise_ratio_v"),
        ("SQIV", "normalized_coherent_power_v"),
    ]

    radar = radar_codes.read_radar(radar_file_name)
    dbz_name = radar_codes.get_corr_refl(radar)
    # Correct OceanPOL offset.
    if radar.nsweeps < 10:
        return None

    # Correct time units.
    if "since " not in radar.time["units"]:
        # Signal processing forgot (sometime) a space in generating the unit.
        radar.time["units"] = radar.time["units"].replace("since", "since ")
    radar_start_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])

    try:
        radar.fields["TH"]
    except KeyError:
        radar.add_field("TH", copy.deepcopy(radar.fields[dbz_name]))

    # ZDR and DBZ calibration factor for OCEANPol before YMC experiment (included).
    if radar_start_date.year <= 2020:
        radar.fields["ZDR"]["data"] += 0.7
        radar.fields[dbz_name]["data"] += 3.5

    fake_ncp = False
    if "NCP" not in radar.fields.keys():
        radar.add_field("NCP", radar.fields["SQI"])
        fake_ncp = True

    try:
        _ = radar.fields['VEL']
    except KeyError:
        do_dealiasing = False

    # Correct data type manually
    try:
        radar.longitude["data"] = np.ma.masked_invalid(radar.longitude["data"].astype(np.float32))
        radar.latitude["data"] = np.ma.masked_invalid(radar.latitude["data"].astype(np.float32))
        radar.altitude["data"] = np.ma.masked_invalid(radar.altitude["data"].astype(np.int32))
    except Exception:
        pass

    # Check if radar reflecitivity field is correct.
    if not radar_codes.check_reflectivity(radar, dbz_name):
        raise TypeError(f"Reflectivity field is empty in {radar_file_name}.")

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like("RHOHV", "RHOHV_CORR", rho_corr, replace_existing=True)

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like("ZDR", "ZDR_CORR", corr_zdr, replace_existing=True)

    # Temperature
    height, temperature = radar_codes.temperature_profile(radar)
    radar.add_field("temperature", temperature, replace_existing=True)
    radar.add_field("height", height, replace_existing=True)

    # GateFilter
    gatefilter = filtering.do_gatefilter_opol(radar, refl_name=dbz_name, rhohv_name="RHOHV_CORR", zdr_name="ZDR")
    radar.fields[dbz_name]["data"][gatefilter.gate_excluded] = np.NaN
    radar.fields["ZDR_CORR"]["data"][gatefilter.gate_excluded] = np.NaN
    # radar.add_field("air_echo_classification", echoclass, replace_existing=True)

    phidp_bringi, kdp_bringi = phase.phidp_bringi(radar, gatefilter, refl_field=dbz_name)
    radar.add_field("PHIDP_BRINGI", phidp_bringi)
    radar.add_field("KDP_BRINGI", kdp_bringi)
    phidp, kdp = phase.phido(radar, gatefilter, dbz_name)
    radar.add_field("PHIDP_PHIDO", phidp)
    radar.add_field("KDP_PHIDO", kdp)
    phidp_field_name = "PHIDP_PHIDO"
    kdp_field_name = "KDP_PHIDO"

    # Unfold VELOCITY
    if do_dealiasing:
        vdop_unfold = radar_codes.unravel(radar, gatefilter)
        radar.add_field("VEL_UNFOLDED", vdop_unfold, replace_existing=True)

    # Correct attenuation ZH and ZDR and hardcode gatefilter
    atten = attenuation.correct_attenuation_zh_pyart(radar, refl_field=dbz_name, phidp_field=phidp_field_name)
    radar.add_field("path_integrated_attenuation", atten)
    radar.fields[dbz_name]["comment"] = (
        "Attenuation has not been corrected. Please consider added the 'path_integrated_attenuation' "
        "to this field to take into account the attenuation."
        )

    zdr_corr = attenuation.correct_attenuation_zdr(radar, gatefilter, phidp_name=phidp_field_name)
    radar.add_field("path_integrated_differential_attenuation", zdr_corr)

    # Hydrometeors classification
    hydro_class = hydrometeors.hydrometeor_classification(
        radar, gatefilter, refl_name=dbz_name, kdp_name="KDP_BRINGI", zdr_name="ZDR_CORR"
    )

    radar.add_field("radar_echo_classification", hydro_class, replace_existing=True)

    # Rainfall rate
    rainfall = hydrometeors.rainfall_rate(
        radar, gatefilter, kdp_name=kdp_field_name, refl_name=dbz_name, zdr_name="ZDR_CORR"
    )
    radar.add_field("radar_estimated_rain_rate", rainfall)

    # Remove obsolete fields:
    if fake_ncp:
        _ = radar.fields.pop("NCP")

    for obsolete_key in ["Refl", "temperature", "PHI_UNF", "PHI_CORR", "height", "TV", "RHOHV"]:
        try:
            radar.fields.pop(obsolete_key)
        except KeyError:
            continue

    radar_codes.set_significant_digits(radar)
    # Change the temporary working name of fields to the one define by the user.
    for old_key, new_key in FIELDS_NAMES:
        try:
            radar.add_field(new_key, radar.fields.pop(old_key), replace_existing=True)
        except KeyError:
            continue

    # Correct the standard_name metadata:
    radar_codes.correct_standard_name(radar)
    # ACDD-1.3 compliant metadata:
    radar_codes.coverage_content_type(radar)

    return radar
