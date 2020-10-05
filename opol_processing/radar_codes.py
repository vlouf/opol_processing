"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 04/10/2020

.. autosummary::
    :toctree: generated/

    check_reflectivity
    check_year
    correct_rhohv
    correct_zdr
    read_radar
    read_era5_temperature
    temperature_profile
"""
# Python Standard Library
import os
import re
import glob
import time
import calendar
import datetime

# Other Libraries
import pyart
import cftime
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr


def _nearest(items, pivot):
    """
    Find the nearest item.

    Parameters:
    ===========
        items:
            List of item.
        pivot:
            Item we're looking for.

    Returns:
    ========
        item:
            Value of the nearest item found.
    """
    return min(items, key=lambda x: abs(x - pivot))


def check_reflectivity(radar, refl_field_name: str = "DBZ") -> bool:
    """
    Checking if radar has a proper reflectivity field.  It's a minor problem
    concerning a few days in 2011 for CPOL.

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        refl_field_name: str
            Name of the reflectivity field.

    Return:
    =======
    True if radar has a non-empty reflectivity field.
    """
    dbz = radar.fields[refl_field_name]["data"]

    if np.ma.isMaskedArray(dbz):
        if dbz.count() == 0:
            # Reflectivity field is empty.
            return False

    return True


def correct_rhohv(radar, rhohv_name: str = "RHOHV", snr_name: str = "SNR"):
    """
    Correct cross correlation ratio (RHOHV) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 5)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        rhohv_name: str
            Cross correlation field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        rho_corr: array
            Corrected cross correlation ratio.
    """
    rhohv = radar.fields[rhohv_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()

    natural_snr = 10 ** (0.1 * snr)
    natural_snr = natural_snr.filled(-9999)
    rho_corr = rhohv * (1 + 1 / natural_snr)

    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return rho_corr


def correct_standard_name(radar) -> None:
    """
    'standard_name' is a protected keyword for metadata in the CF conventions.
    To respect the CF conventions we can only use the standard_name field that
    exists in the CF table.

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    try:
        radar.range.pop("standard_name")
        radar.azimuth.pop("standard_name")
        radar.elevation.pop("standard_name")
    except Exception:
        pass

    try:
        radar.sweep_number.pop("standard_name")
        radar.fixed_angle.pop("standard_name")
        radar.sweep_mode.pop("standard_name")
    except Exception:
        pass

    good_keys = ["corrected_reflectivity", "total_power", "radar_estimated_rain_rate", "corrected_velocity"]
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


def correct_zdr(radar, zdr_name: str = "ZDR", snr_name: str = "SNR"):
    """
    Correct differential reflectivity (ZDR) from noise. From the Schuur et al.
    2003 NOAA report (p7 eq 6)

    Parameters:
    ===========
        radar:
            Py-ART radar structure.
        zdr_name: str
            Differential reflectivity field name.
        snr_name: str
            Signal to noise ratio field name.

    Returns:
    ========
        corr_zdr: array
            Corrected differential reflectivity.
    """
    zdr = radar.fields[zdr_name]["data"].copy()
    snr = radar.fields[snr_name]["data"].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def coverage_content_type(radar) -> None:
    """
    Adding metadata for compatibility with ACDD-1.3

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    radar.range["coverage_content_type"] = "coordinate"
    radar.azimuth["coverage_content_type"] = "coordinate"
    radar.elevation["coverage_content_type"] = "coordinate"
    radar.latitude["coverage_content_type"] = "coordinate"
    radar.longitude["coverage_content_type"] = "coordinate"
    radar.altitude["coverage_content_type"] = "coordinate"

    radar.sweep_number["coverage_content_type"] = "auxiliaryInformation"
    radar.fixed_angle["coverage_content_type"] = "auxiliaryInformation"
    radar.sweep_mode["coverage_content_type"] = "auxiliaryInformation"

    for k in radar.fields.keys():
        if k == "radar_echo_classification":
            radar.fields[k]["coverage_content_type"] = "thematicClassification"
        elif k in ["normalized_coherent_power", "normalized_coherent_power_v"]:
            radar.fields[k]["coverage_content_type"] = "qualityInformation"
        else:
            radar.fields[k]["coverage_content_type"] = "physicalMeasurement"

    return None


def read_radar(radar_file_name: str):
    """
    Read the input radar file.

    Parameter:
    ==========
    radar_file_name: str
        Radar file name.

    Return:
    =======
    radar: struct
        Py-ART radar structure.
    """
    # Read the input radar file.
    if ".nc" in radar_file_name:
        radar = pyart.io.read(radar_file_name)
        myfields = [
            ("UH", "DBZ"),
            ("DBZH", "DBZ_CORR_ORIG"),
            ("NCPH", "NCP"),
            ("SNRHC", "SNR"),
            ("VELH", "VEL"),
            ("WIDTHH", "WIDTH"),
        ]
    else:
        radar = pyart.aux_io.read_odim_h5(radar_file_name, file_field_names=False)
        myfields = [
            ("normalized_coherent_power", "NCP"),
            ("reflectivity", "DBZ_CORR_ORIG"),
            ("spectrum_width", "WIDTH"),
            ("total_power", "DBZ"),
            ("differential_reflectivity", "ZDR"),
            ("velocity", "VEL"),
            ("signal_to_noise_ratio", "SNR"),
            ("cross_correlation_ratio", "RHOHV"),
            ("differential_phase", "PHIDP"),
        ]

    for mykey, newkey in myfields:
        try:
            radar.add_field(newkey, radar.fields.pop(mykey))
        except Exception:
            continue

    radar.fields["VEL"]["units"] = "m s-1"
    return radar


def read_era5_temperature(date, longitude: float, latitude: float):
    """
    Extract the temperature profile from ERA5 data for a given date, longitude
    and latitude.

    Parameters:
    ===========
    date: pd.Timestamp
        Date for extraction.
    longitude: float
        Radar longitude
    latitude: float
        Radar latitude.

    Returns:
    ========
    z: ndarray
        Height in m
    temperature: ndarray
        Temperature in K.
    """
    # Generate filename.
    era5_dir = "/g/data/ub4/era5/netcdf/pressure/t/"
    month = date.month
    year = date.year
    lastday = calendar.monthrange(year, month)[1]
    sdate = f"{year}{month:02}01"
    edate = f"{year}{month:02}{lastday}"
    era5_file = os.path.join(era5_dir, str(year), f"t_era5_aus_{sdate}_{edate}.nc")

    if not os.path.isfile(era5_file):
        raise FileNotFoundError(f"{era5_file} not found.")

    # Get temperature
    dset = xr.open_dataset(era5_file)
    nset = dset.sel(longitude=longitude, latitude=latitude, time=date, method="nearest")
    temperature = nset.t.values
    level = nset.level.values
    z = -2494.3 / 0.218 * np.log(level / 1013.15)

    return z, temperature


def temperature_profile(radar):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
    radar:
        Py-ART radar object.
    Returns:
    ========
    z_dict: dict
        Altitude in m, interpolated at each radar gates.
    temp_info_dict: dict
        Temperature in Celsius, interpolated at each radar gates.
    """
    grlat = radar.latitude["data"][0]
    grlon = radar.longitude["data"][0]
    dtime = pd.Timestamp(cftime.num2pydate(radar.time["data"][0], radar.time["units"]))

    geopot_profile, temp_profile = read_era5_temperature(dtime, grlon, grlat)

    # append surface data using lowest level
    geopot_profile = np.append(geopot_profile, [0])
    temp_profile = np.append(temp_profile, temp_profile[-1])

    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temp_profile, geopot_profile, radar)

    temp_info_dict = {
        "data": temp_dict["data"],  # Switch to celsius.
        "long_name": "Sounding temperature at gate",
        "standard_name": "temperature",
        "valid_min": -100,
        "valid_max": 100,
        "units": "degrees Celsius",
        "comment": "ERA5 data date: %s" % (dtime.strftime("%Y/%m/%d")),
    }

    return z_dict, temp_info_dict
