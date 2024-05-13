"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 14/05/2021

.. autosummary::
    :toctree: generated/

    check_reflectivity
    check_year
    correct_rhohv
    correct_zdr
    read_radar
    read_era5_temperature
    set_significant_digits
    temperature_profile
    unravel
"""
# Python Standard Library
import os
import glob

# Other Libraries
import pyart
import cftime
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


def get_corr_refl(radar):
    for dbz_key in ["DBZH", "DBZ", "TH", None]:
        if dbz_key in radar.fields.keys():
            break

    if dbz_key is None:
        raise ValueError(f"Reflectivity not found.")

    radar.add_field("DBZH_CLEAN", radar.fields[dbz_key])

    return "DBZH_CLEAN"


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
        radar = pyart.aux_io.read_odim_h5(radar_file_name, file_field_names=True)
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

    try:
        radar.fields["VEL"]["units"] = "m s-1"
    except Exception:
        pass

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
    era5_root = "/g/data/rt52/era5/pressure-levels/reanalysis/"
    # Build file paths
    month_str = date.month
    year_str = date.year
    era5_file = glob.glob(f"{era5_root}/t/{year_str}/t_era5_oper_pl_{year_str}{month_str:02}*.nc")[0]

    if not os.path.isfile(era5_file):
        raise FileNotFoundError(f"{era5_file} not found.")

    # Get temperature
    dset = xr.open_dataset(era5_file)
    nset = dset.sel(longitude=longitude, latitude=latitude, time=date, method="nearest")
    temperature = nset.t.values
    level = nset.level.values
    z = -2494.3 / 0.218 * np.log(level / 1013.15)

    return z, temperature


def set_significant_digits(radar) -> None:
    """
    Set _Least_significant_digit netcdf attribute.
    """
    fieldnames = [
        ("VEL", 2),
        ("VEL_UNFOLDED", 2),
        ("DBZ", 2),
        ("DBZ_CORR", 2),
        ("DBZ_CORR_ORIG", 2),
        ("RHOHV_CORR", 2),
        ("ZDR", 2),
        ("ZDR_CORR_ATTEN", 2),
        ("PHIDP", 4),
        ("PHIDP_BRINGI", 4),
        ("PHIDP_GG", 4),
        ("PHIDP_VAL", 4),
        ("KDP", 4),
        ("KDP_BRINGI", 4),
        ("KDP_GG", 4),
        ("KDP_VAL", 4),
        ("WIDTH", 4),
        ("SNR", 2),
        ("NCP", 2),
        ("DBZV", 2),
        ("WRADV", 2),
        ("SNRV", 2),
        ("SQIV", 2),
    ]
    # Change the temporary working name of fields to the one define by the user.
    for key, value in fieldnames:
        try:
            radar.fields[key]["_Least_significant_digit"] = value
        except KeyError:
            continue

    return None


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
    temp_dict["data"] = temp_dict["data"].astype(np.float32)

    temp_info_dict = {
        "data": temp_dict["data"],  # Switch to celsius.
        "long_name": "Sounding temperature at gate",
        "standard_name": "temperature",
        "valid_min": -100,
        "valid_max": 100,
        "units": "degrees Celsius",
        "_Least_significant_digit": 1,
        "comment": "ERA5 data date: %s" % (dtime.strftime("%Y/%m/%d")),
    }

    return z_dict, temp_info_dict


def unravel(radar, gatefilter, vel_name="VEL", dbz_name="DBZ"):
    """
    Unfold Doppler velocity using Py-ART region based algorithm. Automatically
    searches for a folding-corrected velocity field.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Filter excluding non meteorological echoes.
    vel_name: str
        Name of the (original) Doppler velocity field.
    dbz_name: str
        Name of the reflecitivity field.

    Returns:
    ========
    vel_meta: dict
        Unfolded Doppler velocity.
    """
    import unravel

    nyquist = 13.3
    unfvel = unravel.unravel_3D_pyart(
        radar, vel_name, dbz_name, gatefilter=gatefilter, alpha=0.8, nyquist_velocity=nyquist, strategy="long_range"
    )

    vel_meta = pyart.config.get_metadata("velocity")
    vel_meta["data"] = np.ma.masked_where(gatefilter.gate_excluded, unfvel).astype(np.float32)
    vel_meta["_Least_significant_digit"] = 2
    vel_meta["_FillValue"] = np.NaN
    vel_meta["comment"] = "UNRAVEL algorithm."
    vel_meta["long_name"] = "Doppler radial velocity of scatterers away from instrument"
    vel_meta["standard_name"] = "radial_velocity_of_scatterers_away_from_instrument"
    vel_meta["units"] = "m s-1"

    try:
        vel_meta.pop("standard_name")
    except Exception:
        pass

    return vel_meta
