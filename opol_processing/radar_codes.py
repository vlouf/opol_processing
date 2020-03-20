"""
Codes for correcting and estimating various radar and meteorological parameters.

@title: radar_codes
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 19/03/2020

.. autosummary::
    :toctree: generated/
    
    check_reflectivity
    check_year
    correct_rhohv
    correct_zdr    
    read_radar    
"""
# Python Standard Library
import os
import re
import glob
import time
import fnmatch
import datetime

# Other Libraries
import pyart
import netCDF4
import numpy as np


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


def check_reflectivity(radar, refl_field_name='DBZ'):
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
    dbz = radar.fields[refl_field_name]['data']

    if np.ma.isMaskedArray(dbz):
        if dbz.count() == 0:
            # Reflectivity field is empty.
            return False

    return True


def correct_rhohv(radar, rhohv_name='RHOHV', snr_name='SNR'):
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
    rhohv = radar.fields[rhohv_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()

    natural_snr = 10**(0.1 * snr)
    natural_snr = natural_snr.filled(-9999)
    rho_corr = rhohv * (1 + 1 / natural_snr)

    # Not allowing the corrected RHOHV to be lower than the raw rhohv
    rho_corr[np.isnan(rho_corr) | (rho_corr < 0) | (rho_corr > 1)] = 1
    try:
        rho_corr = rho_corr.filled(1)
    except Exception:
        pass

    return rho_corr


def correct_standard_name(radar):
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
        radar.range.pop('standard_name')
        radar.azimuth.pop('standard_name')
        radar.elevation.pop('standard_name')
    except Exception:
        pass

    try:
        radar.sweep_number.pop('standard_name')
        radar.fixed_angle.pop('standard_name')
        radar.sweep_mode.pop('standard_name')
    except Exception:
        pass

    good_keys = ['corrected_reflectivity', 'total_power', 'radar_estimated_rain_rate', 'corrected_velocity']
    for k in radar.fields.keys():
        if k not in good_keys:
            try:
                radar.fields[k].pop('standard_name')
            except Exception:
                continue

    try:
        radar.fields['velocity']['standard_name'] = 'radial_velocity_of_scatterers_away_from_instrument'
        radar.fields['velocity']['long_name'] = 'Doppler radial velocity of scatterers away from instrument'
    except KeyError:
        pass

    radar.latitude['standard_name'] = 'latitude'
    radar.longitude['standard_name'] = 'longitude'
    radar.altitude['standard_name'] = 'altitude'

    return None


def correct_zdr(radar, zdr_name='ZDR', snr_name='SNR'):
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
    zdr = radar.fields[zdr_name]['data'].copy()
    snr = radar.fields[snr_name]['data'].copy()
    alpha = 1.48
    natural_zdr = 10 ** (0.1 * zdr)
    natural_snr = 10 ** (0.1 * snr)
    corr_zdr = 10 * np.log10((alpha * natural_snr * natural_zdr) / (alpha * natural_snr + alpha - natural_zdr))

    return corr_zdr


def coverage_content_type(radar):
    """
    Adding metadata for compatibility with ACDD-1.3

    Parameter:
    ==========
    radar: Radar object
        Py-ART data structure.
    """
    radar.range['coverage_content_type'] = 'coordinate'
    radar.azimuth['coverage_content_type'] = 'coordinate'
    radar.elevation['coverage_content_type'] = 'coordinate'
    radar.latitude['coverage_content_type'] = 'coordinate'
    radar.longitude['coverage_content_type'] = 'coordinate'
    radar.altitude['coverage_content_type'] = 'coordinate'

    radar.sweep_number['coverage_content_type'] = 'auxiliaryInformation'
    radar.fixed_angle['coverage_content_type'] = 'auxiliaryInformation'
    radar.sweep_mode['coverage_content_type'] = 'auxiliaryInformation'

    for k in radar.fields.keys():
        if k == 'radar_echo_classification':
            radar.fields[k]['coverage_content_type'] = 'thematicClassification'
        elif k in ['normalized_coherent_power', 'normalized_coherent_power_v']:
            radar.fields[k]['coverage_content_type'] = 'qualityInformation'
        else:
            radar.fields[k]['coverage_content_type'] = 'physicalMeasurement'

    return None


def get_radiosoundings(sound_dir, radar_start_date):
    """
    Find the radiosoundings
    """
    def _fdate(flist):
        rslt = [None] * len(flist)
        for cnt, f in enumerate(flist):
            try:
                rslt[cnt] = datetime.datetime.strptime(re.findall("[0-9]{8}", f)[0], "%Y%m%d")
            except Exception:
                continue
        return rslt
    # Looking for radiosoundings:
    all_sonde_files = sorted(os.listdir(sound_dir))

    pos = [cnt for cnt, f in enumerate(all_sonde_files) if fnmatch.fnmatch(f, "*" + radar_start_date.strftime("%Y%m%d") + "*")]
    if len(pos) > 0:
        # Looking for the exact date.
        sonde_name = all_sonde_files[pos[0]]
        sonde_name = os.path.join(sound_dir, sonde_name)
    else:
        # Looking for the closest date.
        dtime_none = _fdate(all_sonde_files)
        dtime = [d for d in dtime_none if d is not None]
        closest_date = _nearest(dtime, radar_start_date)
        sonde_name = [e for e in all_sonde_files if closest_date.strftime("%Y%m%d") in e]
        if len(sonde_name) == 0:
            sonde_name = os.path.join(sound_dir, all_sonde_files[-1])
        elif type(sonde_name) is list:
            sonde_name = os.path.join(sound_dir, sonde_name[0])
        else:
            sonde_name = os.path.join(sound_dir, sonde_name)

    return sonde_name


def read_radar(radar_file_name):
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
    radar = pyart.aux_io.read_odim_h5(radar_file_name, file_field_names=False)
    myfields = [('normalized_coherent_power', "NCP"),
                ('reflectivity', "DBZ_CORR_ORIG"),
                ('spectrum_width', "WIDTH"),
                ('total_power', "DBZ"),
                ("differential_reflectivity", "ZDR"),                
                ('velocity', "VEL"),
                ('signal_to_noise_ratio', "SNR"),
                ("cross_correlation_ratio", "RHOHV"),
                ("differential_phase", "PHIDP"),
                ]

    for mykey, newkey in myfields:
        try:
            radar.add_field(newkey, radar.fields.pop(mykey))
        except Exception:
            continue

    radar.fields['VEL']['units'] = "m s-1"
    return radar


def snr_and_sounding(radar, sonde_name, temp_field_name="temp"):
    """
    Compute the signal-to-noise ratio as well as interpolating the radiosounding
    temperature on to the radar grid. The function looks for the radiosoundings
    that happened at the closest time from the radar. There is no time
    difference limit.
    Parameters:
    ===========
        radar:
        sonde_name: str
            Path to the radiosoundings.
        refl_field_name: str
            Name of the reflectivity field.
    Returns:
    ========
        z_dict: dict
            Altitude in m, interpolated at each radar gates.
        temp_info_dict: dict
            Temperature in Celsius, interpolated at each radar gates.
        snr: dict
            Signal to noise ratio.
    """
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    # Altitude hack.
    true_alt = radar.altitude['data'].copy()
    radar.altitude['data'] = np.array([0])

    # print("Reading radiosounding %s" % (sonde_name))
    interp_sonde = netCDF4.Dataset(sonde_name)
    temperatures = interp_sonde.variables[temp_field_name][:]
    temperatures[(temperatures < -100) | (temperatures > 100)] = np.NaN
    try:
        temperatures = temperatures.filled(np.NaN)
    except AttributeError:
        pass
    # times = interp_sonde.variables['time'][:]
    # heights = interp_sonde.variables['height'][:]

    # Height profile corresponding to radar.
    my_profile = pyart.retrieve.fetch_radar_time_profile(interp_sonde, radar)

    # CPOL altitude is 50 m.
    good_altitude = my_profile['height'] >= 0
    # Getting the temperature
    z_dict, temp_dict = pyart.retrieve.map_profile_to_gates(temperatures[good_altitude],
                                                            my_profile['height'][good_altitude],
                                                            radar)

    temp_info_dict = {'data': temp_dict['data'],
                      'long_name': 'Sounding temperature at gate',
                      'standard_name': 'temperature',
                      'valid_min': -100, 'valid_max': 100,
                      'units': 'degrees Celsius',
                      'comment': 'Radiosounding date: %s' % (radar_start_date.strftime("%Y/%m/%d"))}

    # Altitude hack
    radar.altitude['data'] = true_alt

    return z_dict, temp_info_dict
