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
import glob
import time
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
