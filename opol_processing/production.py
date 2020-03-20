"""
OPOL Level 1b main production line. These are the drivers function.

@title: production
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@copyright: Valentin Louf (2017-2020)
@institution: Bureau of Meteorology and Monash University
@date: 19/03/2020

.. autosummary::
    :toctree: generated/

    _mkdir
    process_and_save
    production_line
"""
# Python Standard Library
import gc
import os
import time
import uuid
import datetime
import traceback
import warnings

# Other Libraries
import pyart
import netCDF4
import numpy as np

# Custom modules.
from . import attenuation
from . import filtering
from . import hydrometeors
from . import phase
from . import radar_codes
from . import velocity


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


def process_and_save(radar_file_name,
                     outpath,
                     do_dealiasing=True,
                     use_unravel=True,
                     sound_dir=None):
    """
    Call processing function and write data.

    Parameters:
    ===========
    radar_file_name: str
        Name of the input radar file.
    outpath: str
        Path for saving output data.
    sound_dir: str
        Path to radiosoundings directory.
    do_dealiasing: bool
        Dealias velocity.
    use_unravel: bool
        Use of UNRAVEL for dealiasing the velocity
    """
    today = datetime.datetime.utcnow()

    # Create directories.
    _mkdir(outpath)
    outpath = os.path.join(outpath, "v{}".format(today.strftime('%Y')))
    _mkdir(outpath)
    outpath_ppi = os.path.join(outpath, 'ppi')
    _mkdir(outpath_ppi)
    tick = time.time()

    # Business start here.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        radar = production_line(radar_file_name,
                                do_dealiasing=do_dealiasing,
                                use_unravel=use_unravel,
                                sound_dir=sound_dir)
    # Business over.

    if radar is None:
        print(f'{radar_file_name} has not been processed. Check logs.')
        return None

    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
    radar_end_date = netCDF4.num2date(radar.time['data'][-1], radar.time['units'])
    outpath_ppi = os.path.join(outpath_ppi, str(radar_start_date.year))
    _mkdir(outpath_ppi)
    outpath_ppi = os.path.join(outpath_ppi, radar_start_date.strftime('%Y%m%d'))
    _mkdir(outpath_ppi)

    # Generate output file name.
    outfilename = "rvi6opolppi.b1.{}00.nc".format(radar_start_date.strftime("%Y%m%d.%H%M"))
    outfilename = os.path.join(outpath_ppi, outfilename)

    # Check if output file already exists.
    if os.path.isfile(outfilename):
        print(f"Output file {outfilename} already exists.")
        return None

    # Lat/lon informations
    latitude = radar.gate_latitude['data']
    longitude = radar.gate_longitude['data']
    maxlon = longitude.max()
    minlon = longitude.min()
    maxlat = latitude.max()
    minlat = latitude.min()
    origin_altitude = radar.altitude['data'][0]
    origin_latitude = radar.latitude['data'][0]
    origin_longitude = radar.longitude['data'][0]

    unique_id = str(uuid.uuid4())
    metadata = {'Conventions': "CF-1.6, ACDD-1.3",
                'acknowledgement': 'This work has been supported by the U.S. Department of Energy Atmospheric Systems Research Program through the grant DE-SC0014063. Data may be freely distributed.',
                'country': 'Australia',
                'creator_email': 'valentin.louf@bom.gov.au',
                'creator_name': 'Valentin Louf',
                'creator_url': 'github.com/vlouf',
                'date_created': today.isoformat(),
                "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
                'geospatial_lat_max': f'{maxlat:0.6}',
                'geospatial_lat_min': f'{minlat:0.6}',
                'geospatial_lat_units': "degrees_north",
                'geospatial_lon_max': f'{maxlon:0.6}',
                'geospatial_lon_min': f'{minlon:0.6}',
                'geospatial_lon_units': "degrees_east",
                'history': "created by Valentin Louf on gadi.nci.org.au at " + today.isoformat() + " using Py-ART",
                'id': unique_id,
                'institution': 'Bureau of Meteorology',
                'instrument': 'radar',
                'instrument_name': 'OPOL',
                'instrument_type': 'radar',
                'keywords': 'radar, Doppler, dual-polarization, shipborne',
                'licence': "Freely Distributed",
                'naming_authority': 'au.org.nci',
                'origin_altitude': origin_altitude,
                'origin_latitude': origin_latitude,
                'origin_longitude': origin_longitude,
                'platform_is_mobile': 'true',
                'processing_level': 'b1',
                'project': "OPOL",
                'publisher_name': "NCI",
                'publisher_url': "nci.gov.au",
                'product_version': f"v{today.year}.{today.month:02}",
                'site_name': 'RV Investigator',
                'source': 'radar',
                'standard_name_vocabulary': 'CF Standard Name Table v71',
                'summary': "Volumetric scan from OPOL dual-polarization Doppler radar (RV Investigator)",
                'time_coverage_start': radar_start_date.isoformat(),
                'time_coverage_end': radar_end_date.isoformat(),
                'time_coverage_duration': "P06M",
                'time_coverage_resolution': "PT06M",
                'title': "radar PPI volume from OPOL",
                'uuid': unique_id,
                'version': radar.metadata['version']}
    radar.metadata = metadata

    # Write results
    pyart.io.write_cfradial(outfilename, radar, format='NETCDF4')
    print('%s processed in  %0.2fs.' % (os.path.basename(radar_file_name), (time.time() - tick)))

    # Free memory
    del radar
    gc.collect()

    return None


def production_line(radar_file_name,
                    do_dealiasing=True,
                    use_unravel=True,
                    sound_dir=None):
    """
    Production line for correcting and estimating OPOL data radar parameters.
    The naming convention for these parameters is assumed to be DBZ, ZDR, VEL,
    PHIDP, KDP, SNR, RHOHV, and NCP. KDP, NCP, and SNR are optional and can be
    recalculated.

    Parameters:
    ===========
    radar_file_name: str
        Name of the input radar file.
    sound_dir: str
        Path to radiosounding directory.
    is_OPOL: bool
        Name of radar (only OPOL will change something).
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
    FIELDS_NAMES = [('VEL', 'velocity'),
                    ('VEL_UNFOLDED', 'corrected_velocity'),
                    ('DBZ', 'total_power'),
                    ('DBZ_CORR', 'corrected_reflectivity'),
                    ('DBZ_CORR_ORIG', 'corrected_reflectivity_edge'),
                    ('RHOHV_CORR', 'cross_correlation_ratio'),
                    ('ZDR', 'differential_reflectivity'),
                    ('ZDR_CORR_ATTEN', 'corrected_differential_reflectivity'),
                    ('PHIDP', 'differential_phase'),
                    ('PHIDP_BRINGI', 'bringi_differential_phase'),
                    ('PHIDP_GG', 'giangrande_differential_phase'),
                    ('PHIDP_VAL', 'corrected_differential_phase'),
                    ('KDP', 'specific_differential_phase'),
                    ('KDP_BRINGI', 'bringi_specific_differential_phase'),
                    ('KDP_GG', 'giangrande_specific_differential_phase'),
                    ('KDP_VAL', 'corrected_specific_differential_phase'),
                    ('WIDTH', 'spectrum_width'),
                    ('SNR', 'signal_to_noise_ratio'),
                    ('NCP', 'normalized_coherent_power'),
                    ('DBZV', 'reflectivity_v'),
                    ('WRADV', 'spectrum_width_v'),
                    ('SNRV', 'signal_to_noise_ratio_v'),
                    ('SQIV', 'normalized_coherent_power_v')]

    # List of keys that we'll keep in the output radar dataset.
    OUTPUT_RADAR_FLD = ['corrected_differential_phase',
                        'corrected_differential_reflectivity',
                        'corrected_reflectivity',
                        'corrected_specific_differential_phase',
                        'corrected_velocity',
                        'cross_correlation_ratio',
                        'differential_phase',
                        'differential_reflectivity',
                        'radar_echo_classification',
                        'radar_estimated_rain_rate',
                        'signal_to_noise_ratio',
                        'spectrum_width',
                        'total_power',
                        'velocity']

    radar = radar_codes.read_radar(radar_file_name)

    # Correct data type manually
    try:
        radar.longitude['data'] = np.ma.masked_invalid(radar.longitude['data'].astype(np.float32))
        radar.latitude['data'] = np.ma.masked_invalid(radar.latitude['data'].astype(np.float32))
        radar.altitude['data'] = np.ma.masked_invalid(radar.altitude['data'].astype(np.int32))
    except Exception:
        pass

    # Check if radar reflecitivity field is correct.
    if not radar_codes.check_reflectivity(radar):
        raise TypeError(f"Reflectivity field is empty in {radar_file_name}.")

    # Getting radar's date and time.
    radar_start_date = netCDF4.num2date(radar.time['data'][0], radar.time['units'].replace("since", "since "))
    radar.time['units'] = radar.time['units'].replace("since", "since ")

    # Correct RHOHV
    rho_corr = radar_codes.correct_rhohv(radar)
    radar.add_field_like('RHOHV', 'RHOHV_CORR', rho_corr, replace_existing=True)

    # Correct ZDR
    corr_zdr = radar_codes.correct_zdr(radar)
    radar.add_field_like('ZDR', 'ZDR_CORR', corr_zdr, replace_existing=True)

    # Temperature
    if sound_dir is not None:        
        try:
            radiosonde_fname = radar_codes.get_radiosoundings(sound_dir, radar_start_date)
            height, temperature = radar_codes.snr_and_sounding(radar, radiosonde_fname)
            radar.add_field('temperature', temperature, replace_existing=True)
            radar.add_field('height', height, replace_existing=True)
            has_temperature = True
        except ValueError:
            has_temperature = False
            pass

    # GateFilter
    gatefilter, echoclass = filtering.do_gatefilter_opol(radar,
                                                         refl_name='DBZ',
                                                         phidp_name="PHIDP",
                                                         rhohv_name='RHOHV_CORR',
                                                         zdr_name="ZDR")

    radar.add_field('radar_echo_classification', echoclass)

    phidp, kdp = phase.phidp_giangrande(radar, gatefilter)
    radar.add_field('PHIDP_VAL', phidp)
    radar.add_field('KDP_VAL', kdp)
    kdp_field_name = 'KDP_VAL'
    phidp_field_name = 'PHIDP_VAL'

    # Unfold VELOCITY
    if do_dealiasing:
        vdop_unfold = velocity.unravel(radar, gatefilter)
        radar.add_field('VEL_UNFOLDED', vdop_unfold, replace_existing=True)

    # Correct attenuation ZH and ZDR and hardcode gatefilter
    zh_corr = attenuation.correct_attenuation_zh_pyart(radar, gatefilter, phidp_field=phidp_field_name)
    radar.add_field_like('DBZ', 'DBZ_CORR', zh_corr)

    zdr_corr = attenuation.correct_attenuation_zdr(radar, gatefilter)
    radar.add_field('ZDR_CORR_ATTEN', zdr_corr)

    # Remove obsolete fields:
    for obsolete_key in ["Refl", "PHI_UNF", "PHI_CORR", "height", 'TH', 'TV', 'ZDR_CORR', 'RHOHV']:
        try:
            radar.fields.pop(obsolete_key)
        except KeyError:
            continue

    # Change the temporary working name of fields to the one define by the user.
    for old_key, new_key in FIELDS_NAMES:
        try:
            radar.add_field(new_key, radar.fields.pop(old_key), replace_existing=True)
        except KeyError:
            continue

    # Delete working variables.
    # for k in list(radar.fields.keys()):
    #     if k not in OUTPUT_RADAR_FLD:
    #         radar.fields.pop(k)

    # Correct the standard_name metadata:
    radar_codes.correct_standard_name(radar)
    # ACDD-1.3 compliant metadata:
    radar_codes.coverage_content_type(radar)

    return radar
