"""
Codes for correcting Doppler velocity.

@title: velocity
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 19/03/2020

.. autosummary::
    :toctree: generated/

    _check_nyquist_velocity
    unravel
"""
import pyart
import netCDF4
import numpy as np


def unravel(radar, gatefilter, vel_name='VEL', dbz_name='DBZ'):
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
    unfvel = unravel.unravel_3D_pyart(radar,
                                      vel_name,
                                      dbz_name,
                                      gatefilter=gatefilter,
                                      alpha=0.8,
                                      nyquist_velocity=nyquist,
                                      strategy='long_range')

    vel_meta = pyart.config.get_metadata('velocity')
    vel_meta['data'] = np.ma.masked_where(gatefilter.gate_excluded, unfvel).astype(np.float32)
    vel_meta['_Least_significant_digit'] = 2
    vel_meta['_FillValue'] = np.NaN
    vel_meta['comment'] = 'UNRAVEL algorithm.'
    vel_meta['long_name'] = 'Doppler radial velocity of scatterers away from instrument'
    vel_meta['standard_name'] = 'radial_velocity_of_scatterers_away_from_instrument'
    vel_meta['units'] = 'm s-1'

    try:
        vel_meta.pop('standard_name')
    except Exception:
        pass

    return vel_meta
