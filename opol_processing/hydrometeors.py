"""
Codes for estimating various parameters related to Hydrometeors.
@title: hydrometeors
@author: Valentin Louf <valentin.louf@monash.edu>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 04/09/2020

.. autosummary::
    :toctree: generated/
    hydrometeor_classification
    rainfall_rate
"""
# Other Libraries
import pyart
import numpy as np

from csu_radartools import csu_liquid_ice_mass, csu_fhc, csu_blended_rain, csu_dsd


def hydrometeor_classification(
    radar,
    gatefilter,
    kdp_name,
    zdr_name,
    refl_name="DBZ_CORR",
    rhohv_name="RHOHV_CORR",
    temperature_name="temperature",
    height_name="height",
):
    """
    Compute hydrometeo classification.
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    kdp_name: str
        KDP field name.
    rhohv_name: str
        RHOHV field name.
    temperature_name: str
        Sounding temperature field name.
    height: str
        Gate height field name.
    Returns:
    ========
    hydro_meta: dict
        Hydrometeor classification.
    """
    refl = radar.fields[refl_name]["data"].copy().filled(np.NaN)
    zdr = radar.fields[zdr_name]["data"].copy().filled(np.NaN)
    try:
        kdp = radar.fields[kdp_name]["data"].copy().filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]["data"].copy()
    rhohv = radar.fields[rhohv_name]["data"]
    try:
        radar_T = radar.fields[temperature_name]["data"]
        use_temperature = True
    except Exception:
        use_temperature = False

    if use_temperature:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=True, band="C", T=radar_T)
    else:
        scores = csu_fhc.csu_fhc_summer(dz=refl, zdr=zdr, rho=rhohv, kdp=kdp, use_temp=False, band="C")

    hydro = np.argmax(scores, axis=0) + 1
    hydro[gatefilter.gate_excluded] = 0
    hydro_data = np.ma.masked_equal(hydro.astype(np.int16), 0)

    the_comments = (
        "1: Drizzle; 2: Rain; 3: Ice Crystals; 4: Aggregates; "
        + "5: Wet Snow; 6: Vertical Ice; 7: LD Graupel; 8: HD Graupel; 9: Hail; 10: Big Drops"
    )

    hydro_meta = {
        "data": hydro_data,
        "units": " ",
        "long_name": "Hydrometeor classification",
        "_FillValue": np.int16(0),
        "standard_name": "Hydrometeor_ID",
        "comments": the_comments,
    }

    return hydro_meta


def rainfall_rate(
    radar,
    gatefilter,
    kdp_name,
    zdr_name,
    refl_name="DBZ_CORR",
    hydro_name="radar_echo_classification",
    temperature_name="temperature",
):
    """
    Rainfall rate algorithm from csu_radartools.
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    zdr_name: str
        ZDR field name.
    kdp_name: str
        KDP field name.
    hydro_name: str
        Hydrometeor classification field name.
    Returns:
    ========
    rainrate: dict
        Rainfall rate.
    """
    dbz = radar.fields[refl_name]["data"].filled(np.NaN)
    zdr = radar.fields[zdr_name]["data"].filled(np.NaN)
    fhc = radar.fields[hydro_name]["data"]
    try:
        kdp = radar.fields[kdp_name]["data"].filled(np.NaN)
    except AttributeError:
        kdp = radar.fields[kdp_name]["data"]

    rain, _ = csu_blended_rain.calc_blended_rain_tropical(dz=dbz, zdr=zdr, kdp=kdp, fhc=fhc, band="C")

    rain[(gatefilter.gate_excluded) | np.isnan(rain) | (rain < 0)] = 0

    try:
        temp = radar.fields[temperature_name]["data"]
        rain[temp < 0] = 0
    except Exception:
        pass

    rainrate = {
        "long_name": "Blended Rainfall Rate",
        "units": "mm h-1",
        "standard_name": "rainfall_rate",
        "_Least_significant_digit": 2,
        "_FillValue": np.NaN,
        "description": "Rainfall rate algorithm based on Thompson et al. 2016.",
        "data": rain.astype(np.float32),
    }

    return rainrate
