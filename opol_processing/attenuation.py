"""
Codes for correcting and estimating attenuation on ZH and ZDR.

@title: attenuation
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 24/06/2021

.. autosummary::
    :toctree: generated/

    correct_gaseous_attenuation
    correct_attenuation_zdr
    correct_attenuation_zh_pyart
"""
# Other Libraries
import pyart
import numpy as np

from scipy.integrate import cumtrapz


def correct_gaseous_attenuation(radar):
    """
    Adjust for gaseous attenuation from Doviak and Zrnic note tempgas is in dB,
    elevation in degrees and r in km! Equation valid only for elevation < 10 deg
    and r < 200 km. Right now only doing for tropic atmosphere according to
    Doviak and Zrnic's book; for C-band we can increase the attenuation by a
    FACTOR OF 1.2.

    Note: Doviak and Zrnic's fit is for standard atmosphere (atten by oxygen
    and water vapor. Their equation is at S-band. The factor of 1.2  is a good
    approximation for C-band! Water vapor atten may have to increased in tropics
    or over ocean.
    """
    r = radar.range["data"] / 1000
    theta = radar.elevation["data"]

    R, TH = np.meshgrid(r, theta)

    atten_gas = np.zeros(TH.shape)
    pos = TH <= 10

    tempgas1 = 0.4 + 3.45 * np.exp(-TH / 1.8)
    tempgas2 = 27.8 + 154 * np.exp(-TH / 2.2)
    atten_gas = 1.2 * tempgas1 * (1 - np.exp(-R / tempgas2))  # 1.2 factor for C-band 1.0 for S-band.
    atten_gas[~pos] = 0

    return atten_gas


def correct_attenuation_zdr(radar, gatefilter, zdr_name="ZDR_CORR", phidp_name="PHIDP_VAL", alpha=0.016):
    """
    Correct attenuation on differential reflectivity. KDP_GG has been
    cleaned of noise, that's why we use it.

    V. N. Bringi, T. D. Keenan and V. Chandrasekar, "Correcting C-band radar
    reflectivity and differential reflectivity data for rain attenuation: a
    self-consistent method with constraints," in IEEE Transactions on Geoscience
    and Remote Sensing, vol. 39, no. 9, pp. 1906-1915, Sept. 2001.
    doi: 10.1109/36.951081

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    zdr_name: str
        Differential reflectivity field name.
    gatefilter:
        Filter excluding non meteorological echoes.
    kdp_name: str
        KDP field name.

    Returns:
    ========
    zdr_corr: array
        Attenuation corrected differential reflectivity.
    """
    # zdr = radar.fields[zdr_name]["data"].copy()
    phi = radar.fields[phidp_name]["data"].copy()

    atten_corr = 0.016 * phi
    atten_corr[gatefilter.gate_excluded] = np.NaN
    atten_corr = np.ma.masked_invalid(atten_corr)
    np.ma.set_fill_value(atten_corr, np.NaN)
    # Z-PHI coefficient from Bringi et al. 2001
    atten_zdr_meta = pyart.config.get_metadata("path_integrateddifferential_attenuation")
    atten_zdr_meta["description"] = "Attenuation corrected differential reflectivity using Bringi et al. 2001."
    atten_zdr_meta["_FillValue"] = np.NaN
    atten_zdr_meta["_Least_significant_digit"] = 2
    atten_zdr_meta["data"] = atten_corr.astype(np.float32)

    return atten_zdr_meta


def correct_attenuation_zh_pyart(
    radar, refl_field, ncp_field="NCP", rhv_field="RHOHV_CORR", phidp_field="PHIDP_GG"
):
    """
    Correct attenuation on reflectivity using Py-ART tool. The attenuation from
    atmospheric gases is also corrected.

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    kdp_name: str
        KDP field name.

    Returns:
    ========
    atten_meta: dict
        Specific attenuation.
    zh_corr: array
        Attenuation corrected reflectivity.
    """
    # Compute attenuation
    spec_atten, _ = pyart.correct.calculate_attenuation(
        radar, 0, rhv_min=0.3, refl_field=refl_field, ncp_field=ncp_field, rhv_field=rhv_field, phidp_field=phidp_field
    )

    specific_atten = np.ma.masked_invalid(spec_atten["data"])
    r = radar.range["data"] / 1000
    dr = r[2] - r[1]

    na, nr = radar.fields[refl_field]["data"].shape
    atten_meta = pyart.config.get_metadata("path_integrated_attenuation")    
    attenuation = np.zeros((na, nr))
    attenuation[:, :-1] = 2 * cumtrapz(specific_atten, dx=dr)
    atten_meta["data"] = attenuation.astype(np.float32)
    atten_meta["_Least_significant_digit"] = 2 

    return atten_meta
