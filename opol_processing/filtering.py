"""
Codes for creating and manipulating gate filters. New functions: use of trained
Gaussian Mixture Models to remove noise and clutter from CPOL data before 2009.

@title: filtering
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 22/06/2021

.. autosummary::
    :toctree: generated/

    get_gatefilter_GMM
    do_gatefilter_cpol
"""
import os
import gzip
import pickle

import pyart
import netCDF4
import numpy as np
import pandas as pd


def get_gatefilter_GMM(radar, dbz_name, zdr_name, phidp_name, width_name, rhohv_name):
    """
    Filters non-meteorological signal out using ML classification.

    Parameters:
    -----------
    radar: pyart-Radar object
    dbz_name: str (optional)
        Name of the total_power field.
    zdr_name: str (optional)
        Name of the differential_reflectivity field.
    phidp_name: str (optional)
        Name of the differential_phase field.
    width_name: str (optional)
        Name of the spectrum_width field.
    rhohv_name: str (optional)
        Name of the cross_correlation_ratio field.

    Returns:
    --------
    gf: pyart-gatefilter object
        GateFilter of the Meteorological echoes only.
    """
    # Load Scikit model
    location = os.path.dirname(os.path.realpath(__file__))
    my_file = os.path.join(location, "data", "GM_model_radar_metechoes.pkl.gz")
    with gzip.GzipFile(my_file, "r") as gzid:
        meteorological_echoes_GMM = pickle.load(gzid)

    df_orig = pd.DataFrame(
        {
            "total_power": radar.fields[dbz_name]["data"].flatten(),
            "differential_reflectivity": radar.fields[zdr_name]["data"].flatten(),
            "differential_phase": radar.fields[phidp_name]["data"].flatten(),
            "spectrum_width": radar.fields[width_name]["data"].flatten(),
            "cross_correlation_ratio": radar.fields[rhohv_name]["data"].flatten(),
        }
    )

    pos_droped = df_orig.dropna().index
    radar_cluster = meteorological_echoes_GMM.predict(df_orig.dropna())

    r = radar.range["data"]
    time = radar.time["data"]
    R, T = np.meshgrid(r, time)

    clus = np.zeros_like(R.flatten())
    clus[pos_droped] = radar_cluster + 1
    cluster = clus.reshape(R.shape)

    meteorological_signal = (cluster >= 5) | (cluster == 2)  # | ((R < 20e3) & ((cluster == 1) | (cluster == 4)))

    hydro_class = np.zeros(cluster.shape, dtype=np.int16)
    hydro_class[meteorological_signal] = 3
    hydro_class[((R < 20e3) & ((cluster == 1) | (cluster == 4)))] = 2
    hydro_class[(hydro_class == 0) & (cluster != 0)] = 1

    radar.add_field("good_mask", {"data": meteorological_signal})
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_equal("good_mask", 0)
    gf = pyart.correct.despeckle_field(radar, dbz_name, gatefilter=gf)
    _ = radar.fields.pop("good_mask")

    return gf, hydro_class


# def do_gatefilter_opol(
#     radar, refl_name="DBZ", phidp_name="PHIDP", rhohv_name="RHOHV_CORR", zdr_name="ZDR", width_name="WIDTH"
# ):
#     """
#     Filtering function adapted to CPOL.

#     Parameters:
#     ===========
#         radar:
#             Py-ART radar structure.
#         refl_name: str
#             Reflectivity field name.
#         rhohv_name: str
#             Cross correlation ratio field name.
#         ncp_name: str
#             Name of the normalized_coherent_power field.
#         zdr_name: str
#             Name of the differential_reflectivity field.

#     Returns:
#     ========
#         gf_despeckeld: GateFilter
#             Gate filter (excluding all bad data).
#     """
#     gf, hydroclass = get_gatefilter_GMM(
#         radar,
#         dbz_name=refl_name,
#         zdr_name=zdr_name,
#         phidp_name=phidp_name,
#         width_name=width_name,
#         rhohv_name=rhohv_name,
#     )

#     echoclass = {
#         "data": hydroclass,
#         "long_name": "radar_echo_classification",
#         "units": " ",
#         "description": "0: N/A, 1: Clutter, 2: Clear Air, 3: Meteorological echoes",
#     }

#     return gf, echoclass

def do_gatefilter_opol(
    radar,
    refl_name="DBZ", phidp_name="PHIDP", rhohv_name="RHOHV_CORR", zdr_name="ZDR", width_name="WIDTH", snr_name="SNR"
):
    """
    Filtering function adapted to CPOL.
    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    refl_name: str
        Reflectivity field name.
    rhohv_name: str
        Cross correlation ratio field name.
    ncp_name: str
        Name of the normalized_coherent_power field.
    zdr_name: str
        Name of the differential_reflectivity field.
    Returns:
    ========
    gf_despeckeld: GateFilter
        Gate filter (excluding all bad data).
    """    
    gf = pyart.filters.GateFilter(radar)    
        
    gf.exclude_below(snr_name, 9)
    gf.exclude_below(rhohv_name, 0.75)
    gf.exclude_outside(zdr_name, -3.0, 7.0)
    gf.exclude_outside(refl_name, -20.0, 80.0)

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld
