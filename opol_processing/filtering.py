"""
Codes for creating and manipulating gate filters. New functions: use of trained
Gaussian Mixture Models to remove noise and clutter from CPOL data before 2009.

@title: filtering
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 25/08/2021

.. autosummary::
    :toctree: generated/

    do_gatefilter_cpol
"""
import pyart


def do_gatefilter_opol(
    radar,
    refl_name: str = "DBZ",
    rhohv_name: str = "RHOHV_CORR",
    zdr_name: str = "ZDR",
    ncp_name: str = "NCP"
) -> pyart.filters.GateFilter:
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

    try:
        gf.exclude_below(ncp_name, 0.3)
    except Exception:
        pass
    gf.exclude_below(rhohv_name, 0.5)
    gf.exclude_outside(zdr_name, -2.0, 7.0)
    gf.exclude_outside(refl_name, -15.0, 90.0)

    gf_despeckeld = pyart.correct.despeckle_field(radar, refl_name, gatefilter=gf)

    return gf_despeckeld
