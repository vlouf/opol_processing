"""
Codes for correcting the differential phase and estimating KDP.

@title: phase
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology
@date: 06/05/2024

.. autosummary::
    :toctree: generated/

    _fix_phidp_from_kdp
    phidp_giangrande
    phido
"""
import pyart
import numpy as np

from phido import kdp_pyart
from scipy import integrate
from csu_radartools import csu_kdp


def _fix_phidp_from_kdp(phidp, kdp, r, gatefilter):
    """
    Correct PHIDP and KDP from spider webs.

    Parameters
    ==========
    r:
        Radar range.
    gatefilter:
        Gate filter.
    kdp_name: str
        Differential phase key name.
    phidp_name: str
        Differential phase key name.

    Returns:
    ========
    phidp: ndarray
        Differential phase array.
    """
    kdp[gatefilter.gate_excluded] = 0
    kdp[(kdp < -4)] = 0
    kdp[kdp > 15] = 0
    interg = integrate.cumtrapz(kdp, r, axis=1)

    phidp[:, :-1] = interg / (len(r))
    return phidp, kdp


def phidp_bringi(radar, gatefilter, phidp_name="PHIDP", refl_field="DBZ"):
    """
    Compute PHIDP and KDP Bringi.

    Parameters
    ==========
    radar:
        Py-ART radar data structure.
    gatefilter:
        Gate filter.
    phidp_name: str
        Differential phase key name.
    refl_field: str
        Reflectivity key name.

    Returns:
    ========
    phidpb: ndarray
        Bringi differential phase array.
    kdpb: ndarray
        Bringi specific differential phase array.
    """
    nphase = pyart.correct.phase_proc.det_sys_phase_gf(radar, gatefilter, phidp_field=phidp_name, first_gate=30)
    if nphase is None:
        nphase = 0
    dp = radar.fields[phidp_name]["data"].copy()    
    dp -= nphase
    dz = radar.fields[refl_field]["data"].copy().filled(-9999)

    # Extract dimensions
    rng = radar.range["data"]
    azi = radar.azimuth["data"]
    dgate = rng[1] - rng[0]
    [R, A] = np.meshgrid(rng, azi)

    # Compute KDP bringi.
    kdpb, phidpb, _ = csu_kdp.calc_kdp_bringi(dp, dz, R / 1e3, gs=dgate, bad=-9999, thsd=12, window=3.0, std_gate=11)

    # Mask array
    phidpb = np.ma.masked_where(phidpb == -9999, phidpb)
    kdpb = np.ma.masked_where(kdpb == -9999, kdpb)

    # Get metadata.
    phimeta = pyart.config.get_metadata("differential_phase")
    phimeta["data"] = phidpb
    kdpmeta = pyart.config.get_metadata("specific_differential_phase")
    kdpmeta["data"] = kdpb

    return phimeta, kdpmeta


def phidp_giangrande(radar, gatefilter, refl_field="DBZ", ncp_field="NCP", rhv_field="RHOHV_CORR", phidp_field="PHIDP"):
    """
    Phase processing using the LP method in Py-ART. A LP solver is required,

    Parameters:
    ===========
    radar:
        Py-ART radar structure.
    gatefilter:
        Gate filter.
    refl_field: str
        Reflectivity field label.
    ncp_field: str
        Normalised coherent power field label.
    rhv_field: str
        Cross correlation ration field label.
    phidp_field: str
        Differential phase label.

    Returns:
    ========
    phidp_gg: dict
        Field dictionary containing processed differential phase shifts.
    kdp_gg: dict
        Field dictionary containing recalculated differential phases.
    """
    unfphidic = pyart.correct.dealias_unwrap_phase(
        radar, gatefilter=gatefilter, skip_checks=True, vel_field=phidp_field, nyquist_vel=90
    )

    radar.add_field_like(phidp_field, "PHITMP", unfphidic["data"])

    phidp_gg, kdp_gg = pyart.correct.phase_proc_lp(
        radar,
        0.0,
        LP_solver="cylp",
        ncp_field=ncp_field,
        refl_field=refl_field,
        rhv_field=rhv_field,
        phidp_field="PHITMP",
    )

    phidp_gg["data"], kdp_gg["data"] = _fix_phidp_from_kdp(
        phidp_gg["data"], kdp_gg["data"], radar.range["data"], gatefilter
    )

    try:
        # Remove temp variables.
        radar.fields.pop("unfolded_differential_phase")
        radar.fields.pop("PHITMP")
    except Exception:
        pass

    phidp_gg["data"] = phidp_gg["data"].astype(np.float32)
    phidp_gg["_Least_significant_digit"] = 4
    kdp_gg["data"] = kdp_gg["data"].astype(np.float32)
    kdp_gg["_Least_significant_digit"] = 4

    return phidp_gg, kdp_gg


def phido(radar, gatefilter, refl_field, rhv_field="RHOHV_CORR", phidp_field="PHIDP", zdr_name="ZDR_CORR"):
    # filter bad data
    gatefilter = pyart.correct.GateFilter(radar)
    gatefilter.exclude_masked(refl_field)
    gatefilter.exclude_below(rhv_field, 0.95)
    gatefilter.exclude_outside(zdr_name, -1.0, 4.0)

    # calculate kdp
    kdp_meta, phidp_meta = kdp_pyart(radar, phidp_field, gatefilter, window = (3, 7)) 
    kdp_meta["data"] = kdp_meta["data"].astype(np.float32)
    phidp_meta["data"] = phidp_meta["data"].astype(np.float32)
    radar.add_field('kdp_phido', kdp_meta, replace_existing = True)
    radar.add_field('phidp_phido', phidp_meta, replace_existing = True)
    return phidp_meta, kdp_meta