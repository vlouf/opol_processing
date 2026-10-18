"""
Pyrometeor (wildfire debris/smoke plume) detection using the detection and
replacement algorithm (DRA) of Miller, Schvartzman and Zrnic (2026, "Detecting
and Classifying Wildland Fire-Related Pyrometeors using Polarimetric
Observations from WSR-88Ds").

The DRA masks gates on polarimetric thresholds (rhohv, ZDR, Z and the along-ray
texture of Z), keeps contiguous detected regions larger than a minimum area,
and replaces the classification of the detected gates with a dedicated PYR
class in the existing HID field. Thresholds are the values published in the
paper (derived from S-band WSR-88D statistics).

All functions operate on 2D polar (azimuth x range) plain numpy arrays
(NaN-filled, not masked), consistent with the rest of this package.

@title: pyrometeors
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    texture_along_ray
    detect_pyrometeors
    insert_pyrometeors
"""
from typing import Optional

import numpy as np
from scipy import ndimage

# Thresholds from Miller et al.: gates are pyrometeor candidates when
# Z > -10 dBZ, rhohv < 0.6, -2 <= ZDR <= 8 dB and SD(Z) <= 6 dBZ.
DBZ_MIN = -10.0
RHOHV_MAX = 0.6
ZDR_MIN = -2.0
ZDR_MAX = 8.0
SDZ_MAX = 6.0
# Window length (km) for the along-ray texture of Z (Ryzhkov and Zrnic 2019).
SDZ_WINDOW_KM = 1.0
# Minimum area (km2) for a detected contiguous region to be kept.
MIN_AREA_KM2 = 80.0
# Value assigned to detected gates in the HID field. The CSU summer scheme
# uses classes 0-10, so pyrometeors take the next free value.
PYR_CLASS = 11


def texture_along_ray(field: np.ndarray, gate_spacing: float, window_km: float = SDZ_WINDOW_KM) -> np.ndarray:
    """
    Standard deviation of a field within a running window along each ray (the
    1D radial texture method used for SD(Z) in the WSR-88D HCA).

    Parameters
    ----------
    field : np.ndarray
        2D field (azimuth x range), NaN-filled.
    gate_spacing : float
        Range gate spacing in meters.
    window_km : float
        Window length in km (default 1 km, as for SD(Z)).

    Returns
    -------
    np.ndarray
        2D standard deviation of ``field`` within the window; NaN where
        ``field`` is NaN.
    """
    wlen = int(round(window_km * 1e3 / gate_spacing))
    wlen = max(wlen, 3)
    if wlen % 2 == 0:
        wlen += 1

    valid = np.isfinite(field)
    f0 = np.where(valid, field, 0.0)

    # NaN-aware running mean/variance along range (axis=1).
    n = ndimage.uniform_filter1d(valid.astype(np.float64), wlen, axis=1, mode="nearest") * wlen
    s = ndimage.uniform_filter1d(f0, wlen, axis=1, mode="nearest") * wlen
    ss = ndimage.uniform_filter1d(f0**2, wlen, axis=1, mode="nearest") * wlen

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (ss - s**2 / n) / n
    var[n < 2] = np.nan
    var[var < 0] = 0

    sd = np.sqrt(var)
    sd[~valid] = np.nan
    return sd


def _gate_area_km2(radar_range: np.ndarray, nazimuth: int) -> np.ndarray:
    """
    Approximate area (km2) of each gate of a polar sweep: r * dtheta * dr,
    broadcast to (azimuth x range).
    """
    r_km = radar_range / 1e3
    dr_km = np.median(np.diff(r_km))
    dtheta = 2 * np.pi / nazimuth
    area = r_km * dtheta * dr_km
    return np.broadcast_to(area[np.newaxis, :], (nazimuth, len(r_km)))


def detect_pyrometeors(
    dbz: np.ndarray,
    zdr: np.ndarray,
    rhohv: np.ndarray,
    radar_range: np.ndarray,
    sd_dbz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect pyrometeor regions on a PPI sweep using the DRA thresholds of
    Miller et al.: gates with Z > -10 dBZ, rhohv < 0.6, -2 <= ZDR <= 8 dB and
    SD(Z) <= 6 dBZ (removes complex-terrain clutter), then only contiguous
    regions larger than 80 km2 are kept.

    Parameters
    ----------
    dbz, zdr, rhohv : np.ndarray
        2D radar fields (azimuth x range), NaN-filled.
    radar_range : np.ndarray
        1D range of the gates (meters).
    sd_dbz : np.ndarray, optional
        Precomputed along-ray texture of Z. Computed here if None.

    Returns
    -------
    np.ndarray
        2D boolean mask, True for detected pyrometeor gates.
    """
    if sd_dbz is None:
        gate_spacing = float(np.median(np.diff(radar_range)))
        sd_dbz = texture_along_ray(dbz, gate_spacing)

    mask = (
        (dbz > DBZ_MIN)
        & (rhohv < RHOHV_MAX)
        & (zdr >= ZDR_MIN)
        & (zdr <= ZDR_MAX)
        & (sd_dbz <= SDZ_MAX)
    )
    mask &= np.isfinite(dbz) & np.isfinite(rhohv) & np.isfinite(zdr)
    if not mask.any():
        return mask

    # Contiguous regions (8-connected), merging components across the 0/360
    # degree azimuth boundary.
    labels, nlabels = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    equiv = {}
    first, last = labels[0, :], labels[-1, :]
    wrapped = (first > 0) & (last > 0)
    for a, b in zip(first[wrapped], last[wrapped]):
        ra, rb = equiv.get(a, a), equiv.get(b, b)
        if ra != rb:
            keep, drop = min(ra, rb), max(ra, rb)
            for k, v in equiv.items():
                if v == drop:
                    equiv[k] = keep
            equiv[drop] = keep
    if equiv:
        for drop, keep in equiv.items():
            labels[labels == drop] = keep

    # Area filter: keep regions >= MIN_AREA_KM2.
    area = _gate_area_km2(radar_range, dbz.shape[0])
    out = np.zeros_like(mask)
    for lbl in np.unique(labels[labels > 0]):
        region = labels == lbl
        if area[region].sum() >= MIN_AREA_KM2:
            out |= region
    return out


def insert_pyrometeors(hid: np.ndarray, pyro_mask: np.ndarray) -> np.ndarray:
    """
    Replace the classification of detected pyrometeor gates with the PYR class
    (the "replacement" step of the DRA). The rest of the HID field is left
    untouched.

    Parameters
    ----------
    hid : np.ndarray
        2D hydrometeor classification (int-valued, 0 = no data).
    pyro_mask : np.ndarray
        2D boolean pyrometeor mask from :func:`detect_pyrometeors`.

    Returns
    -------
    np.ndarray
        Copy of ``hid`` with detected gates set to ``PYR_CLASS``.
    """
    out = hid.copy()
    out[pyro_mask] = PYR_CLASS
    return out
