"""
Gate filters and numba cleaning kernels for OceanPOL processing.

Three role-separated gate filters are built, each reproducing an oceanpol_kit
mask (they are intentionally NOT merged: the masks serve opposite purposes and
merging reintroduces clear-air velocity loss and PHIDP streaks):

- ``do_gatefilter_opol``    : reflectivity cleaning (hydrometeor mask).
- ``do_precip_gatefilter``  : strict precipitation gate for phase processing.
- ``do_velocity_gatefilter``: permissive coherence gate for velocity dealiasing.

The despeckle / texture kernels (`area_std`, `speckle_filter`, `despeckle_mask`)
are the oceanpol_kit numba implementations, ported verbatim. They replace
``pyart.correct.despeckle_field`` (connected-component labelling), which is far
too slow per volume.

@title: filtering
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    area_std
    speckle_filter
    despeckle_mask
    get_hydrometeor_mask
    do_gatefilter_opol
    do_precip_gatefilter
    do_velocity_gatefilter
"""
import pyart
import numpy as np

from numba import njit


# ---------------------------------------------------------------------------
# Numba kernels (ported verbatim from oceanpol_kit)
# ---------------------------------------------------------------------------
@njit(cache=True)
def area_std(rawphi: np.ndarray, winlen: int = 10) -> np.ndarray:
    """
    Standard deviation of a sliding window along range (texture), per ray.

    Parameters
    ----------
    rawphi : np.ndarray
        2D array (na, nr), float64.
    winlen : int, optional
        Sliding window length (default 10).

    Returns
    -------
    np.ndarray
        2D texture array, same shape as ``rawphi``.
    """
    na, nr = rawphi.shape
    area_phi = np.zeros_like(rawphi)
    for i in range(na):
        for j in range(nr - winlen):
            window = rawphi[i, j : (j + winlen)]
            area_phi[i, j] = np.std(window)
        for j in range(nr - 1, nr - winlen - 1, -1):
            window = rawphi[i, j - winlen : j]
            area_phi[i, j] = np.std(window)

    return area_phi


@njit(cache=True)
def speckle_filter(data: np.ndarray, mask: np.ndarray, min_dbz: float = -10.0, min_neighbours: int = 3) -> np.ndarray:
    """
    Remove speckle from radar reflectivity.

    A gate is retained if it is above ``min_dbz`` and either (a) it is a masked
    (noise) gate surrounded by enough non-masked neighbours, or (b) it has at
    least ``min_neighbours`` of its 8 neighbours above ``min_dbz``. All other
    gates are set to NaN.

    Parameters
    ----------
    data : np.ndarray
        2D reflectivity (float64), with NaN where already masked.
    mask : np.ndarray
        2D mask (int8), 1 = noise/excluded gate, 0 = kept.
    min_dbz : float, optional
        Minimum reflectivity to consider (default -10).
    min_neighbours : int, optional
        Minimum number of qualifying neighbours to retain a gate (default 3).

    Returns
    -------
    np.ndarray
        2D filtered reflectivity (NaN outside retained gates).
    """
    ny, nx = data.shape
    copy = np.zeros((ny, nx)) + np.nan

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            if data[y][x] <= min_dbz:
                continue

            if mask[y][x] == 1:
                count_mask = (
                    (mask[y - 1, x - 1] == 0)
                    + (mask[y - 1, x] == 0)
                    + (mask[y - 1, x + 1] == 0)
                    + (mask[y, x - 1] == 0)
                    + (mask[y, x + 1] == 0)
                    + (mask[y + 1, x - 1] == 0)
                    + (mask[y + 1, x] == 0)
                    + (mask[y + 1, x + 1] == 0)
                )
                if count_mask > min_neighbours:
                    copy[y][x] = data[y][x]
                    continue

            count = (
                (data[y - 1, x - 1] > min_dbz)
                + (data[y - 1, x] > min_dbz)
                + (data[y - 1, x + 1] > min_dbz)
                + (data[y, x - 1] > min_dbz)
                + (data[y, x + 1] > min_dbz)
                + (data[y + 1, x - 1] > min_dbz)
                + (data[y + 1, x] > min_dbz)
                + (data[y + 1, x + 1] > min_dbz)
            )

            if count >= min_neighbours:
                copy[y][x] = data[y][x]
    return copy


@njit(cache=True)
def despeckle_mask(mask: np.ndarray, min_neighbours: int = 2) -> np.ndarray:
    """
    Remove isolated True gates from a boolean keep-mask.

    A gate is retained only if at least ``min_neighbours`` of its 8 immediate
    neighbours are also True.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array (True = keep).
    min_neighbours : int, optional
        Minimum number of True neighbours required (default 2).

    Returns
    -------
    np.ndarray
        Despeckled boolean array.
    """
    ny, nx = mask.shape
    out = np.zeros((ny, nx), dtype=np.bool_)
    for y in range(ny):
        for x in range(nx):
            if not mask[y, x]:
                continue
            count = 0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < ny and 0 <= xx < nx and mask[yy, xx]:
                        count += 1
            if count >= min_neighbours:
                out[y, x] = True
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _filled(data) -> np.ndarray:
    """Contiguous float64 view of a (masked) field, masked entries -> NaN."""
    return np.ascontiguousarray(np.ma.filled(np.ma.asarray(data, dtype="float64"), np.nan))


def get_hydrometeor_mask(dbz: np.ndarray, phidp: np.ndarray, rhohv: np.ndarray) -> np.ndarray:
    """
    Hydrometeor noise mask (True = remove), reproducing oceanpol_kit:
    exclude where (PHIDP texture > 60) OR (RHOHV < 0.5) OR (refl < -20), but
    rescue gates where RHOHV > 0.90.
    """
    area = area_std(_filled(phidp))
    pos = (area > 60) | (rhohv < 0.5) | (dbz < -20)
    pos[rhohv > 0.90] = False
    return pos


# ---------------------------------------------------------------------------
# Gate filters
# ---------------------------------------------------------------------------
def do_gatefilter_opol(
    radar,
    refl_name: str = "total_power",
    rhohv_name: str = "cross_correlation_ratio",
    phidp_name: str = "PHIDP",
) -> "pyart.filters.GateFilter":
    """
    Reflectivity-cleaning gate filter from ``get_hydrometeor_mask`` thresholds.

    The returned filter carries the hydrometeor noise mask; the actual speckle
    removal on the reflectivity data is done with the numba ``speckle_filter``
    in the production line (faster and matching oceanpol_kit).
    """
    dbz = _filled(radar.fields[refl_name]["data"])
    phidp = _filled(radar.fields[phidp_name]["data"])
    rhohv = _filled(radar.fields[rhohv_name]["data"])

    mask = get_hydrometeor_mask(dbz, phidp, rhohv)

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_gates(mask)
    return gf


def do_precip_gatefilter(
    radar,
    refl_name: str = "corrected_reflectivity",
    rhohv_name: str = "cross_correlation_ratio",
    snr_name: str = "SNR",
    rho_min: float = 0.8,
    snr_min: float = 3.0,
) -> "pyart.filters.GateFilter":
    """
    Strict precipitation gate filter (RHOHV >= 0.8, SNR >= 3 dB, finite refl),
    reproducing ``get_precip_mask``. Applied before phase processing.
    """
    gf = pyart.filters.GateFilter(radar)
    gf.exclude_below(rhohv_name, rho_min)
    gf.exclude_below(snr_name, snr_min)
    gf.exclude_invalid(refl_name)
    return gf


def do_velocity_gatefilter(
    radar,
    vel_name: str = "VRAD",
    sqi_name: str = "SQI",
    th_name: str = "total_power",
    sqi_min: float = 0.4,
    texture_max: float = None,
    texture_win: int = 5,
    min_neighbours: int = 2,
) -> "pyart.filters.GateFilter":
    """
    Permissive coherence gate filter for velocity, reproducing
    ``get_velocity_mask``: finite velocity, finite total power (no power ->
    receiver noise), SQI above threshold, optional velocity texture, then a
    numba despeckle. SNR/RHOHV are deliberately not used so coherent clear-air
    velocity is preserved.
    """
    vel = _filled(radar.fields[vel_name]["data"])
    good = np.isfinite(vel)
    good &= np.isfinite(_filled(radar.fields[th_name]["data"]))

    has_sqi = sqi_name is not None and sqi_name in radar.fields
    if has_sqi:
        good &= _filled(radar.fields[sqi_name]["data"]) >= sqi_min

    use_texture = texture_max if texture_max is not None else (6.0 if not has_sqi else None)
    if use_texture is not None:
        texture = area_std(vel, texture_win)
        good &= texture <= use_texture

    good = despeckle_mask(np.ascontiguousarray(good), min_neighbours)

    gf = pyart.filters.GateFilter(radar)
    gf.exclude_gates(~good)
    return gf
