"""
LROSE self-consistency PHIDP / KDP retrieval - the default phase processing of
the OPOL pipeline.

References
----------
Dixon, M., and J. Hubbert, 2026: A robust, general-purpose KDP algorithm for
    LROSE. Proc. 13th European Conference on Radar in Meteorology and Hydrology
    (ERAD 2026), Belgrade, Serbia.
Hubbert, J., and V. N. Bringi, 1995: An iterative filtering technique for the
    analysis of copolar differential phase and dual-frequency radar
    measurements. J. Atmos. Oceanic Technol., 12, 643-648.
LROSE ``KdpFilt`` (NCAR/lrose-core, ``codebase/libs/radar/src/kdp``).

Two stages, and it matters which is which:

**Stage 1** is a faithful port of ``KdpFilt.cc`` from NCAR/lrose-core
(``codebase/libs/radar/src/kdp``, HEAD 3871d925c, 2026-09-21): folding-range
detection, circular PHIDP statistics over a 9-gate kernel, valid-run
identification, vector-space unfolding, FIR filtering (Hubbert & Bringi 1995),
peak-removal conditioning, and the adaptive-length KDP slope.  Where this file
deviates from the C++ it says so in a comment beginning ``DEVIATION``.

**Stage 2** is the endpoint-constrained local self-consistency method of
Dixon & Hubbert (2026), sections 6.1-6.3.  This is **not** in lrose master:
``KdpFilt::_loadKdpSCRun`` normalises over runs of positive KDP with no
segmentation and no endpoint constraint.  Both are implemented so they can be
compared directly - :func:`kdp_sc_segmented` (paper) and :func:`kdp_sc_runs`
(current LROSE).  The pipeline uses the paper's method: ``kdp_sc`` and
``phidp_sc`` are what :func:`retrieve_phidp_kdp` returns.

Not implemented: section 6.4's two-pass attenuation iteration (Stage 3), and the
quadratic / regression / FFT filter variants of section 5.  A one-pass
attenuation correction is available via ``KdpParams.atten_correct`` and is off by
default (the OPOL pipeline applies its own Z-PHI correction downstream).

Adapted for opol_processing from the stand-alone Python test-bed: the pyodim /
xarray sweep driver has been replaced by a Py-ART volume driver,
:func:`retrieve_phidp_kdp`.

@title: kdp_lrose
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    KdpParams
    process_ray
    retrieve_phidp_kdp
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter

RAD2DEG = 180.0 / np.pi

# --------------------------------------------------------------------------
# Coefficient tables
# --------------------------------------------------------------------------

# FIR filter coefficients, verbatim from KdpFilt.cc:52-148.  Keyed by the LROSE
# name (FIR_LEN_10 has 11 taps, etc).
FIR_COEFFS = {
    10: np.array([
        0.03064579383, 0.0603038422, 0.09022859603, 0.1159074511,
        0.1332367851, 0.1393550634, 0.1332367851, 0.1159074511,
        0.09022859603, 0.0603038422, 0.03064579383]),
    20: np.array([
        0.016976991942, 0.023294989742, 0.030244475217, 0.037550056394,
        0.044888313214, 0.051908191403, 0.058254532798, 0.063592862330,
        0.067633391375, 0.070152221980, 0.071007947209, 0.070152221980,
        0.067633391375, 0.063592862330, 0.058254532798, 0.051908191403,
        0.044888313214, 0.037550056394, 0.030244475217, 0.023294989742,
        0.016976991942]),
    30: np.array([
        0.01040850049, 0.0136551033, 0.01701931136, 0.0204494327,
        0.0238905658, 0.02728575662, 0.03057723021, 0.03370766631,
        0.03662148602, 0.03926611662, 0.04159320123, 0.04355972181,
        0.04512900539, 0.04627158699, 0.04696590613, 0.04719881804,
        0.04696590613, 0.04627158699, 0.04512900539, 0.04355972181,
        0.04159320123, 0.03926611662, 0.03662148602, 0.03370766631,
        0.03057723021, 0.02728575662, 0.0238905658, 0.0204494327,
        0.01701931136, 0.0136551033, 0.01040850049]),
    40: np.array([
        0.007806525986, 0.009628559511, 0.01150585082, 0.01342243276,
        0.01536143961, 0.01730530352, 0.0192359639, 0.02113508696,
        0.02298429218, 0.02476538263, 0.02646057561, 0.02805273044,
        0.02952556994, 0.03086389233, 0.03205377043, 0.03308273518,
        0.03393994069, 0.03461630839, 0.03510464806, 0.0353997539,
        0.03549847428, 0.0353997539, 0.03510464806, 0.03461630839,
        0.03393994069, 0.03308273518, 0.03205377043, 0.03086389233,
        0.02952556994, 0.02805273044, 0.02646057561, 0.02476538263,
        0.02298429218, 0.02113508696, 0.0192359639, 0.01730530352,
        0.01536143961, 0.01342243276, 0.01150585082, 0.009628559511,
        0.007806525986]),
}

# Self-consistency relation, Eq. (3):  KDP = a * Z_H**b * Z_DR**c
# with Z_H, Z_DR in linear units.  Paper Table 4.
SC_COEFFS_PAPER = {
    "S": (0.0001051, 0.96, -0.260),   # Scarchilli et al. (1996)
    "C": (0.0001461, 0.98, -0.200),   # Scarchilli et al. (1996)
    "X": (0.000222, 1.00, -4.58),     # Shi et al. (2018)
}

# Attenuation power laws, Eq. (5)/(6):  A_H = aH*KDP**bH,  A_DP = aDP*KDP**bDP
# Paper Table 5 / Bringi & Chandrasekar Table 7.1.  Same values as
# KdpFilt::compute().
ATTEN_COEFFS = {
    "S": (0.017, 0.84, 0.003, 1.05),
    "C": (0.073, 0.99, 0.013, 1.23),
    "X": (0.233, 1.02, 0.033, 1.15),
}


def band_from_wavelength(wavelength_cm: float) -> str:
    """Band letter, using the same cut points as ``KdpFilt::compute``."""
    if wavelength_cm < 4:
        return "X"
    if wavelength_cm < 7:
        return "C"
    return "S"


def sc_coeffs(wavelength_cm: float, which: str = "lrose") -> tuple[float, float, float]:
    """Self-consistency coefficients ``(a, b, c)`` for Eq. (3).

    ``which="lrose"`` reproduces ``KdpFilt::compute``, which scales a single
    coefficient by wavelength::

        a = 3.32e-5 * (10 / lambda_cm),  b = 1.0,  c = -2.05

    ``which="paper"`` uses the band-specific Table 4 values.

    The two disagree by an order of magnitude in ``a`` - but under the
    endpoint constraint of Eq. (13) ``a`` cancels exactly, so only ``b`` and
    ``c`` (the *shape* of the relation) affect KDP_sc.  ``c`` differs a lot
    (-2.05 vs -0.26 at S-band), so this switch is worth exercising.
    """
    if which == "lrose":
        return 3.32e-5 * (10.0 / wavelength_cm), 1.0, -2.05
    if which == "paper":
        return SC_COEFFS_PAPER[band_from_wavelength(wavelength_cm)]
    raise ValueError(f"unknown sc coefficient set: {which!r}")


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass
class KdpParams:
    """Tunables.  Defaults follow ``paramdef.KdpFilt`` unless noted."""

    # --- Stage 1: validity (paper Table 2 / paramdef.KdpFilt) ---
    n_gates_stats: int = 9          # computational kernel
    phidp_sdev_max: float = 20.0    # deg
    phidp_jitter_max: float = 25.0  # deg; paper Table 2 says 30
    check_snr: bool = True          # paramdef default false; we have SNRH
    snr_threshold: float = 0.0      # dB; paper Table 2. paramdef says -6
    check_rhohv: bool = True
    rhohv_threshold: float = 0.85   # paper Table 2. paramdef says 0.95
    check_zdr_sdev: bool = False
    zdr_sdev_max: float = 2.0

    # --- Stage 1: filtering (section 5) ---
    filter: str = "fir"             # "fir" (Hubbert & Bringi) or "savgol"
    fir_len: int = 10               # key into FIR_COEFFS
    n_filt_iter_unfolded: int = 2
    n_filt_iter_cond: int = 4
    lf_km: float = 3.5              # feature length, section 5.5 (savgol only)

    # --- Stage 1: KDP slope ---
    median_len_zzdr: int = 5        # KDP_median_filter_len_for_ZZDR

    # --- Stage 2: self-consistency ---
    sc_coeff_set: str = "lrose"     # "lrose" or "paper"; see sc_coeffs()
    dbz_max_for_sc: float | None = None  # cap Z before Eq. (3); see note below
    kdp_min_for_sc: float = 0.25    # LROSE run method only
    min_seg_km: float | None = None   # segment floor; defaults to lf_km
    max_seg_km: float | None = 10.0   # segment ceiling; see segment_dividers()
    min_delta_phi: float = 1.0      # deg; skip segments with dPHI below this
    min_sc_phase: float = 0.5       # deg; skip if the SC integral is tiny
    delta_threshold: float = 1.0    # deg; section 6.3 accept/reject on mean delta

    # --- Stage 3 hook (off by default) ---
    atten_correct: bool = False


# --------------------------------------------------------------------------
# Small array helpers
# --------------------------------------------------------------------------

def _win_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Centred sliding-window sum of length ``n``, clipped at the array ends.

    Clipping (rather than padding) reproduces the ``jj < 0 || jj >= _nGates:
    continue`` guard in ``_computePhidpStats``.
    """
    half = n // 2
    csum = np.concatenate([[0.0], np.cumsum(a)])
    idx = np.arange(a.size)
    lo = np.clip(idx - half, 0, a.size)
    hi = np.clip(idx + half + 1, 0, a.size)
    return csum[hi] - csum[lo]


def _running_nanmedian(a: np.ndarray, n: int) -> np.ndarray:
    """Centred running nan-median; the ``n//2`` end gates are passed through."""
    if n <= 1:
        return a.copy()
    half = n // 2
    out = a.copy()
    view = np.lib.stride_tricks.sliding_window_view(a, n)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(view, axis=-1)
    out[half:half + med.size] = med
    return out


def _contiguous(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` index pairs for each True run in ``mask``."""
    if not mask.any():
        return []
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _fill_nan(a: np.ndarray) -> np.ndarray:
    """Linear interpolation over non-finite values, with edge extension.

    DEVIATION: the C++ leaves no gaps at this point in the pipeline, because
    missing PHIDP is filled from the gap-interpolated mean field.  With ODIM
    data the noise region is genuinely absent (NaN) rather than random, so a
    guard is cheap insurance before convolving.
    """
    out = np.asarray(a, dtype=float).copy()
    good = np.isfinite(out)
    if good.all():
        return out
    if not good.any():
        return np.zeros_like(out)
    idx = np.arange(out.size)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def normalize_phidp_range(phidp: np.ndarray) -> np.ndarray:
    """Shift PHIDP stored on 0-360 onto -180..180.

    ``KdpFilt::_computeFoldingRange`` does this per ray, keyed on
    ``phidpMin >= 0 && phidpMax > 180``.

    DEVIATION: we apply it once for the whole sweep.  Per ray it is unstable -
    a weak ray whose PHIDP happens to stay inside 0..180 is left unshifted and
    ends up offset by 180 deg from its neighbours.  KDP is a derivative so it
    does not care, but the PHIDP fields are unusable in a PPI.
    """
    finite = phidp[np.isfinite(phidp)]
    if finite.size and finite.min() >= 0 and finite.max() > 180:
        return phidp - 180.0
    return phidp


# --------------------------------------------------------------------------
# Stage 1 - PHIDP conditioning (port of KdpFilt.cc)
# --------------------------------------------------------------------------

def folding_range(phidp: np.ndarray) -> tuple[float, bool]:
    """``(fold_val, folds_at_90)`` - port of ``_computeFoldingRange``."""
    finite = phidp[np.isfinite(phidp)]
    if finite.size == 0:
        return 180.0, False
    if finite.min() > -90 and finite.max() < 90:
        return 90.0, True
    return 180.0, False


def phidp_stats(phidp: np.ndarray, n_stats: int, folds_at_90: bool):
    """Circular mean, sdev and jitter of PHIDP - port of ``_computePhidpStats``.

    Returns ``(mean, sdev, jitter, mean_xx, mean_yy)``, all NaN where the C++
    would leave the value missing.

    Note that LROSE computes *both* sdev and jitter from the gate-to-gate chord
    length on the unit circle (``GateState::distFromPrev``), treating the chord
    as an angle in radians.  ``PhidpProc.cc`` in the same library computes a
    true circular sdev about the circular mean, which is what section 3.1 of
    the paper describes.  We follow ``KdpFilt.cc``, since the 20/25 deg
    thresholds are the ones tuned against it.
    """
    n = phidp.size
    half = n_stats // 2
    valid = np.isfinite(phidp)

    phase = np.where(valid, phidp, 0.0) * (2.0 if folds_at_90 else 1.0)
    xx = np.where(valid, np.cos(np.deg2rad(phase)), 0.0)
    yy = np.where(valid, np.sin(np.deg2rad(phase)), 0.0)

    # Chord length from the previous gate, 0 unless both gates have data
    dist = np.zeros(n)
    both = valid[1:] & valid[:-1]
    dist[1:] = np.where(both, np.hypot(np.diff(xx), np.diff(yy)), 0.0)

    count = _win_sum(valid.astype(float), n_stats)
    sum_xx = _win_sum(xx, n_stats)
    sum_yy = _win_sum(yy, n_stats)
    sum_d = _win_sum(dist, n_stats)
    sum_d2 = _win_sum(dist * dist, n_stats)

    ok = count > half                         # "if (count <= _nGatesStatsHalf) return"
    idx = np.arange(n)
    ok &= (idx >= half) & (idx < n - half)    # C++ only loops over this range

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_xx = np.where(ok, sum_xx / count, np.nan)
        mean_yy = np.where(ok, sum_yy / count, np.nan)
        mean = np.degrees(np.arctan2(mean_yy, mean_xx))
        mean_d = np.where(ok, sum_d / count, np.nan)
        jitter = mean_d * RAD2DEG
        var = np.where(ok & (count > 2), sum_d2 / count - mean_d ** 2, np.nan)
        sdev = np.sqrt(np.clip(var, 0.0, None)) * RAD2DEG

    if folds_at_90:
        mean = mean * 0.5
        jitter = jitter * 0.5
        sdev = sdev * 0.5

    return mean, sdev, jitter, mean_xx, mean_yy


def gate_valid(mean, sdev, jitter, snr, rhohv, zdr_sdev, p: KdpParams) -> np.ndarray:
    """Port of ``_isGateValid``.  Paper section 3 / Table 2."""
    valid = np.isfinite(mean)
    if p.check_snr:
        valid &= np.isfinite(snr) & (snr >= p.snr_threshold)
    # NaN sdev/jitter pass, matching the C++ where missing is -9999
    valid &= ~(sdev > p.phidp_sdev_max)
    valid &= ~(jitter > p.phidp_jitter_max)
    if p.check_zdr_sdev:
        valid &= ~(zdr_sdev > p.zdr_sdev_max)
    if p.check_rhohv:
        valid &= ~(np.isfinite(rhohv) & (rhohv < p.rhohv_threshold))
    return valid


def find_valid_runs(is_valid: np.ndarray, n_stats: int):
    """Port of ``_findValidRuns``.

    Returns ``(valid_runs, gaps, first_gate, last_gate, for_unfold, for_kdp)``
    or ``None`` when the ray has no usable region.
    """
    n = is_valid.size
    half = n_stats // 2

    # Pass 1 - runs strictly longer than the kernel
    runs: list[list[int]] = []
    run_len = 0
    for i in range(n):
        if is_valid[i]:
            run_len += 1
            if i == n - 1 and run_len > n_stats:
                runs.append([i - run_len + 1, i])
        else:
            if run_len > n_stats:
                runs.append([i - run_len, i - 1])
            run_len = 0

    # Pass 2 - merge runs separated by a gap of at most half a kernel
    merged: list[list[int]] = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= half:
            merged[-1][1] = run[1]
        else:
            merged.append(list(run))

    # Pass 3 - keep long runs only, and trim half a kernel off each end
    valid_runs = [[a + half, b - half] for a, b in merged
                  if (b - a + 1) >= 2 * n_stats]
    if not valid_runs:
        return None

    gaps = [(valid_runs[i - 1][1] + 1, valid_runs[i][0] - 1)
            for i in range(1, len(valid_runs))]

    first_gate = valid_runs[0][0] + 2
    last_gate = valid_runs[-1][1] - 2

    for_unfold = np.zeros(n, dtype=bool)
    for_kdp = np.zeros(n, dtype=bool)
    for a, b in valid_runs:
        for_unfold[a:b + 1] = True
        for_kdp[a:b + 1] = True

    # Bridge a gap that is shorter than both of its neighbours
    for i, (gs, ge) in enumerate(gaps):
        gap_len = ge - gs + 1
        prev_len = valid_runs[i][1] - valid_runs[i][0] + 1
        next_len = valid_runs[i + 1][1] - valid_runs[i + 1][0] + 1
        if prev_len > gap_len and next_len > gap_len:
            for_kdp[gs:ge + 1] = True

    return valid_runs, gaps, first_gate, last_gate, for_unfold, for_kdp


def unfold_phidp(phidp, mean, mean_xx, mean_yy, gaps, first_gate, last_gate,
                 for_unfold, fold_val, n_stats):
    """Port of the unfolding half of ``_unfoldPhidp``.  Paper section 4.

    Returns ``(phidp_mean_unfold, phidp_unfold)``.
    """
    n = phidp.size
    fold_range = 2.0 * fold_val

    mean_v = mean.copy()
    xx = mean_xx.copy()
    yy = mean_yy.copy()

    # Fill each gap from its two edges, meeting in the middle
    for gs, ge in gaps:
        mid = (gs + ge) // 2
        mean_v[gs:mid] = mean_v[gs - 1]
        xx[gs:mid] = mean_xx[gs - 1]
        yy[gs:mid] = mean_yy[gs - 1]
        mean_v[mid:ge + 1] = mean_v[ge + 1]
        xx[mid:ge + 1] = mean_xx[ge + 1]
        yy[mid:ge + 1] = mean_yy[ge + 1]

    # Walk the mean field, counting folds.  A fold is: both vectors in the
    # left half-plane, y changing sign (paper Fig. 4).
    unfold_mean = np.full(n, np.nan)
    sum_fold = 0
    for i in range(first_gate, last_gate + 1):
        if xx[i - 1] < 0 and xx[i] < 0:
            if yy[i - 1] < 0 and yy[i] > 0:
                sum_fold -= 1
            elif yy[i - 1] > 0 and yy[i] < 0:
                sum_fold += 1
        if np.isfinite(mean_v[i]):
            unfold_mean[i] = mean_v[i] + sum_fold * fold_range

    # Straight line across each gap
    for gs, ge in gaps:
        before = unfold_mean[gs - 1]
        after = unfold_mean[ge + 1]
        npts = ge - gs + 1
        step = (after - before) / npts
        unfold_mean[gs:ge + 1] = before + step * np.arange(1, npts + 1)

    unfold_mean[:first_gate] = unfold_mean[first_gate]
    unfold_mean[last_gate + 1:] = unfold_mean[last_gate]

    # Unfold the raw PHIDP onto the unfolded mean
    phidp_unfold = np.full(n, np.nan)
    sl = slice(first_gate, last_gate + 1)
    raw = phidp[sl]
    ref = unfold_mean[sl]
    use_raw = for_unfold[sl] & np.isfinite(raw)
    diff = ref - np.where(use_raw, raw, 0.0)
    nfold = np.trunc(np.abs(diff) / fold_range + 0.5) * np.sign(diff)
    phidp_unfold[sl] = np.where(use_raw, raw + nfold * fold_range, ref)

    mean_at_start = np.nanmean(unfold_mean[first_gate:first_gate + n_stats])
    mean_at_end = np.nanmean(unfold_mean[last_gate - n_stats + 1:last_gate + 1])
    phidp_unfold[:first_gate] = mean_at_start
    phidp_unfold[last_gate + 1:] = mean_at_end

    return unfold_mean, phidp_unfold


def fir_filter(x: np.ndarray, coeffs: np.ndarray, n_iter: int) -> np.ndarray:
    """Port of ``_applyFirFilter`` + ``_padArray``, iterated ``n_iter`` times.

    The C++ pads by ``_firLength`` gates, then each pass rewrites indices
    ``[-firLenHalf, nGates+firLenHalf)`` only - so later passes read a mix of
    filtered values (inner) and the original edge padding (outer).  This
    reproduces that exactly rather than approximating it.
    """
    m = coeffs.size
    half = m // 2
    n = x.size
    pad = m

    buf = np.empty(n + 2 * pad)
    buf[pad:pad + n] = x
    buf[:pad] = x[0]
    buf[pad + n:] = x[-1]

    for _ in range(n_iter):
        # corr[k] = sum_j buf[k+j]*coeffs[j];  out[ii] uses k = ii - half + pad
        corr = np.correlate(buf, coeffs, mode="valid")
        out = buf.copy()
        out[pad - half:pad + n + half] = corr[pad - 2 * half:pad + n]
        buf = out

    return buf[pad:pad + n]


def smooth_phidp(x: np.ndarray, dr: float, p: KdpParams) -> np.ndarray:
    """Filtered PHIDP, section 5.  ``fir`` is the LROSE path."""
    if p.filter == "fir":
        return fir_filter(x, FIR_COEFFS[p.fir_len], p.n_filt_iter_unfolded)
    if p.filter == "savgol":
        # Quadratic local fit over one feature length - Wang & Chandrasekar
        # (2009) in spirit, and the only option here where L_f is set directly.
        win = max(5, int(round(p.lf_km / dr)) | 1)
        return savgol_filter(x, window_length=min(win, x.size - 1 | 1), polyorder=2)
    raise ValueError(f"unknown filter: {p.filter!r}")


def phidp_conditioned(phidp_filt: np.ndarray, valid_runs) -> np.ndarray:
    """Port of ``_computePhidpConditioned`` - the LROSE peak-removal PSOB method.

    Flattens each up-then-down excursion back to the level it started from, by
    looking back for the last gate below the excursion's base.  This is the step
    the paper's segmented self-consistency replaces; it is kept because
    ``_loadKdp`` computes the LROSE KDP from its output.
    """
    cond = phidp_filt.copy()

    for run_begin, run_end in valid_runs:
        increasing = False
        decreasing = False
        prev_diff = 0.0
        top_index = -1
        top_indices: list[int] = []
        bot_indices: list[int] = []

        for i in range(run_begin + 1, run_end + 1):
            diff = phidp_filt[i] - phidp_filt[i - 1]

            if diff > 0 and prev_diff > 0:
                if not increasing:
                    increasing = True
                    if top_index > 0:
                        top_indices.append(top_index)
                        bot_indices.append(i - 2)
            else:
                increasing = False

            if diff < 0 and prev_diff < 0:
                if not decreasing:
                    top_index = i - 2
                    decreasing = True
            else:
                decreasing = False

            prev_diff = diff

        prev_bot = 0
        for top, bot in zip(top_indices, bot_indices):
            bot_val = phidp_filt[bot]
            matched = False
            for i in range(top, prev_bot - 1, -1):
                if phidp_filt[i] < bot_val:
                    cond[i + 1:bot] = bot_val
                    matched = True
                    break
            if not matched and prev_bot > 0:
                prev_bot_val = phidp_filt[prev_bot]
                for i in range(prev_bot + 1, bot + 1):
                    if phidp_filt[i] <= prev_bot_val:
                        cond[prev_bot + 1:i] = prev_bot_val
                        break
            prev_bot = bot

    return cond


def kdp_from_slope(phidp_cond_filt, dbz_max, snr, dr, snr_threshold):
    """Port of ``_loadKdp`` - adaptive-length slope of conditioned PHIDP.

    The half-width is 8, 4 or 2 gates as the surrounding max reflectivity is
    below 20 dBZ, below 35 dBZ, or above.
    """
    n = phidp_cond_filt.size
    idx = np.arange(n)
    dmax = np.where(np.isfinite(dbz_max), dbz_max, -9999.0)

    slopes = {}
    for adap in (2, 4, 8):
        i0 = np.clip(idx - adap, 0, n - 1)
        i1 = np.clip(idx + adap, 0, n - 1)
        span = i1 - i0
        with np.errstate(invalid="ignore", divide="ignore"):
            s = (phidp_cond_filt[i1] - phidp_cond_filt[i0]) / (dr * np.maximum(span, 1)) / 2.0
        slopes[adap] = np.where(span >= 1, s, 0.0)

    kdp = np.where(dmax < 20.0, slopes[8],
                   np.where(dmax < 35.0, slopes[4], slopes[2]))
    kdp = np.where(np.isfinite(snr) & (snr >= snr_threshold), kdp, np.nan)
    return kdp


def kdp_from_z_zdr(dbz, zdr, a, b, c):
    """Self-consistency KDP, Eq. (3)/(11).

    ZDR is floored at 0.1 dB as in ``_computeKdpFromZZdr``.  Gates with no
    reflectivity return 0, matching the C++ where ``10**(-999.9) -> 0``.
    """
    z_lin = np.power(10.0, np.where(np.isfinite(dbz), dbz, -9999.0) / 10.0)
    zdr_lin = np.power(10.0, np.maximum(np.where(np.isfinite(zdr), zdr, 0.1), 0.1) / 10.0)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        out = a * np.power(z_lin, b) * np.power(zdr_lin, c)
    return np.where(np.isfinite(out), out, 0.0)


def attenuation_correction(kdp, dbz, zdr, dr, band):
    """One-pass attenuation correction, Eq. (5)-(8).  Returns ``(dbz, zdr)``.

    DEVIATION: the paper's Eq. (7)/(8) carry the two-way factor of 2 that
    ``_computeAttenCorrection`` omits.  We follow the paper.  At S-band the
    difference is ~0.03 dB/km at KDP=2, i.e. irrelevant; at C- and X-band it
    is not.
    """
    a_h, b_h, a_dp, b_dp = ATTEN_COEFFS[band]
    k = np.clip(np.where(np.isfinite(kdp), kdp, 0.0), 0.0, 20.0)
    ah = a_h * np.power(k, b_h)
    adp = a_dp * np.power(k, b_dp)
    return (dbz + 2.0 * np.cumsum(ah) * dr,
            zdr + 2.0 * np.cumsum(adp) * dr)


def phidp_from_kdp(kdp_field, dr, anchor_phidp=None):
    r"""Integrate a KDP profile back to accumulated PHIDP, Eq. (2) reversed::

        PHIDP(r) = PHIDP(r0) + 2 * \int_{r0}^{r} KDP(s) ds

    i.e. ``2 * dr * np.cumsum(kdp)``, with gates that have no KDP contributing
    nothing.  The profile is anchored so it starts at the first gate carrying
    KDP, at the level of ``anchor_phidp`` there (pass ``phidp_filt``), or at
    zero if no anchor is given.

    This is *not* the same as ``phidp_sc`` from :func:`kdp_sc_segmented`.  That
    one is re-anchored to the measured PHIDP at the start of every segment, so
    it cannot drift by construction.  This one integrates the whole ray in one
    go, so any inconsistency between the KDP field and the measured phase
    accumulates - which is exactly what makes it a useful check.
    """
    kdp_field = np.asarray(kdp_field, dtype=float)
    valid = np.isfinite(kdp_field)
    out = np.full(kdp_field.shape, np.nan)
    if not valid.any():
        return out

    first = int(np.argmax(valid))
    last = int(valid.size - 1 - np.argmax(valid[::-1]))

    accum = 2.0 * dr * np.cumsum(np.nan_to_num(kdp_field, nan=0.0))
    out[first:last + 1] = accum[first:last + 1] - accum[first]

    if anchor_phidp is not None:
        runs = _contiguous(valid & np.isfinite(anchor_phidp))
        if runs:
            j = max(runs, key=lambda ab: ab[1] - ab[0])[0]
            out += anchor_phidp[j] - out[j]

    return out


# --------------------------------------------------------------------------
# Stage 2 - self-consistency
# --------------------------------------------------------------------------

def kdp_sc_runs(kdp, kdp_zzdr, kdp_min):
    """**Current LROSE master.**  Port of ``_loadKdpSC`` / ``_loadKdpSCRun``.

    Segments are runs of ``kdp > kdp_min``.  Within each, the Z-ZDR estimate is
    rescaled by ``sum(kdp) / sum(kdp_zzdr)``.  No endpoint constraint, no
    monotonic segmentation, no delta.
    """
    n = kdp.size
    out = np.full(n, np.nan)
    positive = np.isfinite(kdp) & (kdp > kdp_min) & np.isfinite(kdp_zzdr)

    for start, end in _contiguous(positive):
        if end - start < 3:
            continue
        sum_zzdr = kdp_zzdr[start:end + 1].sum()
        if sum_zzdr < 0.5:
            continue
        factor = kdp[start:end + 1].sum() / sum_zzdr
        out[start:end + 1] = kdp_zzdr[start:end + 1] * factor

    return out


def segment_dividers(phidp_filt, run_begin, run_end, min_seg_gates,
                     max_seg_gates=None):
    """Segment boundaries for section 6.3.

    A divider goes wherever the trend in filtered PHIDP turns from negative to
    positive - i.e. at each local minimum - so every segment holds one
    increasing-then-decreasing excursion.  Dividers closer together than
    ``min_seg_gates`` are dropped; the paper does not specify a floor, but
    without one the noise in a flat profile generates spurious one-gate
    segments.

    ADDITION (not in the paper): segments longer than ``max_seg_gates`` are
    subdivided into equal pieces.  A long monotonic stretch of PHIDP contains
    no increasing-then-decreasing excursion, so there is no delta in it to
    correct - but the endpoint-constrained reconstruction will still happily
    move phase from one end of it to the other, and the residual then shows up
    as tens of degrees of spurious "delta".  Subdividing keeps each
    reconstruction local, which is the stated point of the method.  Each piece
    remains independently endpoint-constrained, so PHIDP_sc stays continuous
    and still matches PHIDP_f at every divider.
    """
    diff = np.diff(phidp_filt[run_begin:run_end + 1])
    if diff.size < 2:
        return [run_begin, run_end]

    minima = np.nonzero((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1 + run_begin

    bounds = [run_begin]
    for pos in minima:
        if pos - bounds[-1] >= min_seg_gates:
            bounds.append(int(pos))
    if len(bounds) > 1 and run_end - bounds[-1] < min_seg_gates:
        bounds.pop()
    bounds.append(run_end)

    if not max_seg_gates:
        return bounds

    subdivided = [bounds[0]]
    for nxt in bounds[1:]:
        start = subdivided[-1]
        span = nxt - start
        if span > max_seg_gates:
            n_pieces = int(np.ceil(span / max_seg_gates))
            for k in range(1, n_pieces):
                subdivided.append(start + int(round(span * k / n_pieces)))
        subdivided.append(nxt)
    return subdivided


def kdp_sc_segmented(phidp_filt, kdp_star, kdp_fallback, for_kdp, dr, p: KdpParams):
    """**The paper's method.**  Endpoint-constrained local self-consistency,
    sections 6.3 and Eq. (12)-(14).

    For each segment between consecutive local minima of the filtered PHIDP:

    * the Z-ZDR relation sets the *shape* of the phase increase,
    * the filtered PHIDP at the two endpoints sets its *total*, Eq. (13),
    * delta is the positive residual, Eq. (14), and the segment is only
      accepted if the mean residual exceeds ``delta_threshold``.

    Returns ``(kdp_sc, phidp_sc, delta, dividers)``.  Rejected segments keep
    ``phidp_filt`` and fall back to ``kdp_fallback``.
    """
    n = phidp_filt.size
    min_seg = max(3, int(round((p.min_seg_km or p.lf_km) / dr)))
    max_seg = int(round(p.max_seg_km / dr)) if p.max_seg_km else None

    kdp_sc = kdp_fallback.copy()
    phidp_sc = phidp_filt.copy()
    delta = np.zeros(n)
    dividers: list[int] = []

    for run_begin, run_end in _contiguous(for_kdp):
        if run_end - run_begin + 1 < 2 * min_seg:
            continue

        bounds = segment_dividers(phidp_filt, run_begin, run_end, min_seg, max_seg)
        dividers.extend(bounds[1:-1])

        for i1, i2 in zip(bounds[:-1], bounds[1:]):
            sl = slice(i1, i2 + 1)

            # KDP_sc* from the Z-ZDR relation, Eq. (11)
            kstar = np.clip(np.nan_to_num(kdp_star[sl], nan=0.0), 0.0, None)
            integ = cumulative_trapezoid(kstar, dx=dr, initial=0.0)
            sc_phase = 2.0 * integ[-1]

            # Endpoint constraint, Eq. (12)
            dphi = phidp_filt[i2] - phidp_filt[i1]

            # Guards the paper leaves implicit: a segment whose far endpoint
            # sits below its near one would flip the sign of KDP_sc, and a
            # vanishing SC integral makes the scale factor meaningless.
            if dphi <= p.min_delta_phi or sc_phase < p.min_sc_phase:
                continue

            scale = dphi / sc_phase                      # Eq. (13)
            seg_phidp = phidp_filt[i1] + 2.0 * scale * integ
            seg_delta = np.clip(phidp_filt[sl] - seg_phidp, 0.0, None)   # Eq. (14)

            if seg_delta.mean() > p.delta_threshold:
                kdp_sc[sl] = kstar * scale
                phidp_sc[sl] = seg_phidp
                delta[sl] = seg_delta

    return kdp_sc, phidp_sc, delta, sorted(set(dividers))


# --------------------------------------------------------------------------
# Ray driver
# --------------------------------------------------------------------------

@dataclass
class KdpRay:
    """Everything the pipeline produced for one ray."""
    valid: np.ndarray
    for_kdp: np.ndarray
    phidp_mean: np.ndarray
    phidp_sdev: np.ndarray
    phidp_jitter: np.ndarray
    phidp_unfold: np.ndarray
    phidp_filt: np.ndarray
    phidp_cond_filt: np.ndarray
    phidp_sc: np.ndarray
    phidp_recon_kdp: np.ndarray
    phidp_recon_sc: np.ndarray
    kdp: np.ndarray
    kdp_zzdr: np.ndarray
    kdp_sc: np.ndarray
    kdp_sc_lrose: np.ndarray
    delta: np.ndarray
    dbz_corr: np.ndarray
    zdr_corr: np.ndarray
    dividers: list[int] = field(default_factory=list)


def _empty_ray(n: int) -> KdpRay:
    nan = lambda: np.full(n, np.nan)
    return KdpRay(
        valid=np.zeros(n, dtype=bool), for_kdp=np.zeros(n, dtype=bool),
        phidp_mean=nan(), phidp_sdev=nan(), phidp_jitter=nan(),
        phidp_unfold=nan(), phidp_filt=nan(), phidp_cond_filt=nan(),
        phidp_sc=nan(), phidp_recon_kdp=nan(), phidp_recon_sc=nan(),
        kdp=nan(), kdp_zzdr=nan(), kdp_sc=nan(),
        kdp_sc_lrose=nan(), delta=nan(), dbz_corr=nan(), zdr_corr=nan(),
    )


def process_ray(dbz, zdr, rhohv, phidp, snr, dr, wavelength_cm,
                p: KdpParams | None = None, diagnostics: bool = True) -> KdpRay:
    """Run Stage 1 + Stage 2 on a single ray.

    ``dr`` is the gate spacing in km.  PHIDP must already be on a symmetric
    interval - call :func:`normalize_phidp_range` on the sweep first.

    With ``diagnostics=False`` the comparison-only products (``kdp_sc_lrose``,
    ``phidp_recon_kdp``, ``phidp_recon_sc``) are skipped and left NaN; the
    pipeline only keeps ``phidp_sc`` and ``kdp_sc``.
    """
    p = p or KdpParams()
    n = phidp.size
    n_stats = p.n_gates_stats
    band = band_from_wavelength(wavelength_cm)
    a, b, c = sc_coeffs(wavelength_cm, p.sc_coeff_set)

    out = _empty_ray(n)

    fold_val, folds_at_90 = folding_range(phidp)
    mean, sdev, jitter, mean_xx, mean_yy = phidp_stats(phidp, n_stats, folds_at_90)
    out.phidp_mean, out.phidp_sdev, out.phidp_jitter = mean, sdev, jitter

    zdr_sdev = np.full(n, np.nan)
    if p.check_zdr_sdev:
        cnt = _win_sum(np.isfinite(zdr).astype(float), n_stats)
        z0 = np.nan_to_num(zdr)
        s1 = _win_sum(z0, n_stats)
        s2 = _win_sum(z0 * z0, n_stats)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = s2 / cnt - (s1 / cnt) ** 2
        zdr_sdev = np.where(cnt > max(2, n_stats // 2),
                            np.sqrt(np.clip(var, 0, None)), np.nan)

    out.valid = gate_valid(mean, sdev, jitter, snr, rhohv, zdr_sdev, p)

    runs = find_valid_runs(out.valid, n_stats)
    if runs is None:
        return out
    valid_runs, gaps, first_gate, last_gate, for_unfold, for_kdp = runs
    out.for_kdp = for_kdp

    unfold_mean, out.phidp_unfold = unfold_phidp(
        phidp, mean, mean_xx, mean_yy, gaps, first_gate, last_gate,
        for_unfold, fold_val, n_stats)

    # Filtered PHIDP - the profile the paper segments, and the LROSE PHIDP_F
    out.phidp_filt = smooth_phidp(_fill_nan(unfold_mean), dr, p)

    # LROSE peak removal, then a second round of filtering, then the slope
    cond = phidp_conditioned(out.phidp_filt, valid_runs)
    if p.filter == "fir":
        out.phidp_cond_filt = fir_filter(cond, FIR_COEFFS[p.fir_len], p.n_filt_iter_cond)
    else:
        out.phidp_cond_filt = smooth_phidp(cond, dr, p)

    dbz_max = maximum_filter1d(np.where(np.isfinite(dbz), dbz, -9999.0),
                               size=n_stats, mode="nearest")
    out.kdp = kdp_from_slope(out.phidp_cond_filt, dbz_max, snr, dr, p.snr_threshold)

    # Z and ZDR for the self-consistency relation: 5-gate median, optionally
    # attenuation-corrected (Stage 3 hook, off by default)
    dbz_med = _running_nanmedian(dbz, p.median_len_zzdr)
    zdr_med = _running_nanmedian(zdr, p.median_len_zzdr)
    if p.atten_correct:
        dbz_med, zdr_med = attenuation_correction(out.kdp, dbz_med, zdr_med, dr, band)
    out.dbz_corr, out.zdr_corr = dbz_med, zdr_med

    # The Z-ZDR relation of Eq. (3) is a rain relation.  Hail breaks it badly:
    # high Z with low ZDR reads as enormous KDP (c = -2.05 makes Z_DR**c blow
    # up), so the reconstruction dumps a whole segment's phase into the hail
    # core.  Capping Z is the usual mitigation; it is not in the paper, so it
    # is off by default.
    if p.dbz_max_for_sc is not None:
        dbz_med = np.minimum(dbz_med, p.dbz_max_for_sc)

    out.kdp_zzdr = kdp_from_z_zdr(dbz_med, zdr_med, a, b, c)
    out.kdp_zzdr = np.where(np.isfinite(out.kdp), out.kdp_zzdr, np.nan)

    # The two self-consistency variants
    if diagnostics:
        out.kdp_sc_lrose = kdp_sc_runs(out.kdp, out.kdp_zzdr, p.kdp_min_for_sc)
    out.kdp_sc, out.phidp_sc, out.delta, out.dividers = kdp_sc_segmented(
        out.phidp_filt, out.kdp_zzdr, out.kdp, for_kdp, dr, p)

    # Report KDP/delta only where the ray supports it.
    #
    # DEVIATION: _loadKdp gates KDP on SNR alone, so lrose emits KDP wherever
    # SNR clears the threshold - which in a 372 km ray means speckle out to
    # max range.  KdpFilt.cc:520-530 has exactly this mask, commented out.
    # The paper's PPIs are plainly masked, and leaving it off would make the
    # KDP / KDP_sc comparison unfair, since KDP_sc is masked either way.
    out.kdp = np.where(for_kdp, out.kdp, np.nan)
    out.kdp_zzdr = np.where(for_kdp, out.kdp_zzdr, np.nan)
    out.kdp_sc = np.where(for_kdp, out.kdp_sc, np.nan)
    out.kdp_sc_lrose = np.where(for_kdp, out.kdp_sc_lrose, np.nan)
    out.delta = np.where(for_kdp, out.delta, np.nan)

    # Outside [first_gate, last_gate] the PHIDP fields are edge padding that
    # _unfoldPhidp puts there for the filter to chew on, not measurements.
    # The C++ leaves them in its output arrays; we mask them so they cannot be
    # mistaken for data in a PPI.
    outside = np.ones(n, dtype=bool)
    outside[first_gate:last_gate + 1] = False
    for name in ("phidp_unfold", "phidp_filt", "phidp_cond_filt", "phidp_sc"):
        getattr(out, name)[outside] = np.nan

    # PHIDP rebuilt by integrating each KDP field over the whole ray, anchored
    # to the measured level at the first gate with KDP.  Unlike phidp_sc these
    # are free to drift away from the measurement.
    if diagnostics:
        out.phidp_recon_kdp = phidp_from_kdp(out.kdp, dr, out.phidp_filt)
        out.phidp_recon_sc = phidp_from_kdp(out.kdp_sc, dr, out.phidp_filt)

    return out


# --------------------------------------------------------------------------
# Py-ART volume driver
# --------------------------------------------------------------------------

# OPOL is a C-band radar (5.625 GHz).  Only used when the volume carries no
# frequency: the wavelength picks the band (attenuation coefficients) and scales
# the self-consistency "a" coefficient, which cancels under the endpoint
# constraint of Eq. (13).
DEFAULT_WAVELENGTH_CM = 5.33


def radar_wavelength_cm(radar, default: float = DEFAULT_WAVELENGTH_CM) -> float:
    """Wavelength (cm) from ``radar.instrument_parameters['frequency']``, else ``default``."""
    try:
        freq = float(np.ravel(radar.instrument_parameters["frequency"]["data"])[0])
        if np.isfinite(freq) and freq > 0:
            return 2.99792458e10 / freq
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        pass
    return default


def _filled(a) -> np.ndarray:
    return np.ma.filled(np.ma.masked_invalid(np.ma.asarray(a, dtype="float64")), np.nan)


def retrieve_phidp_kdp(radar, phidp, dbz, zdr, rhohv, snr, gatefilter=None,
                       params: KdpParams | None = None, wavelength_cm: float | None = None,
                       remove_system_phase: bool = True):
    """Corrected PHIDP and KDP for a whole Py-ART volume, sweep by sweep.

    Runs :func:`process_ray` on every ray and returns a single PHIDP and a
    single KDP: the Dixon & Hubbert (2026) segmented self-consistency products: ``phidp_sc`` (filtered, unfolded PHIDP
    with the backscatter phase removed) and ``kdp_sc`` (the LROSE slope KDP
    wherever a segment fails the delta test).

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure (for the sweep slices, range and frequency).
    phidp, dbz, zdr, rhohv, snr : array-like
        2D (nrays, ngates) raw PHIDP (deg), calibrated reflectivity (dBZ),
        calibrated ZDR (dB), RHOHV and SNR (dB).  Masked or non-finite values
        are treated as missing.
    gatefilter : pyart.filters.GateFilter, optional
        Gates it excludes are removed from the PHIDP input and set to NaN in
        both outputs.
    params : KdpParams, optional
        Algorithm tunables (defaults: ``KdpParams()``).
    wavelength_cm : float, optional
        Radar wavelength; read from the volume when omitted.
    remove_system_phase : bool
        Subtract the system differential phase, so PHIDP starts from zero.
        The offset is the median, over the rays of a sweep, of PHIDP_sc at the
        first gate of each ray's valid region.  Required by the downstream
        Z-PHI and ZDR attenuation corrections, which use PHIDP as accumulated
        phase.

    Returns
    -------
    phidp_corr : np.ndarray
        2D float64 corrected PHIDP (deg), NaN where undefined.
    kdp : np.ndarray
        2D float64 KDP (deg/km), NaN where undefined.
    info : dict
        ``system_phase`` (per-sweep offset removed, deg), ``wavelength_cm``.
    """
    p = params or KdpParams()
    wavelength_cm = wavelength_cm or radar_wavelength_cm(radar)

    rng_km = np.asarray(radar.range["data"], dtype="float64") / 1000.0
    dr = float(np.median(np.diff(rng_km)))

    phidp = _filled(phidp)
    dbz, zdr, rhohv, snr = (_filled(x) for x in (dbz, zdr, rhohv, snr))
    excluded = None
    if gatefilter is not None:
        excluded = np.asarray(gatefilter.gate_excluded, dtype=bool)
        phidp[excluded] = np.nan

    phidp_out = np.full(phidp.shape, np.nan)
    kdp_out = np.full(phidp.shape, np.nan)
    system_phase = []

    for sweep in range(radar.nsweeps):
        sl = radar.get_slice(sweep)
        ph = normalize_phidp_range(phidp[sl])
        d, z, rh, sn = dbz[sl], zdr[sl], rhohv[sl], snr[sl]

        phi_sw = np.full(ph.shape, np.nan)
        kdp_sw = np.full(ph.shape, np.nan)
        starts = []
        for i in range(ph.shape[0]):
            ray = process_ray(d[i], z[i], rh[i], ph[i], sn[i], dr, wavelength_cm, p, diagnostics=False)
            phi_sw[i] = ray.phidp_sc
            kdp_sw[i] = ray.kdp_sc
            finite = np.isfinite(ray.phidp_sc)
            if ray.for_kdp.any() and finite.any():
                starts.append(ray.phidp_sc[np.argmax(finite)])

        offset = float(np.median(starts)) if (remove_system_phase and starts) else 0.0
        system_phase.append(offset)
        phidp_out[sl] = phi_sw - offset
        kdp_out[sl] = kdp_sw

    if excluded is not None:
        phidp_out[excluded] = np.nan
        kdp_out[excluded] = np.nan

    return phidp_out, kdp_out, {"system_phase": system_phase, "wavelength_cm": wavelength_cm}
