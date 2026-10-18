"""Endpoint-constrained self-consistency KDP, for Py-ART radar objects.

Implements the method of Dixon & Hubbert, *A robust, general-purpose KDP
algorithm for LROSE. A constrained local self-consistency approach for S-, C-
and X-band radars*, extended abstract, ERAD 2026.

The retrieval has two halves:

* **PHIDP conditioning** (paper sections 3-5) - identify gates with usable
  PHIDP from its spatial standard deviation, gate-to-gate jitter, SNR and
  RHOHV; unfold in vector space; filter in range with the Hubbert & Bringi FIR
  filter.  This half is a port of ``KdpFilt.cc`` from NCAR/lrose-core
  (``codebase/libs/radar/src/kdp``, HEAD 3871d925c, 2026-09-21).

* **Segmented self-consistency** (sections 6.1-6.4) - split the filtered PHIDP
  profile at each point where the trend turns from decreasing to increasing,
  and within each segment let the ZH-ZDR relation set the *shape* of the phase
  increase while the filtered PHIDP at the two endpoints sets its *total*.
  This half is **not** in lrose master, which normalises over runs of positive
  KDP with no segmentation and no endpoint constraint.

Usage::

    import pyart, kdp_opol

    radar = pyart.io.read(...)
    kdp_field, phidp_field = kdp_opol.retrieve(radar, wavelength_cm=5.5)
    radar.add_field("kdp_sc", kdp_field)
    radar.add_field("phidp_sc", phidp_field)

    # or, in one step
    kdp_opol.add_fields(radar, wavelength_cm=5.5)

Returned PHIDP is the accumulated differential propagation phase, re-zeroed
per ray at the first usable gate, so it starts at 0 and the system offset is
removed.

Two departures from the paper are deliberate and documented at their call
sites: ``KdpParams.max_seg_km`` caps segment length (the paper sets no ceiling,
which misbehaves on long rays with coarse gates), and attenuation correction
carries the two-way factor of 2 of Eq. (7)/(8) that ``_computeAttenCorrection``
omits.  The ZH-ZDR relation is a *rain* relation and is not valid in hail; see
``KdpParams.dbz_max_for_sc``.
"""

from typing import Any
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import maximum_filter1d

__all__ = ["KdpParams", "retrieve", "add_fields", "process_ray", "band_from_wavelength"]

RAD2DEG = 180.0 / np.pi
_LIGHT_SPEED_CM_S = 2.99792458e10

# ---------------------------------------------------------------------------
# Coefficient tables
# ---------------------------------------------------------------------------

# Hubbert & Bringi FIR coefficients, verbatim from KdpFilt.cc:52-148.  Keyed by
# the LROSE name; FIR_LEN_10 has 11 taps.  Pick the length whose physical span
# (n_taps * gate_spacing) is nearest the feature length you want to preserve -
# the paper uses 3.5 km.
FIR_COEFFS: dict[int, np.ndarray] = {
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
}
# Self-consistency relation Eq. (3):  KDP = a * Z_H**b * Z_DR**c, linear units.
SC_COEFFS_PAPER: dict[str, tuple[float, float, float]] = {
    "S": (0.0001051, 0.96, -0.260),  # Scarchilli et al. (1996)
    "C": (0.0001461, 0.98, -0.200),  # Scarchilli et al. (1996)
    "X": (0.000222, 1.00, -4.58),  # Shi et al. (2018)
}

# Eq. (5)/(6):  A_H = aH*KDP**bH,  A_DP = aDP*KDP**bDP.  Paper Table 5 /
# Bringi & Chandrasekar Table 7.1.
ATTEN_COEFFS = {
    "S": (0.017, 0.84, 0.003, 1.05),
    "C": (0.073, 0.99, 0.013, 1.23),
    "X": (0.233, 1.02, 0.033, 1.15),
}


def band_from_wavelength(wavelength_cm: float) -> str:
    """Band letter, using the cut points of ``KdpFilt::compute``."""
    if wavelength_cm < 4:
        return "X"
    if wavelength_cm < 7:
        return "C"
    return "S"


def _sc_coeffs(wavelength_cm: float, which: str) -> tuple[float, float, float]:
    if which == "lrose":
        return 3.32e-5 * (10.0 / wavelength_cm), 1.0, -2.05
    if which == "paper":
        return SC_COEFFS_PAPER[band_from_wavelength(wavelength_cm)]
    raise ValueError(f"sc_coeff_set must be 'lrose' or 'paper', got {which!r}")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class KdpParams:
    """Tunables.  Defaults follow ``paramdef.KdpFilt`` and the paper's tables."""

    # --- usable-PHIDP test, section 3 / Table 2 ---
    n_gates_stats: int = 9  # computational kernel, gates
    phidp_sdev_max: float = 20.0  # deg
    phidp_jitter_max: float = 25.0  # deg (paper Table 2 says 30)
    check_snr: bool = True  # ignored when the radar has no SNR field
    snr_threshold: float = 0.0  # dB
    check_rhohv: bool = True
    rhohv_threshold: float = 0.85

    # --- range filtering, section 5 ---
    fir_len: int = 10  # key into FIR_COEFFS
    n_filt_iter_unfolded: int = 2
    n_filt_iter_cond: int = 4
    peak_removal: bool = True  # LROSE excursion flattening before the
    # KDP slope; set False to take the slope
    # straight off the filtered PHIDP and let
    # the self-consistency step handle delta

    # --- ZH-ZDR relation, section 6.1/6.2 ---
    median_len_zzdr: int = 5
    sc_coeff_set: str = "lrose"  # "lrose" or "paper"
    dbz_max_for_sc: float | None = None  # cap Z before Eq. (3); see note below

    # --- segmentation and acceptance, section 6.3 ---
    min_seg_km: float = 3.5  # the paper's feature length L_f
    max_seg_km: float | None = 10.0  # ADDITION, not in the paper
    min_delta_phi: float = 1.0  # deg; skip segments with dPHI below this
    min_sc_phase: float = 0.5  # deg; skip if the SC integral is tiny
    delta_threshold: float = 1.0  # deg; accept/reject on mean delta

    # --- attenuation and iteration, section 6.4 ---
    atten_correct: bool | None = None  # None -> on for C and X band, off for S
    n_sc_passes: int = 2  # section 6.4: two passes suffice


# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------


def _win_sum(a: np.ndarray, n: int) -> np.ndarray:
    """Centred sliding-window sum of length ``n``, clipped at the array ends.

    Clipping rather than padding reproduces the ``jj < 0 || jj >= _nGates:
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
    out[half : half + med.size] = med
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
    """Linear interpolation over non-finite values, with edge extension."""
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

    ``_computeFoldingRange`` does this per ray.  We do it once for the whole
    volume: per ray it is unstable, because a weak ray whose PHIDP happens to
    stay inside 0..180 is left unshifted and ends up 180 deg from its
    neighbours.
    """
    finite = phidp[np.isfinite(phidp)]
    if finite.size and finite.min() >= 0 and finite.max() > 180:
        return phidp - 180.0
    return phidp


# ---------------------------------------------------------------------------
# PHIDP conditioning - sections 3 to 5
# ---------------------------------------------------------------------------


def folding_range(phidp: np.ndarray) -> tuple[float, bool]:
    """``(fold_val, folds_at_90)`` - port of ``_computeFoldingRange``.

    Alternating-mode radars fold at -90/+90, simultaneous H/V at -180/+180.
    """
    finite = phidp[np.isfinite(phidp)]
    if finite.size == 0:
        return 180.0, False
    if finite.min() > -90 and finite.max() < 90:
        return 90.0, True
    return 180.0, False


def phidp_stats(phidp: np.ndarray, n_stats: int, folds_at_90: bool):
    """Circular mean, sdev and jitter of PHIDP - port of ``_computePhidpStats``.

    Returns ``(mean, sdev, jitter, mean_xx, mean_yy)``, NaN where the C++ would
    leave the value missing.  Both sdev and jitter come from the gate-to-gate
    chord length on the unit circle, treated as an angle in radians, which is
    what ``KdpFilt.cc`` does and what its 20/25 deg thresholds are tuned
    against.  (``PhidpProc.cc`` in the same library computes a true circular
    sdev instead; the two are not interchangeable.)
    """
    n = phidp.size
    half = n_stats // 2
    valid = np.isfinite(phidp)

    phase = np.where(valid, phidp, 0.0) * (2.0 if folds_at_90 else 1.0)
    xx = np.where(valid, np.cos(np.deg2rad(phase)), 0.0)
    yy = np.where(valid, np.sin(np.deg2rad(phase)), 0.0)

    dist = np.zeros(n)
    both = valid[1:] & valid[:-1]
    dist[1:] = np.where(both, np.hypot(np.diff(xx), np.diff(yy)), 0.0)

    count = _win_sum(valid.astype(float), n_stats)
    sum_xx = _win_sum(xx, n_stats)
    sum_yy = _win_sum(yy, n_stats)
    sum_d = _win_sum(dist, n_stats)
    sum_d2 = _win_sum(dist * dist, n_stats)

    ok = count > half
    idx = np.arange(n)
    ok &= (idx >= half) & (idx < n - half)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_xx = np.where(ok, sum_xx / count, np.nan)
        mean_yy = np.where(ok, sum_yy / count, np.nan)
        mean = np.degrees(np.arctan2(mean_yy, mean_xx))
        mean_d = np.where(ok, sum_d / count, np.nan)
        jitter = mean_d * RAD2DEG
        var = np.where(ok & (count > 2), sum_d2 / count - mean_d**2, np.nan)
        sdev = np.sqrt(np.clip(var, 0.0, None)) * RAD2DEG

    if folds_at_90:
        mean, jitter, sdev = mean * 0.5, jitter * 0.5, sdev * 0.5

    return mean, sdev, jitter, mean_xx, mean_yy


def gate_valid(mean, sdev, jitter, snr, rhohv, p: KdpParams, snr_available: bool = True) -> np.ndarray:
    """Port of ``_isGateValid``.  Section 3 / Table 2.

    With no SNR the test falls back on the other three features, as the C++
    does via its ``_checkSnr && _snrAvailable`` guard.  On the test radar this
    costs nothing: dropping SNR loses no gates and changes fewer than 0.3% of
    the rest.
    """
    valid = np.isfinite(mean)
    if p.check_snr and snr_available:
        valid &= np.isfinite(snr) & (snr >= p.snr_threshold)
    # NaN sdev/jitter pass, matching the C++ where missing is -9999
    valid &= ~(sdev > p.phidp_sdev_max)
    valid &= ~(jitter > p.phidp_jitter_max)
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

    # Pass 3 - keep long runs only, trim half a kernel off each end
    valid_runs = [[a + half, b - half] for a, b in merged if (b - a + 1) >= 2 * n_stats]
    if not valid_runs:
        return None

    gaps = [(valid_runs[i - 1][1] + 1, valid_runs[i][0] - 1) for i in range(1, len(valid_runs))]

    first_gate = valid_runs[0][0] + 2
    last_gate = valid_runs[-1][1] - 2

    for_unfold = np.zeros(n, dtype=bool)
    for_kdp = np.zeros(n, dtype=bool)
    for a, b in valid_runs:
        for_unfold[a : b + 1] = True
        for_kdp[a : b + 1] = True

    # Bridge a gap shorter than both of its neighbours
    for i, (gs, ge) in enumerate(gaps):
        gap_len = ge - gs + 1
        if (
            valid_runs[i][1] - valid_runs[i][0] + 1 > gap_len
            and valid_runs[i + 1][1] - valid_runs[i + 1][0] + 1 > gap_len
        ):
            for_kdp[gs : ge + 1] = True

    return valid_runs, gaps, first_gate, last_gate, for_unfold, for_kdp


def unfold_phidp(mean, mean_xx, mean_yy, gaps, first_gate, last_gate, fold_val) -> np.ndarray:
    """Unfolding half of ``_unfoldPhidp``.  Section 4.

    A fold is detected in vector space (paper Fig. 4): both the previous and
    current mean vectors in the left half-plane, with the y coordinate changing
    sign.  Returns the unfolded mean PHIDP, which is what the range filter then
    operates on.
    """
    n = mean.size
    fold_range = 2.0 * fold_val

    mean_v, xx, yy = mean.copy(), mean_xx.copy(), mean_yy.copy()

    # Fill each gap from its two edges, meeting in the middle
    for gs, ge in gaps:
        mid = (gs + ge) // 2
        mean_v[gs:mid] = mean_v[gs - 1]
        xx[gs:mid], yy[gs:mid] = mean_xx[gs - 1], mean_yy[gs - 1]
        mean_v[mid : ge + 1] = mean_v[ge + 1]
        xx[mid : ge + 1], yy[mid : ge + 1] = mean_xx[ge + 1], mean_yy[ge + 1]

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

    # Straight line across each gap, then constant beyond the usable region
    for gs, ge in gaps:
        before, after = unfold_mean[gs - 1], unfold_mean[ge + 1]
        npts = ge - gs + 1
        unfold_mean[gs : ge + 1] = before + ((after - before) / npts) * np.arange(1, npts + 1)

    unfold_mean[:first_gate] = unfold_mean[first_gate]
    unfold_mean[last_gate + 1 :] = unfold_mean[last_gate]
    return unfold_mean


def fir_filter(x: np.ndarray, coeffs: np.ndarray, n_iter: int) -> np.ndarray:
    """Port of ``_applyFirFilter`` + ``_padArray``, iterated ``n_iter`` times.

    The C++ pads by ``_firLength`` gates, then each pass rewrites indices
    ``[-firLenHalf, nGates+firLenHalf)`` only - so later passes read a mix of
    filtered values (inner) and the original edge padding (outer).  This
    reproduces that rather than approximating it.
    """
    m = coeffs.size
    half = m // 2
    n = x.size
    pad = m

    buf = np.empty(n + 2 * pad)
    buf[pad : pad + n] = x
    buf[:pad] = x[0]
    buf[pad + n :] = x[-1]

    for _ in range(n_iter):
        corr = np.correlate(buf, coeffs, mode="valid")
        out = buf.copy()
        out[pad - half : pad + n + half] = corr[pad - 2 * half : pad + n]
        buf = out

    return buf[pad : pad + n]


def phidp_conditioned(phidp_filt: np.ndarray, valid_runs) -> np.ndarray:
    """Port of ``_computePhidpConditioned`` - LROSE excursion flattening.

    Flattens each up-then-down excursion back to the level it started from by
    looking back for the last gate below the excursion's base.  This is the
    step the segmented self-consistency effectively replaces; it is kept
    because ``_loadKdp`` derives the LROSE KDP from its output, and that KDP is
    what the self-consistency step falls back to on rejected segments.  Set
    ``KdpParams.peak_removal = False`` to skip it.
    """
    cond = phidp_filt.copy()

    for run_begin, run_end in valid_runs:
        increasing = decreasing = False
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
                    cond[i + 1 : bot] = bot_val
                    matched = True
                    break
            if not matched and prev_bot > 0:
                prev_bot_val = phidp_filt[prev_bot]
                for i in range(prev_bot + 1, bot + 1):
                    if phidp_filt[i] <= prev_bot_val:
                        cond[prev_bot + 1 : i] = prev_bot_val
                        break
            prev_bot = bot

    return cond


def kdp_from_slope(phidp_cond_filt, dbz_max, snr, dr, snr_threshold, snr_available: bool = True) -> np.ndarray:
    """Port of ``_loadKdp`` - adaptive-length slope of conditioned PHIDP.

    The half-width is 8, 4 or 2 gates as the surrounding max reflectivity is
    below 20 dBZ, below 35 dBZ, or above.

    ``_loadKdp`` applies its SNR test unconditionally, unlike ``_isGateValid``
    which guards on ``_snrAvailable`` - so calling the C++ with ``snr = NULL``
    fills _snr with -9999, fails at every gate and returns an entirely missing
    field.  Here the test is skipped when SNR is unavailable.
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

    kdp = np.where(dmax < 20.0, slopes[8], np.where(dmax < 35.0, slopes[4], slopes[2]))
    if snr_available:
        kdp = np.where(np.isfinite(snr) & (snr >= snr_threshold), kdp, np.nan)
    return kdp


def kdp_from_z_zdr(dbz, zdr, a, b, c) -> np.ndarray:
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
    """One-way attenuation integrated to a two-way correction, Eq. (5)-(8).

    DEVIATION: the paper's Eq. (7)/(8) carry the two-way factor of 2 that
    ``_computeAttenCorrection`` omits.  We follow the paper.  At S-band the
    difference is negligible; at C- and X-band it is not.
    """
    a_h, b_h, a_dp, b_dp = ATTEN_COEFFS[band]
    k = np.clip(np.where(np.isfinite(kdp), kdp, 0.0), 0.0, 20.0)
    return (dbz + 2.0 * np.cumsum(a_h * np.power(k, b_h)) * dr, zdr + 2.0 * np.cumsum(a_dp * np.power(k, b_dp)) * dr)


# ---------------------------------------------------------------------------
# Segmented self-consistency - section 6.3
# ---------------------------------------------------------------------------


def segment_dividers(phidp_filt, run_begin, run_end, min_seg_gates, max_seg_gates=None) -> list[int]:
    """Segment boundaries for section 6.3.

    A divider goes wherever the trend in filtered PHIDP turns from negative to
    positive - each local minimum - so every segment holds one
    increasing-then-decreasing excursion.

    Two guards the paper does not specify.  Dividers closer together than
    ``min_seg_gates`` are dropped, without which noise in a flat profile
    generates spurious one-gate segments.  And segments longer than
    ``max_seg_gates`` are subdivided into equal pieces: a long monotonic
    stretch contains no excursion and so no delta to correct, but the
    endpoint-constrained reconstruction will still move phase from one end of
    it to the other, and the residual then appears as tens of degrees of
    spurious delta.  Each piece stays independently endpoint-constrained, so
    PHIDP_sc remains continuous and still matches PHIDP_f at every divider.
    """
    diff = np.diff(phidp_filt[run_begin : run_end + 1])
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
    """Endpoint-constrained local self-consistency, Eq. (12)-(14).

    For each segment between consecutive local minima of the filtered PHIDP:
    the ZH-ZDR relation sets the *shape* of the phase increase, the filtered
    PHIDP at the two endpoints sets its *total* (Eq. 13), delta is the positive
    residual (Eq. 14), and the segment is accepted only if the mean residual
    exceeds ``delta_threshold``.

    Returns ``(kdp_sc, phidp_sc, delta)``.  Rejected segments keep
    ``phidp_filt`` and fall back to ``kdp_fallback`` - which is most of the
    domain, so the fallback is not a corner case.
    """
    n = phidp_filt.size
    min_seg = max(3, int(round(p.min_seg_km / dr)))
    max_seg = int(round(p.max_seg_km / dr)) if p.max_seg_km else None

    kdp_sc = kdp_fallback.copy()
    phidp_sc = phidp_filt.copy()
    delta = np.zeros(n)

    for run_begin, run_end in _contiguous(for_kdp):
        if run_end - run_begin + 1 < 2 * min_seg:
            continue

        bounds = segment_dividers(phidp_filt, run_begin, run_end, min_seg, max_seg)

        for i1, i2 in zip(bounds[:-1], bounds[1:]):
            sl = slice(i1, i2 + 1)

            kstar = np.clip(np.nan_to_num(kdp_star[sl], nan=0.0), 0.0, None)
            integ = cumulative_trapezoid(kstar, dx=dr, initial=0.0)
            sc_phase = 2.0 * integ[-1]
            dphi = phidp_filt[i2] - phidp_filt[i1]

            # A segment whose far endpoint sits below its near one would flip
            # the sign of KDP_sc; a vanishing SC integral makes the scale
            # factor meaningless.  Neither guard is spelled out in the paper.
            if dphi <= p.min_delta_phi or sc_phase < p.min_sc_phase:
                continue

            scale = dphi / sc_phase  # Eq. (13)
            seg_phidp = phidp_filt[i1] + 2.0 * scale * integ
            seg_delta = np.clip(phidp_filt[sl] - seg_phidp, 0.0, None)  # Eq. (14)

            if seg_delta.mean() > p.delta_threshold:
                kdp_sc[sl] = kstar * scale
                phidp_sc[sl] = seg_phidp
                delta[sl] = seg_delta

    return kdp_sc, phidp_sc, delta


# ---------------------------------------------------------------------------
# Ray driver
# ---------------------------------------------------------------------------


def process_ray(
    dbz, zdr, rhohv, phidp, snr, dr, wavelength_cm, p: KdpParams | None = None, snr_available: bool | None = None
):
    """Run the full retrieval on one ray.

    ``dr`` is the gate spacing in km; PHIDP must already be on a symmetric
    interval (see :func:`normalize_phidp_range`).  ``snr`` may be ``None``.

    Returns ``(kdp_sc, phidp_sc, delta)``, all NaN outside the usable region.
    ``phidp_sc`` still carries the system offset here; :func:`retrieve`
    re-zeroes it.
    """
    p = p or KdpParams()
    n = phidp.size
    nan = lambda: np.full(n, np.nan)

    if snr is None:
        snr = np.full(n, np.nan)
    if snr_available is None:
        snr_available = bool(np.isfinite(snr).any())

    band = band_from_wavelength(wavelength_cm)
    a, b, c = _sc_coeffs(wavelength_cm, p.sc_coeff_set)
    do_atten = p.atten_correct if p.atten_correct is not None else band in ("C", "X")

    # --- section 3: usable PHIDP -----------------------------------------
    fold_val, folds_at_90 = folding_range(phidp)
    mean, sdev, jitter, mean_xx, mean_yy = phidp_stats(phidp, p.n_gates_stats, folds_at_90)
    valid = gate_valid(mean, sdev, jitter, snr, rhohv, p, snr_available)

    runs = find_valid_runs(valid, p.n_gates_stats)
    if runs is None:
        return nan(), nan(), nan()
    valid_runs, gaps, first_gate, last_gate, _, for_kdp = runs

    # --- sections 4-5: unfold and filter ---------------------------------
    unfold_mean = unfold_phidp(mean, mean_xx, mean_yy, gaps, first_gate, last_gate, fold_val)
    coeffs = FIR_COEFFS[p.fir_len]
    phidp_filt = fir_filter(_fill_nan(unfold_mean), coeffs, p.n_filt_iter_unfolded)

    # --- LROSE KDP, the fallback for rejected segments -------------------
    if p.peak_removal:
        cond = fir_filter(phidp_conditioned(phidp_filt, valid_runs), coeffs, p.n_filt_iter_cond)
    else:
        cond = phidp_filt
    dbz_max = maximum_filter1d(np.where(np.isfinite(dbz), dbz, -9999.0), size=p.n_gates_stats, mode="nearest")
    kdp = kdp_from_slope(cond, dbz_max, snr, dr, p.snr_threshold, snr_available)

    # --- section 6.4: self-consistency, iterated -------------------------
    # (a) attenuation from the filtered KDP, (b) KDP_sc, (c) attenuation again
    # from KDP_sc, (d) KDP_sc again.  Two passes suffice for the paper's cases.
    dbz_med = _running_nanmedian(dbz, p.median_len_zzdr)
    zdr_med = _running_nanmedian(zdr, p.median_len_zzdr)

    kdp_for_atten = kdp
    kdp_sc = phidp_sc = delta = None
    for _ in range(max(1, p.n_sc_passes)):
        if do_atten:
            dbz_c, zdr_c = attenuation_correction(kdp_for_atten, dbz_med, zdr_med, dr, band)
        else:
            dbz_c, zdr_c = dbz_med, zdr_med

        # The ZH-ZDR relation is a rain relation.  Hail - high Z with low ZDR -
        # reads as enormous KDP and the reconstruction dumps a whole segment's
        # phase into the hail core.  Capping Z is the usual mitigation; it is
        # not in the paper, so it is off unless asked for.
        if p.dbz_max_for_sc is not None:
            dbz_c = np.minimum(dbz_c, p.dbz_max_for_sc)

        kdp_zzdr = np.where(np.isfinite(kdp), kdp_from_z_zdr(dbz_c, zdr_c, a, b, c), np.nan)
        kdp_sc, phidp_sc, delta = kdp_sc_segmented(phidp_filt, kdp_zzdr, kdp, for_kdp, dr, p)
        kdp_for_atten = kdp_sc

    # --- mask to the usable region ---------------------------------------
    # _loadKdp gates KDP on SNR alone, so lrose emits speckle to max range;
    # KdpFilt.cc:520-530 is exactly this mask, commented out.
    outside = np.ones(n, dtype=bool)
    outside[first_gate : last_gate + 1] = False
    kdp_sc = np.where(for_kdp, kdp_sc, np.nan)
    delta = np.where(for_kdp, delta, np.nan)
    phidp_sc = np.where(outside, np.nan, phidp_sc)

    return kdp_sc, phidp_sc, delta


# ---------------------------------------------------------------------------
# Py-ART layer
# ---------------------------------------------------------------------------

# Searched in order; the first name present in radar.fields wins.  Raw fields
# are preferred over corrected ones because the retrieval does its own
# attenuation correction.
FIELD_CANDIDATES = {
    "dbz": (
        "DBZH",
        "reflectivity",
        "reflectivity_horizontal",
        "DBZ",
        "horizontal_reflectivity",
        "corrected_reflectivity",
        "TH",
        "total_power",
        "total_power_horizontal",
    ),
    "zdr": ("ZDR", "differential_reflectivity", "ZDRC", "corrected_differential_reflectivity"),
    "rhohv": ("RHOHV", "cross_correlation_ratio", "corrected_cross_correlation_ratio", "copol_coeff"),
    "phidp": ("PHIDP", "differential_phase", "uncorrected_differential_phase", "PHIDP_UNF"),
    "snr": ("SNRH", "signal_to_noise_ratio", "SNR", "snr"),
}

_REQUIRED = ("dbz", "zdr", "phidp")


def _resolve_fields(radar, field_names=None) -> dict:
    """Map logical names to the radar's actual field names."""
    field_names = dict(field_names or {})
    resolved = {}
    for key, candidates in FIELD_CANDIDATES.items():
        name = field_names.get(key)
        if name is not None:
            if name not in radar.fields:
                raise KeyError(
                    f"field {name!r} requested for {key!r} is not in the radar; " f"available: {sorted(radar.fields)}"
                )
            resolved[key] = name
            continue
        resolved[key] = next((c for c in candidates if c in radar.fields), None)

    missing = [k for k in _REQUIRED if resolved[k] is None]
    if missing:
        raise KeyError(
            f"could not find a field for {missing} in this radar; "
            f"available: {sorted(radar.fields)}. "
            f"Pass field_names={{'dbz': '...', 'zdr': '...', 'phidp': '...'}}."
        )
    return resolved


def _as_float(radar, name) -> np.ndarray:
    """Field data as plain float with masked and fill values turned into NaN."""
    if name is None:
        return None
    data = radar.fields[name]["data"]
    arr = np.ma.filled(np.ma.masked_invalid(np.ma.asanyarray(data)), np.nan).astype(float)
    fill = radar.fields[name].get("_FillValue")
    if fill is not None and np.isfinite(fill):
        arr[arr == fill] = np.nan
    return arr


def _wavelength_cm(radar) -> float:
    """Wavelength in cm from instrument_parameters, or raise with advice."""
    ip = getattr(radar, "instrument_parameters", None) or {}
    if "radar_wavelength" in ip:
        wl = float(np.atleast_1d(ip["radar_wavelength"]["data"])[0])
        units = str(ip["radar_wavelength"].get("units", "m")).lower()
        return wl * 100.0 if units.startswith("m") and "c" not in units else wl
    if "frequency" in ip:
        freq = float(np.atleast_1d(ip["frequency"]["data"])[0])
        if freq > 0:
            return _LIGHT_SPEED_CM_S / freq
    raise ValueError(
        "radar.instrument_parameters carries no frequency or radar_wavelength, "
        "so the band cannot be determined. Pass wavelength_cm explicitly "
        "(OceanPOL C-band is about 5.3 cm; S-band about 10)."
    )


def _field_dict(data, units, standard_name, long_name, fill=-9999.0) -> dict:
    """A Py-ART field dictionary."""
    masked = np.ma.masked_invalid(data)
    masked.set_fill_value(fill)
    return {
        "data": masked,
        "units": units,
        "standard_name": standard_name,
        "long_name": long_name,
        "_FillValue": fill,
        "coordinates": "elevation azimuth range",
    }


def retrieve(
    radar,
    wavelength_cm: float | None = None,
    params: KdpParams | None = None,
    field_names: dict | None = None,
    include_delta: bool = False,
):
    """Run the retrieval over every ray of a Py-ART radar object.

    Parameters
    ----------
    radar : pyart.core.Radar
        Any scan type.  Rays are treated independently, so sweeps need no
        special handling; ``radar.range`` is shared by all rays.
    wavelength_cm : float, optional
        Radar wavelength.  Read from ``radar.instrument_parameters`` when not
        given; many readers leave that as ``None``, in which case pass it.
    params : KdpParams, optional
    field_names : dict, optional
        Override the auto-detection, e.g. ``{"phidp": "PHIDP_CORR"}``.
    include_delta : bool
        Also return the backscatter differential phase field.

    Returns
    -------
    (kdp_field, phidp_field) : tuple of dict
        Py-ART field dictionaries, ready for ``radar.add_field``.  PHIDP is the
        accumulated differential propagation phase, **re-zeroed per ray at the
        first usable gate**, so every ray starts at 0 and the system offset is
        gone.  With ``include_delta`` a third field is appended.
    """
    p = params or KdpParams()
    names = _resolve_fields(radar, field_names)
    if wavelength_cm is None:
        wavelength_cm = _wavelength_cm(radar)

    rng_km = np.asarray(radar.range["data"], dtype=float) / 1000.0
    dr = float(np.median(np.diff(rng_km)))
    if not np.isfinite(dr) or dr <= 0:
        raise ValueError(f"could not determine a sane gate spacing (got {dr})")

    dbz = _as_float(radar, names["dbz"])
    zdr = _as_float(radar, names["zdr"])
    phidp = normalize_phidp_range(_as_float(radar, names["phidp"]))
    rhohv = _as_float(radar, names["rhohv"])
    snr = _as_float(radar, names["snr"])
    if rhohv is None:
        rhohv = np.full(dbz.shape, np.nan)

    # Decided once for the volume, not per ray
    snr_available = snr is not None and bool(np.isfinite(snr).any())

    n_rays, n_gates = phidp.shape
    kdp_out = np.full((n_rays, n_gates), np.nan)
    phidp_out = np.full((n_rays, n_gates), np.nan)
    delta_out = np.full((n_rays, n_gates), np.nan)

    for i in range(n_rays):
        kdp_ray, phidp_ray, delta_ray = process_ray(
            dbz[i], zdr[i], rhohv[i], phidp[i], None if snr is None else snr[i], dr, wavelength_cm, p, snr_available
        )

        # Re-zero PHIDP at the first usable gate so each ray starts at 0 and
        # reports accumulated propagation phase rather than system offset.
        finite = np.isfinite(phidp_ray)
        if finite.any():
            phidp_ray = phidp_ray - phidp_ray[np.argmax(finite)]

        kdp_out[i] = kdp_ray
        phidp_out[i] = phidp_ray
        delta_out[i] = delta_ray

    kdp_field = _field_dict(
        kdp_out,
        "degrees/km",
        "specific_differential_phase_hv",
        "Specific differential phase (ERAD 2026 segmented self-consistency)",
    )
    phidp_field = _field_dict(
        phidp_out,
        "degrees",
        "differential_phase_hv",
        "Differential propagation phase, self-consistency, zeroed at first gate",
    )

    if include_delta:
        delta_field = _field_dict(
            delta_out, "degrees", "backscatter_differential_phase_hv", "Backscatter differential phase (delta)"
        )
        return kdp_field, phidp_field, delta_field
    return kdp_field, phidp_field
