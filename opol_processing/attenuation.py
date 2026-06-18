"""
Codes for correcting attenuation on ZH (Z-PHI) and ZDR.

The ZH path-integrated attenuation uses the Z-PHI method (Gu et al. 2011),
ported from the oceanpol_kit reference. The ZDR differential attenuation
correction (Bringi et al. 2001) is retained from the original OPOL pipeline.
The legacy gaseous-attenuation step has been removed.

@title: attenuation
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institutions: Monash University and the Australian Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    smooth_and_trim
    correct_attenuation
    correct_attenuation_zdr
"""
import pyart
import numpy as np

from scipy.integrate import cumulative_trapezoid


def smooth_and_trim(x, window_len=11, window="hanning"):
    """
    Smooth data using a window with requested size.

    The signal is prepared by introducing reflected copies of the signal (with
    the window size) at both ends so that transient parts are minimised in the
    beginning and end of the output signal.

    Parameters
    ----------
    x : array
        The input signal (1D).
    window_len : int, optional
        The dimension of the smoothing window; should be an odd integer.
    window : str
        One of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman' or
        'sg_smooth'. A flat window produces a moving-average smoothing.

    Returns
    -------
    y : array
        The smoothed signal, same length as the input.
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    valid_windows = ["flat", "hanning", "hamming", "bartlett", "blackman", "sg_smooth"]
    if window not in valid_windows:
        raise ValueError("Window is one of " + " ".join(valid_windows))

    s = np.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]

    if window == "flat":  # moving average
        w = np.ones(int(window_len), "d")
    elif window == "sg_smooth":
        w = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")

    return y[int(window_len / 2) : len(x) + int(window_len / 2)]


def correct_attenuation(r, refl, proc_dp_phase_shift, temp):
    """
    Calculate the path-integrated attenuation (PIA) for a polarimetric radar
    using the Z-PHI method.

    References
    ----------
    Gu et al. Polarimetric Attenuation Correction in Heavy Rain at C Band,
    JAMC, 2011, 50, 39-58.

    Parameters
    ----------
    r : np.ndarray
        1D range array (m).
    refl : np.ndarray
        2D (masked) cleaned reflectivity (dBZ).
    proc_dp_phase_shift : np.ndarray
        2D corrected differential phase (deg).
    temp : np.ndarray
        2D (masked) temperature (degC); attenuation is integrated only where
        temperature >= 0.

    Returns
    -------
    np.ndarray
        2D path-integrated attenuation (dB), float32.
    """
    a_coef = 0.06
    beta = 0.8

    phidp = np.nan_to_num(np.asarray(proc_dp_phase_shift, dtype="float64"))
    init_refl_correct = refl + phidp * a_coef
    dr = (r[1] - r[0]) / 1000.0

    specific_atten = np.zeros(refl.shape, dtype="float32")
    atten = np.zeros(refl.shape, dtype="float32")

    mask = np.ma.getmaskarray(np.ma.asarray(refl)) | (np.ma.filled(temp, -9999) < 0)

    for i in range(refl.shape[0]):
        ray_phase_shift = phidp[i, :]
        ray_init_refl = np.ma.filled(init_refl_correct[i, :], np.nan)

        good = np.where(~mask[i, :])[0]
        if good.size == 0:
            # Whole ray masked: no valid gates, leave attenuation at zero.
            continue
        last_six_good = good[-6:]
        phidp_max = np.median(ray_phase_shift[last_six_good])
        if not np.isfinite(phidp_max):
            continue

        sm_refl = smooth_and_trim(ray_init_refl, window_len=5)
        reflectivity_linear = 10.0 ** (0.1 * beta * sm_refl)
        reflectivity_linear[np.isnan(reflectivity_linear)] = 0
        self_cons_number = 10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0
        I_indef = cumulative_trapezoid(0.46 * beta * dr * reflectivity_linear[::-1])
        I_indef = np.append(I_indef, I_indef[-1])[::-1]

        # set the specific attenuation and the path-integrated attenuation
        specific_atten[i, :] = reflectivity_linear * self_cons_number / (I_indef[0] + self_cons_number * I_indef)

        atten[i, :-1] = cumulative_trapezoid(specific_atten[i, :]) * dr * 2.0
        atten[i, -1] = atten[i, -2]

    return atten


def correct_attenuation_zdr(radar, gatefilter, zdr_name="ZDR_CORR", phidp_name="PHIDP_PHIDO", alpha=0.016):
    """
    Correct differential attenuation on differential reflectivity (ZDR).

    V. N. Bringi, T. D. Keenan and V. Chandrasekar, "Correcting C-band radar
    reflectivity and differential reflectivity data for rain attenuation: a
    self-consistent method with constraints," IEEE TGRS, 39(9), 1906-1915, 2001.
    doi: 10.1109/36.951081

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART radar structure.
    gatefilter : pyart.filters.GateFilter
        Filter excluding non-meteorological echoes.
    zdr_name : str
        Differential reflectivity field name (unused; kept for signature
        compatibility).
    phidp_name : str
        Corrected differential phase field name.
    alpha : float, optional
        Z-PHI differential attenuation coefficient (default 0.016).

    Returns
    -------
    dict
        Py-ART field dictionary of path-integrated differential attenuation.
    """
    phi = radar.fields[phidp_name]["data"].copy()

    atten_corr = alpha * phi
    atten_corr[gatefilter.gate_excluded] = np.nan
    atten_corr = np.ma.masked_invalid(atten_corr)
    np.ma.set_fill_value(atten_corr, np.nan)

    atten_zdr_meta = pyart.config.get_metadata("path_integrated_differential_attenuation")
    atten_zdr_meta["description"] = "Differential attenuation correction using Bringi et al. 2001."
    atten_zdr_meta["_FillValue"] = np.nan
    atten_zdr_meta["_Least_significant_digit"] = 2
    atten_zdr_meta["data"] = atten_corr.astype(np.float32)

    return atten_zdr_meta
