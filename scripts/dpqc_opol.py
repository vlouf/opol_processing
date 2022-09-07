import gc
import os
import glob
import argparse
import warnings
import traceback

from typing import AnyStr, Dict

import pyart
import numpy as np
import pandas as pd

import dask.bag as db


def get_statistics(
    dbzh: np.ndarray, phidp: np.ndarray, zdr: np.ndarray, rhohv: np.ndarray, kdp: np.ndarray, pos: np.ndarray,
) -> Dict:
    """
    Perform Marks et al. (2011) series of statistics for dual-polarisation
    quality checks.
    Parameters:
    ===========
    dbzh: ndarray<azimuth, range>
        Array of reflectivity (lowest elevation scan)
    phidp: ndarray<azimuth, range>
        Array of differential phase (lowest elevation scan)
    zdr: ndarray<azimuth, range>
        Array of differential reflectivity (lowest elevation scan)
    rhohv: ndarray<azimuth, range>
        Array of cross correlation ratio (lowest elevation scan)
    kdp: ndarray<azimuth, range>
        Array of specific differential phase (lowest elevation scan)
    pos: ndarray<azimuth, range>
        Mask for light rain filtering.
    Returns:
    ========
    statistics: dict
        DP quality checks statistics for given scan.
    """

    def zdr_stats(zdrp):
        szdr = np.std(zdrp)
        aad = np.sum(np.abs(zdrp - np.mean(zdrp))) / len(zdrp)
        return szdr, aad

    # PHIDP rolling window.
    wsize = 15  # Window size for avg phidp.
    sigma_phi = np.zeros_like(phidp)
    for i in range(phidp.shape[0]):
        try:
            sigma_phi[i, :] = pd.Series(np.ma.masked_where(~pos, phidp)[i, :]).rolling(wsize).std()
        except Exception:
            continue
    sigma_phi = np.ma.masked_invalid(sigma_phi).filled(np.NaN)
    nsample_kdp = (~np.isnan(sigma_phi)).sum()
    if nsample_kdp == 0:
        sig_kdp = np.NaN
        sig_phi_med = np.NaN
        sig_phi_std = np.NaN
    else:
        sig_kdp = np.nanmedian(np.sqrt(3) * sigma_phi / (wsize ** 1.5 * 0.25))
        sig_phi_med = np.nanmedian(sigma_phi)
        sig_phi_std = np.nanstd(sigma_phi)

    # ZDR stats.
    szdr, aad = zdr_stats(zdr[pos])

    statistics = {
        "N": pos.sum(),
        "ZH_med": np.median(dbzh[pos]),
        "RHOHV_med": np.median(rhohv[pos]),
        "RHOHV_std": np.std(rhohv[pos]),
        "KDP_med": np.median(kdp[pos]),
        "KDP_std": np.std(kdp[pos]),
        "ZDR_std": szdr,
        "ZDR_aad": aad,
        "PHIDP_med": np.nanmedian(phidp[pos]),
        "PHIDP_std": np.nanstd(phidp[pos]),
        "N_kdp_sample": nsample_kdp,
        "SIGMA_PHIDP_med": sig_phi_med,
        "SIGMA_PHIDP_std": sig_phi_std,
        "SIGMA_kdp": sig_kdp,
    }

    for k, v in statistics.items():
        if type(v) == np.ma.core.MaskedConstant:
            statistics[k] = np.NaN

    return statistics


def read_data(infile)    :
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        try:
            radar = pyart.io.read(infile, delay_field_loading=True)
        except Exception:
            print(f"Could not read {infile}")
            return None

        if radar.nsweeps <= 2:
            return None

        # Load data
        sl = radar.get_slice(0)
        dbzh = radar.fields["corrected_reflectivity"]["data"][sl]

        zdr = radar.fields["differential_reflectivity"]["data"][sl]
        rhohv = radar.fields["cross_correlation_ratio"]["data"][sl]
        phidp = radar.fields["differential_phase"]["data"][sl]
        kdp = radar.fields["corrected_specific_differential_phase"]["data"][sl]
        x = radar.gate_x['data'][sl] / 1000
        y = radar.gate_y['data'][sl] / 1000
        r = np.sqrt(x ** 2 + y ** 2)

        dtime = pd.Timestamp(radar.time["units"].split(" ")[-1]) + pd.Timedelta(seconds=radar.time["data"][0])

        pos_lowcut = (
            ~np.isnan(dbzh)
            & (dbzh >= 20)
            & (dbzh <= 28)
            & (rhohv > 0.8)
            & (r < 150)
            & (r >= 20)
            & (kdp > -2)
            & (kdp < 3)
            & (zdr < 2.5)
            & (zdr > -1)
        )

        if np.sum(pos_lowcut) < 100:
            return None

        lowstats = get_statistics(dbzh, phidp, zdr, rhohv, kdp, pos_lowcut)

        df = pd.DataFrame(lowstats, index=[dtime])
        df.index.name = "time"

    return df


def main():
    inpath = INDIR

    for date in os.listdir(inpath):
        flist = sorted(glob.glob(os.path.join(inpath, date, "*.nc")))

        outfilename = os.path.join(OUTDIR, f"dpqc.{date}.csv")
        if os.path.exists(outfilename):
            continue

        try:
            bag = db.from_sequence(flist).map(read_data)
            rslt = bag.compute()
            df = pd.concat([r for r in rslt if r is not None])
            df.to_csv(outfilename)
        except Exception:
            traceback.print_exc()
        finally:
            gc.collect()

    return None


if __name__ == "__main__":
    parser_description = """Extract ZDR Bragg"""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", dest="indir", type=str, help="Input directory", required=True)
    parser.add_argument(
        "-o", "--output", dest="outdir", type=str, help="Output directory.", default="/scratch/kl02/vhl548/opol/dpqc/"
    )
    args = parser.parse_args()
    OUTDIR = args.outdir
    INDIR = args.indir

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

