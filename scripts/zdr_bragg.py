import os
import gc
import glob
import uuid
import argparse
import datetime
import traceback
from typing import AnyStr, Tuple, List
import warnings

import pyodim
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import dask.bag as db


class NoRainError(Exception):
    pass


class FileInformation:
    def __init__(self, input_file) -> None:
        self.filename = self.check_input_file(input_file)
        self.rid = 502
        self.set_infos()

    def check_input_file(self, incoming_data: AnyStr) -> str:
        if type(incoming_data) is str:
            filename = incoming_data
        else:
            filename = incoming_data.decode("utf-8")

        if not os.path.isfile(filename) or os.stat(filename).st_size == 0:
            raise FileNotFoundError(f"Incoming data {filename} is not a file or does not exist.")

        return filename

    def set_infos(self) -> None:
        var = []
        with netCDF4.Dataset(self.filename) as ncid:
            self.lat = ncid["where"].lat
            self.lon = ncid["where"].lon
            date = ncid["what"].date
            time = ncid["what"].time
            groups = ncid["/dataset1"].groups.keys()
            for group in groups:
                if "data" not in group:
                    continue
                name = ncid[f"/dataset1/{group}/what"].getncattr("quantity")
                var.append(name)

        self.datetime = datetime.datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")
        self.moments = var

    def get_rid(self) -> int:
        rid = int(os.path.basename(self.filename).split("_")[0])
        if rid <= 0 or rid > 1000:
            ValueError(f"Could not determine radar Rapic ID {rid} for file {self.filename}. Doing nothing.")
        return rid


def extract_bragg(radar: xr.Dataset) -> Tuple[np.ndarray, float]:
    """
    Extract Bragg scatterer's reflectivity

    Parameter:
    ==========
    radar: xr.Dataset
        Input radar ODIM h5 dataset.

    Returns:
    ========
    zdr: np.ndarray
        Differential reflecitivity of Bragg scatterers.
    refl_range: np.ndarray
        Reflectivity of ALL scatterers in valid spatial domain
    refl_brag: np.ndarray
        Reflectivity of Bragg scatterers only.

    Reference:
    ==========
    Richardson et al. (2017), Bragg Scatter Detection by the WSR-88D. Part I:
    Algorithm Development, JAOT, 10.1175/JTECH-D-16-0030.1
    """
    r = radar.range.values
    azi = radar.azimuth.values
    [R, A] = np.meshgrid(r, azi)

    dbz = radar.TH.values
    zdr = radar.ZDR.values
    rhohv = radar.RHOHV.values
    snr = radar.SNR.values
    vel = radar.VRAD.values
    spectrum_width = radar.WRAD.values

    pos_range = (R > 10e3) & (R < 80e3) & ~np.isnan(dbz)

    pos_brag = (
        pos_range
        & ~np.isnan(zdr)
        & (rhohv >= 0.9)
        & (dbz < 10)
        & (snr < 15)
        & (np.abs(vel) > 2)
        & (spectrum_width > 0)
    )

    return zdr[pos_brag], dbz[pos_range], dbz[pos_brag]


def extract_zdr_hist(radar: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ZDR from input file and compute the histogram over the Birdbath
    scan in light precipitation.

    Parameter:
    ==========
    radar: xr.Dataset
        Input radar ODIM h5 dataset.

    Returns:
    ========
    zdrhist_azi: np.ndarray
        Birdbath histogram ZDR of light precipitation as a function of Azimuth
    zdrhist_range: np.ndarray
        Birdbath histogram ZDR of light precipitation as a function of Range
    """
    refl = get_any_reflectivity(radar)

    try:
        rhohv = radar["RHOHV"].values
    except KeyError:
        raise KeyError("DP not found")

    zdr = radar["ZDR"].values  # File is being tested before.

    r = radar.range.values
    azi = radar.azimuth.values
    R, A = np.meshgrid(r, azi)
    pos = (refl > 10) & (refl < 36) & (rhohv > 0.95) & ~np.isnan(zdr) & (R >= 1500) & (R <= 4000)
    trlimiter_pos = (refl > 10) & (refl < 36) & (rhohv > 0.95) & ~np.isnan(zdr)

    if np.sum(pos) < 5:
        raise NoRainError("No precipitation")

    zdrhist_azi, _, _ = np.histogram2d(A[pos], zdr[pos], range=[(0, 360), ZDR_RANGE], bins=(360, NBINS))
    zdrhist_range, _, _ = np.histogram2d(
        R[trlimiter_pos], zdr[trlimiter_pos], range=[(0, 20e3), ZDR_RANGE], bins=(81, NBINS)
    )

    return zdrhist_azi, zdrhist_range


def read_radar_data(infile: str) -> xr.Dataset:
    """
    Read radar data and extract ZDR calibration values.

    Parameter:
    ==========
    infile: str
        Input file name.

    Returns:
    ========
    dset: xr.Dataset
        Contains: zdr_azimuth_hbins, zdr_range_hbins, and zdr_bragg.
    """
    rads = pyodim.read_odim(infile, lazy_load=False)

    bragg = np.array([])
    refl = np.array([])
    refl_brag = np.array([])
    zdrhist_azi = None
    zdrhist_range = None
    for radar in rads:
        elev = radar.elevation.values[0]
        if elev >= 2.5 and elev <= 4.5:  # Bragg
            try:
                zdr, dbz_r, dbz_b = extract_bragg(radar)
            except Exception:
                traceback.print_exc()
                continue
            bragg = np.append(bragg, zdr)
            refl = np.append(refl, dbz_r)
            refl_brag = np.append(refl_brag, dbz_b)
        elif elev >= 89:  # Birdbath
            try:
                zdrhist_azi, zdrhist_range = extract_zdr_hist(radar)
            except (KeyError, NoRainError):
                continue
            except Exception:
                print(f"PROBLEM with file {infile} for birdbath extraction")
                traceback.print_exc()
                pass

    # QC for Bragg
    try:
        if (len(refl) < 10e3) or (np.percentile(refl_brag, 90) > 6):
            raise ValueError("Not Enough Points.")
        else:
            iqr = np.abs(np.percentile(bragg, 75) - np.percentile(bragg, 25))
            if iqr > 0.9:
                raise ValueError("Not Enough Points.")

        zdr_bragg, _ = np.histogram(bragg, range=ZDR_RANGE, bins=NBINS)
    except (ValueError, IndexError):
        zdr_bragg = np.zeros((NBINS), dtype=np.int32)

    if (zdr_bragg.sum() == 0) and (zdrhist_azi is None):
        print(f"No valid rain value found for either technique for {infile}")
        raise NoRainError

    if zdrhist_azi is None:
        zdrhist_azi = np.zeros((360, NBINS), dtype=np.int32)
        zdrhist_range = np.zeros((81, NBINS), dtype=np.int32)

    azi = np.linspace(0, 360, 360)
    r = np.linspace(0, 20e3, 81)
    y = np.linspace(ZDR_RANGE[0], ZDR_RANGE[1], NBINS)

    dset = xr.Dataset(
        {
            "zdr_azimuth_hbins": (("azimuth", "zdr"), zdrhist_azi.astype(np.int32)),
            "zdr_range_hbins": (("range", "zdr"), zdrhist_range.astype(np.int32)),
            "zdr_bragg": (("zdr"), zdr_bragg.astype(np.int32)),
            "azimuth": (("azimuth"), azi.astype(np.float32)),
            "range": (("range"), r.astype(np.float32)),
            "zdr": (("zdr"), y.astype(np.float32)),
        }
    )

    return dset


def savedata(dset, rid, date, outfilename: str) -> None:
    """
    Save the output data into a netCDF.

    Parameters:
    ===========
    outfilename: str
        Output file name.
    hist: ndarray
        Output directory.
    """
    datestr = date.strftime("%Y-%m-%d")
    if not os.path.isfile(outfilename):
        # Metadata
        dset.range.attrs = {
            "units": "meters",
            "standard_name": "projection_range_coordinate",
            "long_name": "range_to_measurement_volume",
        }
        dset.zdr.attrs = {
            "units": "dB",
            "standard_name": "log_differential_reflectivity_hv",
            "long_name": "Differential reflectivity",
        }
        dset.azimuth.attrs = {
            "units": "degrees",
            "standard_name": "beam_azimuth_angle",
            "long_name": "azimuth_angle_from_true_north",
        }
        dset.zdr_azimuth_hbins.attrs = {
            "units": "1",
            "description": "Birdbath histogram ZDR of light precipitation as a function of Azimuth",
        }
        dset.zdr_range_hbins.attrs = {
            "units": "1",
            "description": "Birdbath histogram ZDR of light precipitation as a function of Range",
        }
        dset.zdr_bragg.attrs = {
            "units": "1",
            "description": "Histogram of Bragg's scatterer ZDR.",
        }
        dset.attrs = {
            "title": f"Daily ZDR birdbath histogram in light rain precip for {datestr}",
            "country": "Australia",
            "creator_email": "valentin.louf@bom.gov.au",
            "creator_name": "Valentin Louf",
            "date_created": datetime.datetime.now().isoformat(),
            "description": "Ground radar monitoring of Differential Reflectivity in light rain or Bragg scattering.",
            "institution": "Bureau of Meteorology",
            "project": "s3car-server",
            "radar_date": datestr,
            "rapic_ID": str(rid),
            "uuid": str(uuid.uuid4()),
            "tr_limiter": "test_v0.1",
        }
        dset.to_netcdf(outfilename)
        print(f"Output {outfilename} written.")
    else:
        with xr.open_dataset(outfilename, mode="a") as nset:
            nset.zdr_azimuth_hbins.values += dset.zdr_azimuth_hbins.values
            nset.zdr_range_hbins.values += dset.zdr_range_hbins.values
            nset.zdr_bragg.values += dset.zdr_bragg.values

            nset.to_netcdf(outfilename)
        print(f"Output {outfilename} updated.")

    return None


def generate_output_filename(rid, date):
    zdr_path = OUTDIR
    datestr = date.strftime("%Y%m%d")

    outpath = os.path.join(zdr_path, str(rid))
    os.makedirs(outpath, exist_ok=True)
    outfilename = os.path.join(outpath, f"zdr.{rid}.{datestr}.nc")

    return outfilename


def processing_radar_file(incoming_data: AnyStr) -> None:
    """
    Driver for layered-flow and manage the directories, configuration files,
    lag0 file, etc...

    Parameter:
    ==========
    filename: AnyStr
        Input radar file name.
    """
    fileinfo = FileInformation(incoming_data)
    date = pd.Timestamp(fileinfo.datetime)

    if "ZDR" not in fileinfo.moments:
        print(f"{fileinfo.filename} does not contain ZDR. Doing nothing")
        return None

    outfilename = generate_output_filename(fileinfo.rid, date)
    if outfilename == '/scratch/kl02/vhl548/opol/zdrcal/502/zdr.502.20180211.nc':
        return None

    try:
        dset = read_radar_data(fileinfo.filename)
        savedata(dset, fileinfo.rid, date, outfilename)
    except NoRainError:
        return None

    return None


def main():
    # inpath = "/g/data/hj10/admin/opol/level_1/in2018_v01/"
    # inpath = r"/g/data/kl02/vhl548/CAPRIX 2022/OceanPOL/hdf5"
    for date in os.listdir(INDIR):
        flist = sorted(glob.glob(os.path.join(INDIR, date, "*.hdf")))
        if len(flist) == 0:
            continue

        try:
            bag = db.from_sequence(flist).map(processing_radar_file)
            rslt = bag.compute()
        except Exception:
            traceback.print_exc()
        finally:
            gc.collect()

    return None


if __name__ == "__main__":
    ZDR_RANGE: Tuple[float, float] = (-2, 2)  # ZDR min/max for histogram.
    ZDR_STEP: float = abs(0.08)  # ZDR resolution in dB
    NBINS: int = int(np.round(abs(ZDR_RANGE[1] - ZDR_RANGE[0]) / ZDR_STEP) + 1)

    parser_description = """Extract ZDR Bragg"""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", dest="indir", type=str, help="Input directory", required=True)
    parser.add_argument(
        "-o", "--output", dest="outdir", type=str, help="Output directory.", default="/scratch/kl02/vhl548/opol/zdrcal/"
    )
    args = parser.parse_args()
    OUTDIR = args.outdir
    INDIR = args.indir

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()