import os
import sys
import glob
import shutil
import argparse
import datetime
import warnings
import traceback

import h5py
import pyart
import pyodim
import numpy as np
import dask
import dask.bag as db
import opol_processing


def chunks(l, n: int):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def coord_from_metadata(metadata):
    """
    Create the radar coordinates from the ODIM H5 metadata specification.
    Parameter:
    ==========
    metadata: dict()
        Metadata dictionnary containing the specific ODIM H5 keys: astart,
        nrays, nbins, rstart, rscale, elangle.
    Returns:
    ========
    r: ndarray<nbins>
        Sweep range
    azimuth: ndarray<nrays>
        Sweep azimuth
    elev: float
        Sweep elevation
    """
    da = 360 / metadata["nrays"]
    azimuth = np.linspace(metadata["astart"] + da / 2, 360 - da, metadata["nrays"], dtype=np.float32)

    # rstart is in KM !!! STUPID.
    rstart_center = 1e3 * metadata["rstart"] + metadata["rscale"] / 2
    r = np.arange(
        rstart_center, rstart_center + metadata["nbins"] * metadata["rscale"], metadata["rscale"], dtype=np.float32
    )

    elev = np.array([metadata["elangle"]], dtype=np.float32)
    return r, azimuth, elev


def get_dataset_metadata(hfile: str, dataset: str="dataset1"):
    """
    Get the dataset metadata of the ODIM H5 file.
    Parameters:
    ===========
    hfile: h5py.File
        H5 file identifier.
    dataset: str
        Key of the dataset for which to extract the metadata
    Returns:
    ========
    metadata: dict
        General metadata of the dataset.
    coordinates_metadata: dict
        Coordinates-specific metadata.
    """
    metadata = dict()
    coordinates_metadata = dict()
    # General metadata
    metadata["NI"] = hfile[f"/{dataset}/how"].attrs["NI"]
    metadata["highprf"] = hfile[f"/{dataset}/how"].attrs["highprf"]
    metadata["product"] = _to_str(hfile[f"/{dataset}/what"].attrs["product"])

    sdate = _to_str(hfile[f"/{dataset}/what"].attrs["startdate"])
    stime = _to_str(hfile[f"/{dataset}/what"].attrs["starttime"])
    edate = _to_str(hfile[f"/{dataset}/what"].attrs["enddate"])
    etime = _to_str(hfile[f"/{dataset}/what"].attrs["endtime"])
    metadata["start_time"] = f"{sdate}_{stime}"
    metadata["end_time"] = f"{edate}_{etime}"

    # Coordinates:
    try:
        coordinates_metadata["astart"] = hfile[f"/{dataset}/how"].attrs["astart"]
    except KeyError:
        # Optional coordinates (!).
        coordinates_metadata["astart"] = 0
    coordinates_metadata["a1gate"] = hfile[f"/{dataset}/where"].attrs["a1gate"]
    coordinates_metadata["nrays"] = hfile[f"/{dataset}/where"].attrs["nrays"]

    coordinates_metadata["rstart"] = hfile[f"/{dataset}/where"].attrs["rstart"]
    coordinates_metadata["rscale"] = hfile[f"/{dataset}/where"].attrs["rscale"]
    coordinates_metadata["nbins"] = hfile[f"/{dataset}/where"].attrs["nbins"]

    coordinates_metadata["elangle"] = hfile[f"/{dataset}/where"].attrs["elangle"]

    return metadata, coordinates_metadata


def _to_str(t) -> str:
    """
    Transform binary into string.
    """
    return t.decode("utf-8")


def cleanup_ppi(infile: str) -> str:
    """
    Copy input file in tmp directory and removed all the surveillance sweeps
    from the ODIM h5 file.

    Parameters:
    ===========
    infile: str
        Input ODIM h5 file

    Returns:
    ========
    filename: str
        Path to the cleaned up file.
    """
    fname = os.path.basename(infile)
    filename = os.path.join(TMPDIR, fname)
    shutil.copy(infile, filename)

    with h5py.File(filename, "a") as hfile:
        sweeps = dict()
        nsweep = len([k for k in hfile["/"].keys() if k.startswith("dataset")])

        for key in hfile["/"].keys():
            if key.startswith("dataset"):
                sweeps[key] = hfile[f"/{key}/where"].attrs["elangle"]

        sorted_keys = sorted(sweeps, key=lambda k: sweeps[k])
        for nslice in range(nsweep):
            rootkey = sorted_keys[nslice]
            _, coordinates_metadata = get_dataset_metadata(hfile, rootkey)
            if coordinates_metadata['nbins'] == 1920:
                del hfile[f"/{rootkey}"]
                break

    return filename


def buffer(infile: str) -> None:
    """
    It calls the production line and manages it. Buffer function that is used
    to catch any problem with the processing line without screwing the whole
    multiprocessing stuff.

    Parameters:
    ===========
    infile: str
        Name of the input radar file.
    outpath: str
        Path for saving output data.
    """
    file_to_process = cleanup_ppi(infile)
    try:
        opol_processing.process_and_save(file_to_process, OUTPATH, do_dealiasing=DO_DEALIASING, use_unravel=USE_UNRAVEL)
    except Exception:
        traceback.print_exc()
    finally:
        os.remove(file_to_process)

    return None


def main() -> None:
    flist = sorted(glob.glob(os.path.join(INPATH, "*.hdf")))
    if len(flist) == 0:
        raise FileNotFoundError(f"No file found in {INPATH}")

    print(f"Found {len(flist)} files in {INPATH}")
    for fchunk in chunks(flist, 64):
        bag = db.from_sequence(fchunk).map(buffer)
        _ = bag.compute()

    return None


if __name__ == "__main__":
    """
    Global variables definition.
    """
    # Main global variables (Path directories).
    OUTPATH = "/scratch/kl02/vhl548/opol/"
    TMPDIR = "/scratch/kl02/vhl548/tmp/"

    # Parse arguments
    parser_description = """Raw radar PPIs processing. It provides Quality
control, filtering, attenuation correction, dealiasing, unfolding, hydrometeors
calculation, and rainfall rate estimation."""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument(
        "-i", "--input-dir", dest="indir", default=None, type=str, help="Input directory.", required=True
    )
    parser.add_argument("--unravel", dest="unravel", action="store_true")
    parser.add_argument("--no-unravel", dest="unravel", action="store_false")
    parser.set_defaults(unravel=True)
    parser.add_argument("--dealias", dest="dealias", action="store_true")
    parser.add_argument("--no-dealias", dest="dealias", action="store_false")
    parser.set_defaults(dealias=True)

    args = parser.parse_args()
    USE_UNRAVEL = args.unravel
    DO_DEALIASING = args.dealias
    INPATH = args.indir

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()