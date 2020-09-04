"""
Raw radar PPIs processing. Quality control, filtering, attenuation correction,
dealiasing, unfolding, hydrometeors calculation, rainfall rate estimation.

@title: opol_processing
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Monash University and Bureau of Meteorology
@date: 04/09/2020

.. autosummary::
    :toctree: generated/

    chunks
    main
    welcome_message
"""
# Python Standard Library
import os
import sys
import glob
import argparse
import datetime
import warnings
import traceback

import dask
import dask.bag as db
import crayons
import unravel
import opol_processing


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def buffer(infile):
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
    try:
        opol_processing.process_and_save(infile, OUTPATH, do_dealiasing=DO_DEALIASING, use_unravel=USE_UNRAVEL)
    except Exception:
        traceback.print_exc()

    return None


def main():
    flist = sorted(glob.glob(os.path.join(INPATH, "**/*.hdf")))
    print(f"Found {len(flist)} files in {INPATH}")
    for fchunk in chunks(flist, 64):
        bag = db.from_sequence(fchunk).map(buffer)
        _ = bag.compute()
        del bag

    return None


if __name__ == "__main__":
    """
    Global variables definition.
    """
    # Main global variables (Path directories).
    OUTPATH = "/scratch/kl02/vhl548/opol/"

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
