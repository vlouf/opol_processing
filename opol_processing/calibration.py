"""
Date-indexed calibration offsets for OceanPOL reflectivity (ZH) and
differential reflectivity (ZDR).

The radar calibration drifted over the life of the archive, so the offsets are
stored as step tables keyed by date. ``get_calib_offset`` returns the offset in
force on a given day (the most recent entry at or before that date), matching
the behaviour of the oceanpol_kit reference processing.

@title: calibration
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology

.. autosummary::
    :toctree: generated/

    get_calib_offset
"""
import io
from typing import Tuple

import pandas as pd


# Reflectivity (ZH) calibration offset in dB, by date of change.
DBZ_CALIB = """date,offset
2000-01-01,1.4
2020-07-31,1.7
2020-11-30,2.3
2021-01-31,2.5
2021-04-30,1.5
2022-05-31,0.0
2024-06-30,3.2
2026-01-01,1.7"""

# Differential reflectivity (ZDR) calibration offset in dB, by date of change.
ZDR_CALIB = """date,offset
2000-01-01,0.5
2020-01-01,0.5
2021-01-01,0.75
2022-01-01,1.0
2023-01-01,1.0
2023-05-01,0.5
2024-06-30,0.5"""


def _load(table: str) -> "pd.Series":
    frame = pd.read_csv(io.StringIO(table), sep=",", parse_dates=["date"], index_col="date")
    return frame["offset"].sort_index()


def get_calib_offset(date) -> Tuple[float, float]:
    """
    Return ``(dbz_offset, zdr_offset)`` in dB for a given date.

    The value returned is the most recent calibration entry at or before
    ``date`` (step/forward-fill semantics). Dates before the first entry get
    0.0; dates after the last entry keep the last value.

    Parameters
    ----------
    date : datetime-like
        Volume date (time-of-day is ignored).

    Returns
    -------
    Tuple[float, float]
        (dbz_offset, zdr_offset).
    """
    day = pd.Timestamp(date).normalize()

    dbz = _load(DBZ_CALIB).asof(day)
    zdr = _load(ZDR_CALIB).asof(day)

    dbz = 0.0 if pd.isna(dbz) else float(dbz)
    zdr = 0.0 if pd.isna(zdr) else float(zdr)

    return dbz, zdr
