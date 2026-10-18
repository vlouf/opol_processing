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
#
# Derived from the light-rain method in notebooks/oceanpol_liquid_dsd_epochs.ipynb
# (ERAD 2026 statistics, Sep 2026): per day, median ZDR of liquid gates in the
# 19-23 dBZ window minus a 0.23 dB reference; per voyage, the median of the daily
# estimates. That "residual" was measured on data already corrected with the previous
# table, so each entry here is (previous offset - residual). One entry per voyage, keyed
# on the voyage start date. Voyages without a usable estimate keep the previous value.
# Pre-2020 residuals were measured on the level_1b v2020 archive, assumed to carry the
# previous 0.5 dB offset. Per-voyage values in notebooks/oceanpol_zdr_voyage_offsets.csv.
# Text after '#' on a line is ignored.
ZDR_CALIB = """date,offset
2000-01-01,0.50   # before the archive; previous default
2017-03-16,0.50   # in2017_v02  no light-rain estimate -> previous calibration kept
2017-05-04,0.50   # in2017_c02  no light-rain estimate -> previous calibration kept
2017-05-15,0.50   # in2017_v03  no light-rain estimate -> previous calibration kept
2017-08-23,0.50   # in2017_e03  no light-rain estimate -> previous calibration kept
2017-08-31,0.50   # in2017_v04  no light-rain estimate -> previous calibration kept
2017-09-24,0.50   # in2017_t01  no light-rain estimate -> previous calibration kept
2017-09-28,0.50   # in2017_v05  no light-rain estimate -> previous calibration kept
2017-11-23,0.50   # in2017_t02  no light-rain estimate -> previous calibration kept
2017-11-24,0.18   # in2018_v01  residual +0.32 dB (36 d)
2018-03-13,0.50   # in2018_v02  no light-rain estimate -> previous calibration kept
2018-04-05,0.50   # in2018_t01  no light-rain estimate -> previous calibration kept
2018-04-18,-0.17  # in2018_v03  residual +0.67 dB (21 d)
2018-05-14,0.50   # in2018_t02  no light-rain estimate -> previous calibration kept
2018-05-28,0.50   # in2018_c01  no light-rain estimate -> previous calibration kept
2018-07-26,0.50   # in2018_v07  no light-rain estimate -> previous calibration kept
2018-09-11,0.50   # in2018_v04  no light-rain estimate -> previous calibration kept
2018-10-16,-0.68  # in2018_v05  residual +1.18 dB (20 d)  [beyond 1 dB QC limit]
2018-11-23,0.50   # in2018_v06  no light-rain estimate -> previous calibration kept
2018-12-27,0.50   # in2018_v08  no light-rain estimate -> previous calibration kept
2019-01-21,0.50   # in2019_v01  no light-rain estimate -> previous calibration kept
2019-03-13,0.50   # in2019_v02  no light-rain estimate -> previous calibration kept
2019-04-11,-0.09  # in2019_v07  residual +0.59 dB (9 d)
2019-04-28,-0.11  # in2019_t01  residual +0.61 dB (11 d)
2019-05-14,0.14   # in2019_v03  residual +0.36 dB (21 d)
2019-08-06,0.07   # in2019_v04  residual +0.43 dB (20 d)
2019-09-09,0.50   # in2019_v05  no light-rain estimate -> previous calibration kept
2019-09-14,0.39   # in2019_v06  residual +0.11 dB (35 d)
2019-12-05,0.50   # in2019_t03  no light-rain estimate -> previous calibration kept
2020-01-10,0.15   # IN2020_V01  residual +0.35 dB (12 d)
2020-07-29,0.29   # IN2020_E01  residual +0.21 dB (9 d)
2020-08-25,0.38   # IN2020_V09  residual +0.12 dB (18 d)
2020-10-08,0.07   # IN2020_V06  residual +0.43 dB (22 d)
2020-11-12,0.43   # IN2020_V10  residual +0.07 dB (10 d)
2020-12-03,0.20   # IN2020_v08  residual +0.30 dB (34 d)
2021-01-29,0.29   # IN2021_V01  residual +0.46 dB (14 d)
2021-04-14,0.41   # IN2021_V02  residual +0.34 dB (13 d)
2021-05-07,0.87   # IN2021_V03  residual -0.12 dB (26 d)
2021-06-10,0.66   # IN2021_T01  residual +0.09 dB (13 d)
2021-06-30,0.71   # IN2021_V04  residual +0.04 dB (22 d)
2021-11-09,0.83   # IN2021_E01  residual -0.08 dB (6 d)
2021-11-20,0.84   # IN2021_E02  residual -0.09 dB (6 d)
2021-11-30,0.69   # IN2021_E03  residual +0.06 dB (7 d)
2021-12-13,0.84   # IN2021_E04  residual -0.09 dB (7 d)
2022-03-10,1.21   # IN2022_V02  residual -0.21 dB (27 d)
2022-05-04,0.89   # IN2022_V03  residual +0.11 dB (10 d)
2022-05-28,0.96   # IN2022_V05  residual +0.04 dB (30 d)
2022-07-13,1.13   # IN2022_V06  residual -0.13 dB (15 d)
2022-08-13,1.05   # IN2022_V07  residual -0.05 dB (17 d)
2022-09-30,0.99   # IN2022_V08  residual +0.01 dB (27 d)
2022-11-19,1.00   # IN2022_V09  no light-rain estimate -> previous calibration kept
2023-01-06,0.91   # IN2023_E01  residual +0.09 dB (6 d)
2023-01-25,0.59   # IN2023_V01  residual +0.41 dB (19 d)
2023-03-23,0.93   # IN2023_V02  residual +0.07 dB (37 d)
2023-05-10,0.18   # IN2023_V03  residual +0.32 dB (13 d)
2023-06-05,0.17   # IN2023_V04  residual +0.33 dB (12 d)
2023-06-28,0.94   # IN2023_V05  residual -0.44 dB (22 d)
2023-09-21,0.50   # IN2023_E03  no light-rain estimate -> previous calibration kept
2023-10-08,0.07   # IN2023_V06  residual +0.43 dB (12 d)
2023-11-02,0.50   # IN2023_T01  no light-rain estimate -> previous calibration kept
2023-11-14,0.50   # IN2023_V07  no light-rain estimate -> previous calibration kept
2024-03-09,0.50   # IN2024_T01  no light-rain estimate -> previous calibration kept
2024-05-01,0.94   # IN2024_V03  residual -0.44 dB (30 d)
2024-06-07,0.64   # IN2024_V04  residual -0.14 dB (25 d)
2024-07-07,-1.23  # IN2024_T02  residual +1.73 dB (6 d)  [beyond 1 dB QC limit]
2025-01-06,-0.08  # IN2025_V02  residual +0.58 dB (17 d)
2025-04-10,0.36   # IN2025_V03  residual +0.14 dB (21 d)
2025-06-26,0.43   # IN2025_V04  residual +0.07 dB (2 d)
2025-07-02,0.42   # IN2025_V07  residual +0.08 dB (3 d)
2025-07-26,0.13   # IN2025_E02  residual +0.37 dB (4 d)
2025-08-10,0.65   # IN2025_V05  residual -0.15 dB (14 d)
2025-10-10,-1.39  # IN2025_V06  residual +1.89 dB (11 d)  [beyond 1 dB QC limit]
2026-03-05,-0.26  # IN2026_T01  residual +0.76 dB (8 d)
2026-03-16,-0.53  # IN2026_V02  residual +1.03 dB (20 d)  [beyond 1 dB QC limit]
2026-04-23,-0.56  # IN2026_T02  residual +1.06 dB (12 d)  [beyond 1 dB QC limit]
2026-05-13,-0.28  # IN2026_V03  residual +0.78 dB (10 d)"""


def _load(table: str) -> "pd.Series":
    clean = "\n".join(line.split("#", 1)[0].strip() for line in table.splitlines())
    frame = pd.read_csv(io.StringIO(clean), sep=",", parse_dates=["date"], index_col="date")
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
