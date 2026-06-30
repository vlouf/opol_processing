# OceanPOL processing pipeline

Processing of the dual-polarisation OceanPOL (RV *Investigator*) radar data into
the public **CF/Radial netCDF Level 1b** product.

The science is kept in sync with the [`oceanpol_kit`](https://github.com/vlouf/oceanpol_kit)
reference (alias-based field resolution, date-indexed calibration, robust
ERA5/ERA5T + bright-band temperature, SQI-coherence velocity censoring with
UNRAVEL, precip-gated PHIDO phase, Z-PHI attenuation, Southern-Ocean rainfall,
DSD/snow/HID). `oceanpol_kit` writes the compact ODIM HDF5 product; this pipeline
emits the verbose CF/Radial netCDF that downstream users rely on.

> **Note:** algorithm changes here must be mirrored in `oceanpol_kit` (and vice
> versa). Each ported module carries a header noting its reference origin.

## Output variables (CF/Radial verbose names)

`total_power`, `reflectivity`, `corrected_reflectivity`,
`attenuation_corrected_reflectivity`,
`differential_reflectivity`, `corrected_differential_reflectivity`,
`cross_correlation_ratio`, `velocity`, `corrected_velocity`,
`differential_phase`, `corrected_differential_phase`,
`corrected_specific_differential_phase`, `temperature`,
`radar_echo_classification`, `radar_estimated_rain_rate`,
`radar_estimated_snow_rate`, `normalized_intercept_parameter`,
`median_volume_diameter`, `signal_to_noise_ratio`, `spectrum_width`,
`signal_quality_index`.

`corrected_reflectivity` is cleaned + calibrated **without** attenuation;
`attenuation_corrected_reflectivity` is the same field with the Z-PHI
path-integrated attenuation added.

## Compression

Output files use aggressive integer packing (Rainbow 8-bit encoding) with
gzip level 9, reducing file size from ~200 MB to ~65 MB with no meaningful
loss of precision:

| Field group | Storage | Scale (step) |
|---|---|---|
| Reflectivity (dBZ) | `byte` (int8) | 0.5 dBZ |
| Velocity, corrected velocity (m/s) | `byte`/`short` | 0.236 / 0.006 m/s |
| ZDR (dB) | `byte` | 0.079 dB |
| KDP (°/km) | `byte` | 0.213 °/km |
| RhoHV | `byte` | 0.004 |
| Spectrum width (m/s) | `byte` | 0.059 m/s |
| SNR / SQI | `byte` | 0.5 dB / 0.004 |
| Differential phase (°) | `short` | 0.005 ° |
| Rainfall, DSD, snow, other | `float32` | — |

Encoding ranges and bit-depths are configurable in
`opol_processing/compression_config.py`.

## Azimuth decimation

Volumes scanned at 0.5° azimuth resolution are decimated to 1° by averaging
adjacent ray pairs before writing, halving the time and range dimensions.

## Requirements

- [arm_pyart](https://github.com/ARM-DOE/pyart)
- [numpy](https://numpy.org/)
- [numba](https://numba.pydata.org/)
- [scipy](https://scipy.org/)
- [pandas](http://pandas.pydata.org/)
- [cftime](https://unidata.github.io/cftime/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [dask](https://www.dask.org/)
- [csu_radartools](https://github.com/CSU-Radarmet/CSU_RadarTools)
- [unravel](https://github.com/vlouf/dealias)
- [phido](https://github.com/jordanbrook/phido)

## Installation

```sh
pip install -e .
```

This registers the `opol_process` CLI entry point.

## Usage

### Single file (CLI)

```sh
opol_process /path/to/PPIVol.hdf /path/to/output_dir
opol_process /path/to/PPIVol.hdf /path/to/output_dir --no-dealiasing
opol_process /path/to/PPIVol.hdf /path/to/output_dir --debug
```

Output is written to `<output_dir>/<input_stem>.cfradial.nc`.

### Single file (Python API)

```python
from opol_processing import process_and_save

process_and_save(
    "9776HUB-PPIVol-20260518-153000.hdf",
    output_filename="/path/to/output.cfradial.nc",
    do_dealiasing=True,
    debug=True,
)
```

Returns `None` and prints a message for unsuitable scans (fewer than 10 sweeps).

### Batch processing (distributed)

```sh
python scripts/dask_pack.py \
    --input  /path/to/voyage_data \
    --output /path/to/output_dir \
    --workers 16 \
    --chunk-size 64
```

Output is written to `<output_dir>/processed/<YYYYMMDD>/<filename>.cfradial.nc`.
Processing state is stored under `<output_dir>/.processing/`:

| Path | Contents |
|---|---|
| `completed.json` | Successfully processed files with timestamps |
| `failed.json` | Failed files with error messages |
| `checkpoint.txt` | Last successfully processed filename |
| `logs/processing_*.log` | Timestamped run logs |

Re-running the same command resumes from where it left off, skipping already
completed files. Files skipped by the pipeline (e.g. too few sweeps) are
tracked separately and do not pollute `completed.json`.

#### Batch options

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Input directory (searched recursively for `.hdf`) |
| `--output` | *(required)* | Output root directory |
| `--workers` | `16` | Parallel Dask workers |
| `--chunk-size` | `64` | Files processed per Dask chunk |
| `--no-dealiasing` | — | Disable UNRAVEL velocity dealiasing |
| `--debug` | — | Print per-step timings |

## Project structure

```
opol_processing/
    __init__.py              public API (production_line, process_and_save)
    production.py            main pipeline orchestration
    utils.py                 compression, ray decimation, timing helpers
    compression_config.py    field encoding ranges and gzip level
    cli.py                   opol_process entry point
    attenuation.py           Z-PHI attenuation correction
    calibration.py           reflectivity calibration
    filtering.py             signal quality / censoring
    hydrometeors.py          HID, rainfall, DSD, snow
    phase.py                 PhiDP/KDP processing (PHIDO)
    radar_codes.py           I/O, sweep ordering, metadata
    temperature.py           ERA5 / bright-band temperature profile
scripts/
    dask_pack.py             distributed batch processing
    radar_single.py          legacy single-file script
```
