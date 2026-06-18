# OceanPOL processing pipeline 🚢📡

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
`differential_reflectivity`, `corrected_differential_reflectivity`, `cross_correlation_ratio`,
`velocity`, `corrected_velocity`, `differential_phase`,
`corrected_differential_phase`, `corrected_specific_differential_phase`,
`temperature`, `radar_echo_classification`, `radar_estimated_rain_rate`,
`radar_estimated_snow_rate`, `normalized_intercept_parameter`,
`median_volume_diameter`.

`corrected_reflectivity` is cleaned + calibrated **without** attenuation;
`attenuation_corrected_reflectivity` is the same field with the Z-PHI
path-integrated attenuation added.

## Requirements

- [arm_pyart](https://github.com/ARM-DOE/pyart)
- [numpy](https://numpy.org/)
- [numba](https://numba.pydata.org/)
- [scipy](https://scipy.org/)
- [pandas](http://pandas.pydata.org/)
- [xarray](https://xarray.pydata.org/en/stable/)
- [cftime](https://unidata.github.io/cftime/)
- [netCDF4](https://unidata.github.io/netcdf4-python/)
- [csu_radartools](https://github.com/CSU-Radarmet/CSU_RadarTools)
- [unravel](https://github.com/vlouf/dealias)
- [phido](https://github.com/jordanbrook/phido)

## Usage

Single file:

```sh
python scripts/radar_single.py -i /path/to/PPIVol.hdf -o /path/to/output
```

Batch (a directory of ODIM volumes):

```sh
python scripts/dask_pack.py -i /path/to/indir -o /path/to/output
```

Both write `*.cfradial.nc` under `<output>/ppi/<YYYYMMDD>/`.
