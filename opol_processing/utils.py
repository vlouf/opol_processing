"""
Utility functions for OPOL processing pipeline.

@project: OCEANPol
@title: utils
@author: Valentin Louf
@email: valentin.louf@bom.gov.au
@institution: Bureau of Meteorology and Monash University
"""

import os
import time
import tempfile
import numpy as np
import pyart

from opol_processing.compression_config import COMPRESSION_DICT, ZLIB_COMPLEVEL, KEEP_FIELDS


def meta(data, **kwargs) -> dict:
    """Build a Py-ART field dictionary."""
    field = {"data": data}
    field.update(kwargs)
    return field


def toc(label: str, t0: float, debug: bool) -> float:
    """Print the elapsed time for a step (when debug) and return a fresh tic."""
    now = time.time()
    if debug:
        print(f"  [{label:<22}] {now - t0:7.3f} s")
    return now


def determine_optimal_packing(data_min, data_max, bits=16):
    """
    Determine optimal scale_factor and add_offset for packing float data into integers.

    Uses the full signed integer range with one slot reserved for the fill
    sentinel, matching Rainbow 8-bit encoding (254 usable levels for int8
    gives 0.5 dBZ step for reflectivity).

    Parameters
    ----------
    data_min : float
        Minimum physical value to represent.
    data_max : float
        Maximum physical value to represent.
    bits : int
        Integer storage width (8 → int8, 16 → int16).

    Returns
    -------
    scale_factor : float
    add_offset : float

    Notes
    -----
    CF/NetCDF decoding: ``physical = integer * scale_factor + add_offset``.
    Valid data integers span ``-(2^(bits-1)-1)`` to ``+(2^(bits-1)-1)``;
    the most-negative integer (e.g. -128 for int8) is the fill sentinel.
    """
    scale_factor = (data_max - data_min) / (2 ** bits - 2)
    add_offset = (data_min + data_max) / 2
    return float(scale_factor), float(add_offset)




def decimate_rays_to_1degree(radar, debug=False):
    """
    Decimate radar rays from 0.5° to 1° azimuth resolution.
    Groups consecutive ray pairs and averages their data using masked array operations.

    Parameters
    ----------
    radar : pyart.core.Radar
        Input radar with 0.5° azimuth resolution.
    debug : bool
        Print before/after dimension check.

    Returns
    -------
    radar : pyart.core.Radar
        Radar with 1° azimuth resolution (decimated).
    """
    n_rays = radar.nrays
    n_pairs = (n_rays + 1) // 2  # Number of ray pairs

    if debug:
        sample_field = list(radar.fields.keys())[0]
        sample_shape_before = radar.fields[sample_field]['data'].shape
        print(f"  [decimate check] Before: nrays={n_rays}, sample field '{sample_field}' shape={sample_shape_before}")
        print(f"  [decimate check] Expected after: nrays={n_pairs}, expected shape=({n_pairs}, {sample_shape_before[1]})")
    
    # Aggregate field data by averaging ray pairs
    # Use list(radar.fields.keys()) to avoid dict iteration issues
    for field_name in list(radar.fields.keys()):
        data = radar.fields[field_name]['data']  # Shape: (nrays, ngates)
        
        # Reshape to group rays in pairs: (n_pairs, 2, ngates)
        # Pad with last ray if odd number of rays
        if n_rays % 2 == 1:
            # Pad the last ray
            last_ray = data[-1:, :]
            data_padded = np.ma.vstack([data, last_ray])
        else:
            data_padded = data
        
        # Reshape to (n_pairs, 2, ngates) and average along axis 1
        data_reshaped = data_padded.reshape(n_pairs, 2, data.shape[1])
        
        # Use masked array mean to properly handle NaN/masked values
        aggregated = np.ma.mean(data_reshaped, axis=1)
        
        # Update field data
        radar.fields[field_name]['data'] = aggregated
    
    # Decimate coordinate arrays (keep every 2nd ray, or average pairs)
    radar.azimuth['data'] = radar.azimuth['data'][::2]
    radar.elevation['data'] = radar.elevation['data'][::2]
    radar.time['data'] = radar.time['data'][::2]
    
    # Update sweep indices (they were halved after decimation)
    if 'data' in radar.sweep_start_ray_index:
        radar.sweep_start_ray_index['data'] = radar.sweep_start_ray_index['data'] // 2
    if 'data' in radar.sweep_end_ray_index:
        radar.sweep_end_ray_index['data'] = ((radar.sweep_end_ray_index['data'] + 1) // 2) - 1
    
    # Update radar.nrays to match new time array length
    # PyART stores nrays as simple attribute set from len(time['data']) during __init__
    # We must update it explicitly after decimation
    radar.nrays = len(radar.time['data'])

    if debug:
        sample_field = list(radar.fields.keys())[0]
        sample_shape_after = radar.fields[sample_field]['data'].shape
        print(f"  [decimate check] After: nrays={radar.nrays}, sample field '{sample_field}' shape={sample_shape_after}")
        print(f"  [decimate check] Dimension match: rows={sample_shape_after[0] == radar.nrays}, cols={sample_shape_after[1] == sample_shape_before[1]}")

    return radar


def write_compressed_cfradial(radar, outfilename):
    """
    Write PyART Radar object to compressed CF/Radial netCDF4.
    
    Uses configuration-driven compression with optimal scale_factor/add_offset
    calculation. Prepares radar object with encoding metadata, then writes
    directly with PyART.
    
    Single-write approach:
    1. For each field in COMPRESSION_DICT:
       - Get min/max ranges
       - Calculate optimal scale_factor/add_offset
       - Clamp data to [min, max]
       - Mask invalid values (NaN, inf)
       - Set encoding metadata on field
    2. Call pyart.io.write_cfradial() with NETCDF4 format (includes gzip compression)
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object to write.
    outfilename : str
        Output netCDF file path.
        
    """
    # Step 1: Filter fields if KEEP_FIELDS is specified
    if KEEP_FIELDS is not None:
        fields_to_remove = [f for f in radar.fields.keys() if f not in KEEP_FIELDS]
        for field in fields_to_remove:
            radar.fields.pop(field, None)
    
    # Step 2: Prepare compression metadata for each field
    for field_name in list(radar.fields.keys()):
        if field_name not in COMPRESSION_DICT:
            # Field not in compression config - skip encoding
            continue
        
        config = COMPRESSION_DICT[field_name]
        
        # Skip if config indicates no compression (None values)
        if config[0] is None or config[1] is None:
            # Field uses default float32 with gzip only
            # Don't set any special encoding, PyART will handle compression
            continue
        
        data_min, data_max, bits, has_fill = config
        
        # Get field data (handle both regular arrays and masked arrays)
        field_data = radar.fields[field_name]['data']
        
        # Copy data to avoid modifying original
        data_to_pack = np.copy(field_data)
        
        # Clamp to valid range
        data_to_pack = np.clip(data_to_pack, data_min, data_max)
        
        # Mask invalid values (NaN, inf)
        data_to_pack[np.isnan(data_to_pack)] = data_min
        data_to_pack[np.isinf(data_to_pack)] = data_min
        
        # Calculate optimal packing parameters
        scale_factor, add_offset = determine_optimal_packing(data_min, data_max, bits=bits)
        
        # Fill integer = the packed integer that represents data_min.
        # Masked / invalid data has already been set to data_min above, so they
        # share the same integer sentinel.  _Write_as_dtype tells PyART to
        # quantize the float array on write — no manual cast needed (avoids
        # int8 overflow / saturation when values round to ±128).
        packed_fill = int(round((data_min - add_offset) / scale_factor))
        if bits == 16:
            write_dtype = 'i2'
            fill_int = np.int16(packed_fill)
        elif bits == 8:
            write_dtype = 'i1'
            fill_int = np.int8(packed_fill)
        else:
            continue

        # Replace masked values in the float array with data_min so they map
        # to the same packed integer as the fill sentinel.
        if isinstance(field_data, np.ma.MaskedArray):
            mask = np.ma.getmaskarray(field_data)
            data_to_pack[mask] = data_min

        # Keep data as float; PyART applies scale/offset and writes as write_dtype
        radar.fields[field_name]['data'] = data_to_pack
        radar.fields[field_name]['scale_factor'] = scale_factor
        radar.fields[field_name]['add_offset'] = add_offset
        radar.fields[field_name]['_FillValue'] = fill_int
        radar.fields[field_name]['_Write_as_dtype'] = write_dtype

    # Step 3: Write
    pyart.io.write_cfradial(outfilename, radar, format='NETCDF4')
