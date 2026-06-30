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
import xarray as xr


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


def decimate_rays_to_1degree(radar):
    """
    Decimate radar rays from 0.5° to 1° azimuth resolution.
    Groups consecutive ray pairs and averages their data using masked array operations.
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Input radar with 0.5° azimuth resolution.
        
    Returns
    -------
    radar : pyart.core.Radar
        Radar with 1° azimuth resolution (decimated).
    """
    n_rays = radar.nrays
    n_pairs = (n_rays + 1) // 2  # Number of ray pairs
    
    # ===== BEFORE DECIMATION CHECK =====
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
    
    # ===== AFTER DECIMATION CHECK =====
    sample_shape_after = radar.fields[sample_field]['data'].shape
    print(f"  [decimate check] After: nrays={radar.nrays}, sample field '{sample_field}' shape={sample_shape_after}")
    print(f"  [decimate check] Dimension match: rows={sample_shape_after[0] == radar.nrays}, cols={sample_shape_after[1] == sample_shape_before[1]}")
    
    return radar


def write_compressed_cfradial(radar, outfilename):
    """
    Write PyART Radar object to CF/Radial netCDF4 with xarray encoding compression.
    
    Uses temporary file approach:
    1. Write with PyART (uncompressed) to temp file
    2. Read with xarray
    3. Apply encoding dict (scale_factor/add_offset for int types)
    4. Write final file with gzip compression
    
    Parameters
    ----------
    radar : pyart.core.Radar
        Radar object to write.
    outfilename : str
        Output netCDF file path.
        
    Returns
    -------
    size_saved_mb : float
        Size reduction in MB
    size_saved_pct : float
        Size reduction as percentage
    """
    # Create temporary file for uncompressed intermediate
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
        temp_filename = tmp.name
    
    try:
        # Step 1: Write uncompressed CF/Radial with PyART
        pyart.io.write_cfradial(temp_filename, radar, format='NETCDF4')
        uncompressed_size = os.path.getsize(temp_filename)
        
        # Step 2: Define encoding parameters for each variable type
        encoding = {
            # Reflectivity variables: int16 with scale=0.5, offset=-32
            'corrected_reflectivity': {
                'dtype': 'int16',
                'scale_factor': 0.5,
                'add_offset': -32.0,
                'zlib': True,
                'complevel': 4,
            },
            'reflectivity': {
                'dtype': 'int16',
                'scale_factor': 0.5,
                'add_offset': -32.0,
                'zlib': True,
                'complevel': 4,
            },
            'attenuation_corrected_reflectivity': {
                'dtype': 'int16',
                'scale_factor': 0.5,
                'add_offset': -32.0,
                'zlib': True,
                'complevel': 4,
            },
            'total_power': {
                'dtype': 'int16',
                'scale_factor': 0.5,
                'add_offset': -32.0,
                'zlib': True,
                'complevel': 4,
            },
            # Temperature: int16 with scale=0.1, offset=-50
            'temperature': {
                'dtype': 'int16',
                'scale_factor': 0.1,
                'add_offset': -50.0,
                'zlib': True,
                'complevel': 4,
            },
            # Differential reflectivity: int16 with scale=0.05, offset=-6
            'differential_reflectivity': {
                'dtype': 'int16',
                'scale_factor': 0.05,
                'add_offset': -6.0,
                'zlib': True,
                'complevel': 4,
            },
            'corrected_differential_reflectivity': {
                'dtype': 'int16',
                'scale_factor': 0.05,
                'add_offset': -6.0,
                'zlib': True,
                'complevel': 4,
            },
            # Differential phase and KDP: float32 with compression only
            'differential_phase': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'corrected_differential_phase': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'corrected_specific_differential_phase': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            # Velocity: float32 with compression only (no re-encoding)
            'velocity': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'corrected_velocity': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            # Cross correlation ratio: int16 with scale=0.01, offset=0
            'cross_correlation_ratio': {
                'dtype': 'int16',
                'scale_factor': 0.01,
                'add_offset': 0.0,
                'zlib': True,
                'complevel': 4,
            },
            # Other variables: float32 with compression only
            'radar_estimated_rain_rate': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'radar_estimated_snow_rate': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'normalized_intercept_parameter': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'median_volume_diameter': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'signal_to_noise_ratio': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'spectrum_width': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'signal_quality_index': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
            'path_integrated_differential_attenuation': {
                'dtype': 'float32',
                'zlib': True,
                'complevel': 4,
            },
        }
        
        # Step 3: Read uncompressed file with xarray
        # Use decode_times=False to preserve original netCDF time encoding and avoid nanosecond conversion
        ds = xr.open_dataset(temp_filename, engine='netcdf4', decode_times=False)
        
        # Step 4: Build encoding dict for variables that exist in the dataset
        # Only apply encoding to variables that are actually in the file
        final_encoding = {}
        for var_name in ds.data_vars:
            if var_name in encoding:
                final_encoding[var_name] = encoding[var_name]
            else:
                # Default: just add compression for any other variables
                final_encoding[var_name] = {'zlib': True, 'complevel': 4}
        
        # Coordinate variables: use gzip compression only
        for coord_name in ds.coords:
            if coord_name not in final_encoding:
                # For all coordinate variables (including time), just add compression
                # Time is preserved in its original encoding from PyART via decode_times=False
                final_encoding[coord_name] = {'zlib': True, 'complevel': 4}
        
        # Step 5: Write with encoding - xarray handles scale_factor/add_offset automatically
        ds.to_netcdf(
            outfilename,
            encoding=final_encoding,
            engine='netcdf4',
            unlimited_dims=['time'] if 'time' in ds.dims else [],
        )
        ds.close()
        
        # Step 6: Calculate compression savings
        compressed_size = os.path.getsize(outfilename)
        size_reduction_mb = (uncompressed_size - compressed_size) / (1024 * 1024)
        size_reduction_pct = 100 * (1 - compressed_size / uncompressed_size) if uncompressed_size > 0 else 0
        
        return size_reduction_mb, size_reduction_pct
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
