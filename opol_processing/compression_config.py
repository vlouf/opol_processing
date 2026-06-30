"""
Compression configuration for OPOL radar data.

Defines min/max ranges for each field used to calculate optimal
scale_factor and add_offset for int16 bit-packing with xarray encoding.

Configuration format:
    COMPRESSION_DICT = {
        "field_name": (data_min, data_max, bits, has_fill),
        ...
    }

Parameters:
    data_min: Minimum physical value (will be clipped)
    data_max: Maximum physical value (will be clipped)
    bits: Number of bits for integer storage (16 for int16, 8 for int8)
    has_fill: Whether to reserve one integer value for fill/invalid data
"""

COMPRESSION_DICT = {
    # Reflectivity fields - Rainbow packing: 8-bit, -31.5 to 95.5 dB, step 0.5
    # Replaces 16-bit encoding, saves 50% on these fields
    "reflectivity": (-31.5, 95.5, 8, True),
    "corrected_reflectivity": (-31.5, 95.5, 8, True),
    "total_power": (-31.5, 95.5, 8, True),
    "uncorrected_reflectivity": (-31.5, 95.5, 8, True),
    
    # Velocity fields - Rainbow packing: 8-bit, -30 to +30 m/s, step 0.236 m/s
    # Replaces 16-bit encoding, saves 50% on these fields
    "velocity": (-30, 30, 8, True),
    "corrected_velocity": (-100, 100, 16, True),
    
    # Differential reflectivity - Rainbow packing: 8-bit, -8 to +12 dB, step 0.079 dB
    "differential_reflectivity": (-8, 12, 8, True),
    "corrected_differential_reflectivity": (-8, 12, 8, True),
    
    # Differential phase (PhiDP) - Rainbow packing: 8-bit, 0 to 360°, step 1.417°
    "differential_phase": (-360, 360, 16, True),
    "uncorrected_differential_phase": (-360, 360, 16, True),
    
    # Specific differential phase (KDP) - Rainbow packing: 8-bit, -18 to +36 degree/km, step 0.213
    "specific_differential_phase": (-18, 36, 8, True),
    "corrected_specific_differential_phase": (-18, 36, 8, True),
    
    # Kdp - Rainbow packing: 8-bit, -18 to +36 degree/km
    "kdp": (-18, 36, 8, True),
    
    # Cross-correlation ratio (RhoHV) - Rainbow packing: 8-bit, 0 to 1.0, step 0.00394
    "cross_correlation_ratio": (0, 1.0, 8, True),
    "corrected_cross_correlation_ratio": (0, 1.0, 8, True),
    
    # Temperature (int8 instead of int16, range: -50 to +50°C)
    "temperature": (-50, 50, 8, False),
    
    # Hydrometeor classification (uint8, no compression)
    "radar_echo_classification": (0, 10, 8, False),
    
    # Rainfall and related fields (int16 from float32)
    "rainfall_rate": (0, 100, 16, False),
    "rain_rate_A": (0, 100, 16, False),
    "rain_rate_B": (0, 100, 16, False),
    "rain_rate_Z": (0, 100, 16, False),
    "rain_rate_KDP": (0, 100, 16, False),
    "rain_rate_ZDR": (0, 100, 16, False),
    "rain_rate_HYBRID": (0, 100, 16, False),
    
    # DSD parameters (int16 from float32)
    "Nw": (0, 1000, 16, False),
    "D0": (0, 10, 16, False),
    
    # Spectrum width - Rainbow packing: 8-bit, 0 to 15 m/s, step 0.059 m/s
    "spectrum_width": (0, 15, 8, True),
    
    # SNR - Rainbow packing: 8-bit, 0 to 127 dB, step 0.5 dB
    "signal_to_noise_ratio": (0, 127, 8, True),
    "snr": (0, 127, 8, True),
    
    # Other fields
    "path_integrated_attenuation": (0, 10, 16, False),
    "path_integrated_differential_attenuation": (-2, 5, 16, False),
    # Signal quality index (DOP) - Rainbow packing: 8-bit, 0 to 1.0, step 0.00394
    "signal_quality_index": (0, 1.0, 8, True),
    "snow_rate": (0, 100, 16, False),
}

# Zlib compression level for all variables
# 4 = balanced (default), 9 = maximum compression (slower, more space savings)
ZLIB_COMPLEVEL = 9

# Fields to keep (if set, only these fields are written to output)
# Set to None to keep all fields
KEEP_FIELDS = None
# KEEP_FIELDS = [
#     "corrected_reflectivity",
#     "corrected_velocity",
#     "corrected_differential_reflectivity",
#     "corrected_cross_correlation_ratio",
#     "temperature",
#     "radar_echo_classification",
#     "rainfall_rate",
# ]
