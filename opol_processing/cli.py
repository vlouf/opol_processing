#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OPOL radar file processing CLI.

Simple command-line interface for processing ocean polarimetric radar files.

Usage:
    opol_process /path/to/input.hdf /path/to/output_dir
    opol_process /path/to/input.hdf /path/to/output_dir --no-dealiasing
    opol_process /path/to/input.hdf /path/to/output_dir --debug
"""

import os
import sys
import argparse
import warnings
from pathlib import Path


def main():
    """
    Process a single OPOL radar file.
    """
    parser = argparse.ArgumentParser(
        description="Process OPOL radar PPI files with quality control, filtering, "
        "attenuation correction, dealiasing, hydrometeors, and rainfall estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opol_process input.hdf output_dir
  opol_process input.hdf output_dir --no-dealiasing
  opol_process input.hdf output_dir --debug
        """,
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input .hdf radar file",
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to output directory (will be created if it doesn't exist)",
    )

    parser.add_argument(
        "--no-dealiasing",
        dest="dealiasing",
        action="store_false",
        default=True,
        help="Disable velocity dealiasing (default: enabled)",
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Print per-step processing timings",
    )

    args = parser.parse_args()

    # Validate input file
    input_file = Path(args.input_file).resolve()
    if not input_file.exists():
        parser.error(f"Input file not found: {input_file}")
    if not input_file.is_file():
        parser.error(f"Input is not a file: {input_file}")

    # Validate output directory
    output_dir = Path(args.output_dir).resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        parser.error(f"Cannot create output directory {output_dir}: {e}")

    # Print processing info
    print("=" * 80)
    print("OPOL RADAR PROCESSING")
    print("=" * 80)
    print(f"Input file:       {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Dealiasing:       {'enabled' if args.dealiasing else 'disabled'}")
    print(f"Debug:            {'enabled' if args.debug else 'disabled'}")
    print("=" * 80)
    print()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import opol_processing

            # Generate output filename from input filename
            output_basename = input_file.stem + ".cfradial.nc"
            output_file = output_dir / output_basename

            opol_processing.process_and_save(
                str(input_file),
                output_filename=str(output_file),
                do_dealiasing=args.dealiasing,
                use_csu=True,
                debug=args.debug,
                exist_ok=True,
            )

        print()
        print("=" * 80)
        print("Processing completed successfully!")
        print(f"Output file: {output_file}")
        print("=" * 80)
        return 0

    except Exception as e:
        print()
        print("=" * 80)
        print(f"ERROR: Processing failed - {type(e).__name__}: {e}")
        print("=" * 80)
        if args.debug:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
