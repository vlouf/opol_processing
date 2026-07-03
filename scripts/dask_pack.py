"""
Distributed radar processing with voyage-based organization.

Processes OPOL ODIM .hdf files organized by voyage with resumable,
checkpointed execution. Tracks progress via JSON to enable recovery
from interruptions and skip already-processed files.

Features:
- Voyage-based directory organization
- Resumable processing (load checkpoint on restart)
- Progress tracking (completed, failed, skipped files)
- Atomic checkpoint updates
- Comprehensive error logging
- Per-voyage isolation

@title: opol_processing dask_pack
@author: Valentin Louf <valentin.louf@bom.gov.au>
@institution: Bureau of Meteorology and Monash University

.. autosummary::
    :toctree: generated/

    VoyageProcessor
    process_buffer
    main
"""

import os
import re
import sys
import json
import glob
import time
import argparse
import warnings
import traceback
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import dask.bag as db
import opol_processing
import unravel

class VoyageProcessor:
    """
    Manages processing state and progress tracking for a single voyage.

    Handles:
    - Progress tracking (completed, failed, skipped files)
    - Atomic checkpoint updates
    - Work queue building (filtering completed files)
    - Logging and summary reporting
    """

    def __init__(self, input_dir: str, output_dir: str, chunk_size: int = 64,
                 do_dealiasing: bool = True, debug: bool = False):
        """
        Initialize voyage processor.

        Parameters
        ----------
        input_dir : str
            Path to read-only input directory containing raw .hdf files
        output_dir : str
            Path to output directory for processed files and tracking
        chunk_size : int
            Number of files to process per chunk
        do_dealiasing : bool
            Enable velocity dealiasing
        debug : bool
            Print debug output
        """
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.processed_dir = self.output_dir / "processed"
        self.tracking_dir = self.output_dir / ".processing"

        self.chunk_size = chunk_size
        self.do_dealiasing = do_dealiasing
        self.debug = debug

        # Tracking files
        self.completed_file = self.tracking_dir / "completed.json"
        self.failed_file = self.tracking_dir / "failed.json"
        self.checkpoint_file = self.tracking_dir / "checkpoint.txt"
        self.log_dir = self.tracking_dir / "logs"

        # State
        self.completed = {}
        self.failed = {}
        self.current_checkpoint = None

        # Setup
        self._validate_paths()
        self._init_tracking()
        self._load_progress()
        self._setup_logging()

    def _validate_paths(self):
        """Validate and create necessary directories."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

        if not os.access(self.input_dir, os.R_OK):
            raise PermissionError(f"Input directory is not readable: {self.input_dir}")

        # Output dir must exist or be creatable (and writable)
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise PermissionError(f"Cannot create output directory {self.output_dir}: {e}")

        if not os.access(self.output_dir, os.W_OK):
            raise PermissionError(f"Output directory is not writable: {self.output_dir}")

    def _init_tracking(self):
        """Create tracking directory and initialize tracking files."""
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking files if they don't exist
        if not self.completed_file.exists():
            self._write_json(self.completed_file, {})
        if not self.failed_file.exists():
            self._write_json(self.failed_file, {})

    def _load_progress(self):
        """Load existing progress from tracking files."""
        if self.completed_file.exists():
            with open(self.completed_file, 'r') as f:
                self.completed = json.load(f)

        if self.failed_file.exists():
            with open(self.failed_file, 'r') as f:
                self.failed = json.load(f)

        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                self.current_checkpoint = f.read().strip()

    def _setup_logging(self):
        """Setup logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"processing_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _write_json(filepath: Path, data: dict):
        """Atomically write JSON file."""
        # Write to temp file first, then rename (atomic)
        temp_file = filepath.with_suffix(filepath.suffix + '.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.replace(filepath)

    @staticmethod
    def _extract_date_from_filename(basename: str) -> Optional[str]:
        """
        Extract YYYYMMDD date from filename using first 8 consecutive digits.

        Example: "9776HUB-PPIVol-20260518-153000.hdf" -> "20260518"

        Parameters
        ----------
        basename : str
            Filename (not full path)

        Returns
        -------
        date_str : str or None
            8-digit date string (YYYYMMDD) or None if not found
        """
        match = re.search(r'\d{8}', basename)
        if match:
            return match.group(0)
        return None

    def build_work_queue(self) -> List[str]:
        """
        Build list of input files to process.

        Filters out already-completed and failed files.

        Returns
        -------
        work_queue : list of str
            Input file paths to process
        """
        # Get all .hdf files in input directory
        all_files = sorted(glob.glob(str(self.input_dir / "*.hdf")))
        if not all_files:
            # Try recursive search
            all_files = sorted(glob.glob(str(self.input_dir / "**/*.hdf"), recursive=True))

        # Filter to only files not yet completed
        work_queue = []
        skipped = 0

        for input_file in all_files:
            basename = os.path.basename(input_file)

            if basename in self.completed:
                skipped += 1
                continue

            if basename in self.failed:
                # Skip recently failed (can be retried with --retry flag)
                skipped += 1
                continue

            work_queue.append(input_file)

        self.logger.info(f"Found {len(all_files)} total files: {len(work_queue)} to process, {skipped} skipped")
        return work_queue

    def generate_output_filename(self, input_file: str) -> str:
        """
        Generate output filename from input filename with date-based organization.

        Extracts YYYYMMDD from filename and creates directory structure:
        processed/YYYYMMDD/filename.cfradial.nc

        Parameters
        ----------
        input_file : str
            Input .hdf filename

        Returns
        -------
        output_file : str
            Full path to output .cfradial.nc file
        """
        basename = os.path.basename(input_file)

        # Extract date from filename (first 8 consecutive digits)
        date_str = self._extract_date_from_filename(basename)
        if date_str is None:
            # Fallback: use current date if extraction fails
            date_str = datetime.now().strftime("%Y%m%d")

        # Replace extension: .hdf -> .cfradial.nc
        output_basename = basename.replace('.hdf', '.cfradial.nc')

        # Create date subdirectory structure: processed/YYYYMMDD/filename.nc
        date_dir = self.processed_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)

        return str(date_dir / output_basename)

    def record_success(self, input_file: str, output_file: str):
        """Record successful processing."""
        basename = os.path.basename(input_file)
        self.completed[basename] = {
            'output': output_file,
            'timestamp': datetime.now().isoformat()
        }
        # Remove from failed if it was previously failed
        self.failed.pop(basename, None)
        self._write_json(self.completed_file, self.completed)

    def record_failure(self, input_file: str, error_msg: str):
        """Record failed processing."""
        basename = os.path.basename(input_file)
        self.failed[basename] = {
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        self._write_json(self.failed_file, self.failed)

    def update_checkpoint(self, input_file: str):
        """Update processing checkpoint."""
        with open(self.checkpoint_file, 'w') as f:
            f.write(os.path.basename(input_file))

    def print_summary(self, start_time: float):
        """Print processing summary."""
        elapsed = time.time() - start_time
        completed_count = len(self.completed)
        failed_count = len(self.failed)

        self.logger.info("=" * 80)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Input: {self.input_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Total completed: {completed_count}")
        self.logger.info(f"Total failed: {failed_count}")
        self.logger.info(f"Total elapsed: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")

        if completed_count > 0:
            rate = completed_count / elapsed
            self.logger.info(f"Processing rate: {rate:.2f} files/second")

        if failed_count > 0:
            self.logger.warning("Failed files:")
            for basename, info in self.failed.items():
                self.logger.warning(f"  - {basename}: {info['error']}")

        self.logger.info("=" * 80)


def process_buffer(input_file: str, processor: VoyageProcessor) -> Tuple[str, str, str, Optional[str]]:
    """
    Process a single file with error handling.

    Wraps production_line, catches exceptions, and returns status.
    Note: Does NOT update tracking files (to avoid race conditions in parallel Dask).
    Tracking updates happen serially in main thread after all workers complete.

    Parameters
    ----------
    input_file : str
        Input .hdf file path
    processor : VoyageProcessor
        Voyage processor instance

    Returns
    -------
    tuple
        (input_file, output_file, status, error_msg)
        - status: "success", "failed", "skipped"
        - error_msg: None on success, error string on failure
    """
    try:
        output_file = processor.generate_output_filename(input_file)

        # Check if output already exists (and is in completed tracking)
        basename = os.path.basename(input_file)
        if basename in processor.completed:
            return (input_file, output_file, "skipped", None)

        # Process file
        result = opol_processing.process_and_save(
            input_file,
            output_filename=output_file,
            do_dealiasing=processor.do_dealiasing,
            use_csu=True,  # Always enabled (no alternative)
            debug=processor.debug,
            exist_ok=True
        )

        # process_and_save returns None for unsuitable scans (e.g. < 10 sweeps)
        # without raising — treat as skipped, not success.
        if result is None:
            return (input_file, "", "skipped", "scan skipped by production_line (e.g. too few sweeps)")

        # Return success (tracking updated serially in main thread)
        return (input_file, output_file, "success", None)

    except FileExistsError as e:
        # Output already exists, might be from previous run
        output_file = processor.generate_output_filename(input_file)
        return (input_file, output_file, "success", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        # Return failure (tracking updated serially in main thread)
        return (input_file, "", "failed", error_msg)


def chunks(items: list, n: int):
    """
    Yield successive n-sized chunks from list.

    Parameters
    ----------
    items : list
        List to chunk
    n : int
        Chunk size

    Yields
    ------
    list
        Chunks of size n
    """
    for i in range(0, len(items), n):
        yield items[i : i + n]


def main():
    """
    Main processing loop.

    Processes voyage files in chunks using Dask.
    """
    parser = argparse.ArgumentParser(
        description="Distributed OPOL radar processing with separate input/output directories."
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to read-only input directory containing .hdf files"
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        required=True,
        help="Path to output directory for processed files and tracking"
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=64,
        help="Number of files to process per chunk (default: 64)"
    )
    parser.add_argument(
        "--workers",
        dest="n_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)"
    )
    parser.add_argument(
        "--do-dealiasing",
        dest="do_dealiasing",
        action="store_true",
        default=True,
        help="Enable velocity dealiasing (default: True)"
    )
    parser.add_argument(
        "--no-dealiasing",
        dest="do_dealiasing",
        action="store_false",
        help="Disable velocity dealiasing"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print debug output"
    )

    args = parser.parse_args()

    # Initialize processor
    processor = VoyageProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        do_dealiasing=args.do_dealiasing,
        debug=args.debug
    )

    processor.logger.info("=" * 80)
    processor.logger.info("OPOL DISTRIBUTED PROCESSING")
    processor.logger.info("=" * 80)
    processor.logger.info(f"Input directory: {processor.input_dir}")
    processor.logger.info(f"Output directory: {processor.output_dir}")
    processor.logger.info(f"Processed files: {processor.processed_dir}")
    processor.logger.info(f"Tracking: {processor.tracking_dir}")
    processor.logger.info(f"Chunk size: {args.chunk_size}")
    processor.logger.info(f"Workers: {args.n_workers}")
    processor.logger.info(f"Dealiasing: {args.do_dealiasing}")
    processor.logger.info("=" * 80)

    # Build work queue
    work_queue = processor.build_work_queue()

    if not work_queue:
        processor.logger.info("No files to process. All files are completed or failed.")
        processor.print_summary(time.time())
        return 0

    processor.logger.info(f"Starting processing of {len(work_queue)} files...")

    start_time = time.time()
    processed_this_run = 0

    # Process in chunks
    for chunk_idx, file_chunk in enumerate(chunks(work_queue, args.chunk_size), 1):
        processor.logger.info(f"\nProcessing chunk {chunk_idx} ({len(file_chunk)} files)...")
        chunk_start = time.time()

        try:
            # Create dask bag and process
            bag = db.from_sequence(file_chunk)
            bag = bag.map(lambda f: process_buffer(f, processor))
            results = bag.compute(num_workers=args.n_workers)

            # Update tracking serially for all results (no race conditions)
            success_count = 0
            failed_count = 0
            skipped_count = 0

            for input_file, output_file, status, error_msg in results:
                if status == "success":
                    processor.record_success(input_file, output_file)
                    processor.update_checkpoint(input_file)
                    success_count += 1
                elif status == "failed":
                    processor.record_failure(input_file, error_msg)
                    failed_count += 1
                elif status == "skipped":
                    skipped_count += 1

            chunk_elapsed = time.time() - chunk_start
            processor.logger.info(f"Chunk {chunk_idx} complete: {success_count} success, {failed_count} failed, {skipped_count} skipped ({chunk_elapsed:.1f}s)")

            processed_this_run += success_count

        except Exception as e:
            processor.logger.error(f"Chunk {chunk_idx} failed with error: {e}\n{traceback.format_exc()}")
            processor.logger.error("Continuing with next chunk...")

        del bag

    # Final summary
    processor.logger.info(f"\nFiles processed in this run: {processed_this_run}")
    processor.print_summary(start_time)

    return 0


if __name__ == "__main__":
    if "PYART_QUIET" not in os.environ:
        os.environ["PYART_QUIET"] = "1"  # Suppress Py-ART warnings
    unravel.warmup()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
