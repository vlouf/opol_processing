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
import csv
import time
import argparse
import warnings
import traceback
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

from collections import Counter
from functools import partial
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

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

    def __init__(self, input_dir: str, output_dir: str,
                 do_dealiasing: bool = True, debug: bool = False,
                 retry_failed: bool = False):
        """
        Initialize voyage processor.

        Parameters
        ----------
        input_dir : str
            Path to read-only input directory containing raw .hdf files
        output_dir : str
            Path to output directory for processed files and tracking
        do_dealiasing : bool
            Enable velocity dealiasing
        debug : bool
            Print debug output
        retry_failed : bool
            If True, include previously failed files in the work queue
        """
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.processed_dir = self.output_dir / "processed"
        self.tracking_dir = self.output_dir / ".processing"

        self.do_dealiasing = do_dealiasing
        self.debug = debug
        self.retry_failed = retry_failed

        # Tracking files
        self.completed_file = self.tracking_dir / "completed.json"
        self.failed_file = self.tracking_dir / "failed.json"
        self.skipped_file = self.tracking_dir / "skipped.json"
        self.checkpoint_file = self.tracking_dir / "checkpoint.txt"
        self.log_dir = self.tracking_dir / "logs"

        # State
        self.completed = {}
        self.failed = {}
        self.skipped = {}

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
            raise PermissionError(f"Cannot create output directory {self.output_dir}: {e}") from e

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
        if not self.skipped_file.exists():
            self._write_json(self.skipped_file, {})

    def _load_progress(self):
        """Load existing progress from tracking files."""
        if self.completed_file.exists():
            with open(self.completed_file, 'r', encoding='utf-8') as f:
                self.completed = json.load(f)

        if self.failed_file.exists():
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                self.failed = json.load(f)

        if self.skipped_file.exists():
            with open(self.skipped_file, 'r', encoding='utf-8') as f:
                self.skipped = json.load(f)

    def _setup_logging(self):
        """Setup logging to file and console."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"processing_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _write_json(filepath: Path, data: dict):
        """Atomically write JSON file."""
        # Write to temp file first, then rename (atomic)
        temp_file = filepath.with_suffix(filepath.suffix + '.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
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
        # Always scan recursively so nested files are never missed.
        all_files = sorted(glob.glob(str(self.input_dir / "**/*.hdf"), recursive=True))

        # Edge case support: duplicate basenames can exist across subdirectories.
        # Keep only one path per basename (deterministically the first sorted path).
        unique_by_basename = {}
        duplicate_paths = {}
        for input_file in all_files:
            basename = os.path.basename(input_file)
            if basename not in unique_by_basename:
                unique_by_basename[basename] = input_file
                continue

            duplicate_paths.setdefault(basename, [unique_by_basename[basename]]).append(input_file)

        candidate_files = list(unique_by_basename.values())

        # Filter to only files not yet completed
        work_queue = []
        skipped_completed = 0
        skipped_failed = 0
        skipped_duplicates = len(all_files) - len(candidate_files)

        for input_file in candidate_files:
            basename = os.path.basename(input_file)

            if basename in self.completed:
                skipped_completed += 1
                continue

            if basename in self.failed and not self.retry_failed:
                # Skip recently failed (can be retried with --retry flag)
                skipped_failed += 1
                continue

            work_queue.append(input_file)

        self.logger.info(
            f"Found {len(all_files)} total files ({len(candidate_files)} unique basenames): {len(work_queue)} to process, "
            f"{skipped_completed} skipped (already completed), {skipped_failed} skipped (previously failed), "
            f"{skipped_duplicates} duplicate-path files ignored"
        )

        if duplicate_paths:
            self.logger.warning(
                f"Duplicate basenames detected across subdirectories: {len(duplicate_paths)} basenames. "
                "Only one file per basename will be processed (first path in sorted order)."
            )
            # Keep the log concise while still actionable.
            preview_items = list(duplicate_paths.items())[:10]
            for basename, paths in preview_items:
                self.logger.warning(f"  Duplicate basename {basename}: {paths}")
            if len(duplicate_paths) > len(preview_items):
                self.logger.warning(
                    f"  ... {len(duplicate_paths) - len(preview_items)} more duplicate basenames not shown"
                )

        return work_queue

    def record_success(self, input_file: str, output_file: str):
        """Record successful processing."""
        basename = os.path.basename(input_file)
        self.completed[basename] = {
            'output': output_file,
            'timestamp': datetime.now().isoformat()
        }
        # Remove from failed if it was previously failed
        self.failed.pop(basename, None)
        # Success supersedes any previous skip record.
        self.skipped.pop(basename, None)

    def record_failure(self, input_file: str, error_msg: str):
        """Record failed processing."""
        basename = os.path.basename(input_file)
        self.failed[basename] = {
            'error': error_msg,
            'timestamp': datetime.now().isoformat()
        }
        self.skipped.pop(basename, None)

    def record_skip(self, input_file: str, reason: str):
        """Record skipped processing (e.g. unsuitable scan)."""
        basename = os.path.basename(input_file)
        self.skipped[basename] = {
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }

    def flush_tracking(self):
        """Persist all tracking dictionaries once per chunk."""
        self._write_json(self.completed_file, self.completed)
        self._write_json(self.failed_file, self.failed)
        self._write_json(self.skipped_file, self.skipped)

    def update_checkpoint(self, input_file: str):
        """Update processing checkpoint."""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            f.write(os.path.basename(input_file))

    def print_summary(self, start_time: float):
        """Print processing summary."""
        elapsed = time.time() - start_time
        completed_count = len(self.completed)
        failed_count = len(self.failed)
        skipped_count = len(self.skipped)

        self.logger.info("=" * 80)
        self.logger.info("PROCESSING SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Input: {self.input_dir}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Total completed: {completed_count}")
        self.logger.info(f"Total failed: {failed_count}")
        self.logger.info(f"Total skipped: {skipped_count}")
        self.logger.info(f"Total elapsed: {elapsed:.1f} seconds ({elapsed/3600:.1f} hours)")

        if completed_count > 0:
            rate = completed_count / elapsed
            self.logger.info(f"Processing rate: {rate:.2f} files/second")

        if failed_count > 0:
            self.logger.warning("Failed files:")
            for basename, info in self.failed.items():
                self.logger.warning(f"  - {basename}: {info['error']}")

        if skipped_count > 0:
            reason_counter = Counter(info.get('reason', 'unknown') for info in self.skipped.values())
            reason_summary = ", ".join([f"{reason}={count}" for reason, count in reason_counter.items()])
            self.logger.info(f"Skip reasons: {reason_summary}")

        self.logger.info("=" * 80)


class RunResultHandler:
    """Centralizes per-chunk result handling and run artifacts updates."""

    def __init__(self, processor: VoyageProcessor, failure_csv: Path):
        self.processor = processor
        self.failure_csv = failure_csv

    def handle_results(
        self,
        chunk_idx: int,
        chunk_start: float,
        results: List[Tuple[str, str, str, Optional[str]]],
        retried_serial: bool = False,
    ) -> int:
        """Update tracking for a chunk and emit actionable diagnostics."""
        success_count = 0
        failed_count = 0
        skipped_count = 0
        skip_reasons = Counter()
        last_success_file = None

        for input_file, output_file, status, error_msg in results:
            basename = os.path.basename(input_file)
            if status == "success":
                self.processor.record_success(input_file, output_file)
                success_count += 1
                last_success_file = input_file
                self.processor.logger.info(f"SUCCESS {basename} -> {output_file}")
            elif status == "failed":
                self.processor.record_failure(input_file, error_msg or "Unknown error")
                failed_count += 1
                short_error = (error_msg or "").splitlines()[0] if error_msg else "Unknown error"
                self.processor.logger.error(f"FAILED {basename}: {short_error}")

                if error_msg:
                    head, _, tb = error_msg.partition("\n")
                    error_type, _, error_message = head.partition(": ")
                else:
                    error_type, error_message, tb = "UnknownError", "Unknown error", ""

                with open(self.failure_csv, "a", newline="", encoding="utf-8") as fcsv:
                    csv_writer = csv.writer(fcsv)
                    csv_writer.writerow([
                        datetime.now().isoformat(),
                        chunk_idx,
                        input_file,
                        basename,
                        status,
                        error_type,
                        error_message,
                        tb,
                    ])
            elif status == "skipped":
                skipped_count += 1
                reason = error_msg or "unknown"
                skip_reasons[reason] += 1
                self.processor.record_skip(input_file, reason)

        self.processor.flush_tracking()
        if last_success_file is not None:
            self.processor.update_checkpoint(last_success_file)

        chunk_elapsed = time.time() - chunk_start
        summary_prefix = "Chunk retry" if retried_serial else "Chunk"
        self.processor.logger.info(
            f"{summary_prefix} {chunk_idx} complete: {success_count} success, "
            f"{failed_count} failed, {skipped_count} skipped ({chunk_elapsed:.1f}s)"
        )
        if skip_reasons:
            reason_summary = ", ".join([f"{reason}={count}" for reason, count in skip_reasons.items()])
            self.processor.logger.info(f"Chunk {chunk_idx} skip reasons: {reason_summary}")

        return success_count


class StartupDiagnostics:
    """Encapsulates startup environment checks and process backend setup."""

    def __init__(self, logger, args):
        self.logger = logger
        self.args = args

    @staticmethod
    def effective_cpu_count() -> int:
        """Return effective CPU count (affinity-aware when available)."""
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("Cpus_allowed_list:"):
                        cpu_list = line.split(":", 1)[1].strip()
                        if not cpu_list:
                            break

                        count = 0
                        for part in cpu_list.split(","):
                            part = part.strip()
                            if not part:
                                continue
                            if "-" in part:
                                start_s, end_s = part.split("-", 1)
                                count += int(end_s) - int(start_s) + 1
                            else:
                                count += 1

                        if count > 0:
                            return count
        except Exception:
            pass
        return os.cpu_count() or 1

    @staticmethod
    def total_memory_bytes() -> Optional[int]:
        """Best-effort total memory probe from /proc/meminfo (Linux)."""
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        # MemTotal is in kB.
                        return int(parts[1]) * 1024
        except Exception:
            return None
        return None

    @staticmethod
    def fmt_gib(num_bytes: int) -> str:
        """Format bytes as GiB for logs."""
        return f"{num_bytes / (1024 ** 3):.1f} GiB"

    @staticmethod
    def get_mp_context(start_method: str):
        """Return multiprocessing context for a requested start method."""
        if start_method == "auto":
            # On Linux/HPC, fork often gives much better memory sharing than spawn.
            if sys.platform.startswith("linux"):
                try:
                    return mp.get_context("fork"), "fork"
                except ValueError:
                    pass
            return mp.get_context("spawn"), "spawn"

        return mp.get_context(start_method), start_method

    def log_runtime_limits(self) -> int:
        """Log startup diagnostics and return effective CPU count."""
        detected_cpus = os.cpu_count() or 1
        effective_cpus = self.effective_cpu_count()
        mem_bytes = self.total_memory_bytes()
        omp_threads = os.environ.get("OMP_NUM_THREADS", "unset")
        openblas_threads = os.environ.get("OPENBLAS_NUM_THREADS", "unset")
        mkl_threads = os.environ.get("MKL_NUM_THREADS", "unset")
        numexpr_threads = os.environ.get("NUMEXPR_NUM_THREADS", "unset")

        self.logger.info(
            "Startup diagnostics: cpu_count=%s, effective_cpu_affinity=%s, requested_workers=%s, chunk_size=%s",
            detected_cpus,
            effective_cpus,
            self.args.n_workers,
            self.args.chunk_size,
        )
        self.logger.info(
            "Startup diagnostics: OMP_NUM_THREADS=%s, OPENBLAS_NUM_THREADS=%s, MKL_NUM_THREADS=%s, NUMEXPR_NUM_THREADS=%s",
            omp_threads,
            openblas_threads,
            mkl_threads,
            numexpr_threads,
        )

        if mem_bytes is not None:
            mem_per_worker_est = 12 * 1024 ** 3  # Heuristic for this workload.
            mem_bound_workers = max(1, mem_bytes // mem_per_worker_est)
            suggested_workers = max(1, min(effective_cpus, mem_bound_workers, self.args.n_workers))
            self.logger.info(
                "Startup diagnostics: total_mem=%s, heuristic_mem_bound_workers(~12GiB/worker)=%s, suggested_workers=%s",
                self.fmt_gib(mem_bytes),
                mem_bound_workers,
                suggested_workers,
            )
        else:
            self.logger.info(
                "Startup diagnostics: total_mem=unknown (could not read /proc/meminfo), suggested_workers<=%s",
                effective_cpus,
            )

        if self.args.n_workers > effective_cpus:
            self.logger.warning(
                "Requested workers (%s) exceeds effective CPU affinity (%s). Some workers may remain idle.",
                self.args.n_workers,
                effective_cpus,
            )

        if self.args.chunk_size < min(self.args.n_workers, effective_cpus):
            self.logger.warning(
                "Chunk size (%s) is lower than active worker target (%s). This can underutilize workers.",
                self.args.chunk_size,
                min(self.args.n_workers, effective_cpus),
            )

        return effective_cpus


def generate_output_filename(input_file: str, processed_root: str) -> str:
    """Build output path using YYYYMMDD subdirectories under processed_root."""
    basename = os.path.basename(input_file)
    date_str = VoyageProcessor._extract_date_from_filename(basename)
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    output_basename = basename.replace('.hdf', '.cfradial.nc')
    date_dir = Path(processed_root) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return str(date_dir / output_basename)


def process_buffer(
    input_file: str,
    processed_root: str,
    do_dealiasing: bool,
    debug: bool,
) -> Tuple[str, str, str, Optional[str]]:
    """
    Process a single file with error handling.

    Wraps production_line, catches exceptions, and returns status.
    Note: Does NOT update tracking files (to avoid race conditions in parallel Dask).
    Tracking updates happen serially in main thread after all workers complete.

    Parameters
    ----------
    input_file : str
        Input .hdf file path
    processed_root : str
        Root directory for processed files
    do_dealiasing : bool
        Enable velocity dealiasing
    debug : bool
        Enable debug output

    Returns
    -------
    tuple
        (input_file, output_file, status, error_msg)
        - status: "success", "failed", "skipped"
        - error_msg: None on success, error string on failure
    """
    try:
        output_file = generate_output_filename(input_file, processed_root)

        # Process file
        result = opol_processing.process_and_save(
            input_file,
            output_filename=output_file,
            do_dealiasing=do_dealiasing,
            use_csu=True,  # Always enabled (no alternative)
            debug=debug,
            exist_ok=True
        )

        # process_and_save returns None for both success and skipped scans.
        # Decide based on output existence.
        if os.path.isfile(output_file):
            return (input_file, output_file, "success", None)

        if result is None:
            return (input_file, "", "skipped", "production_line returned None (e.g. unsuitable scan)")

        # Return success (tracking updated serially in main thread)
        return (input_file, output_file, "success", None)

    except FileExistsError:
        # Output already exists, might be from previous run
        output_file = generate_output_filename(input_file, processed_root)
        return (input_file, output_file, "success", None)

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        # Return failure (tracking updated serially in main thread)
        return (input_file, "", "failed", error_msg)


def build_parser() -> argparse.ArgumentParser:
    """Build and return CLI argument parser."""
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
    parser.add_argument(
        "--retry-failed",
        dest="retry_failed",
        action="store_true",
        help="Retry files listed in failed.json instead of skipping them"
    )
    parser.add_argument(
        "--backend",
        dest="backend",
        type=str,
        default="processpool",
        choices=["processpool", "dask"],
        help="Parallel backend to use (default: processpool)"
    )
    parser.add_argument(
        "--mp-start-method",
        dest="mp_start_method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method for processpool backend (default: auto)"
    )
    return parser


def main():
    """
    Main processing loop.

    Processes voyage files in chunks using Dask.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Initialize processor
    processor = VoyageProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        do_dealiasing=args.do_dealiasing,
        debug=args.debug,
        retry_failed=args.retry_failed,
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
    processor.logger.info(f"Backend: {args.backend}")
    processor.logger.info(f"Retry failed: {args.retry_failed}")
    processor.logger.info("=" * 80)

    diagnostics = StartupDiagnostics(processor.logger, args)
    diagnostics.log_runtime_limits()

    pool_context = None
    selected_start_method = None
    executor: Optional[ProcessPoolExecutor] = None
    broken_pool_count = 0
    if args.backend == "processpool":
        pool_context, selected_start_method = diagnostics.get_mp_context(args.mp_start_method)
        processor.logger.info(
            "Startup diagnostics: processpool start_method=%s",
            selected_start_method,
        )
        executor = ProcessPoolExecutor(max_workers=args.n_workers, mp_context=pool_context)

    # Run-level failure CSV report for postmortem diagnostics.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_csv = processor.log_dir / f"failure_report_{run_timestamp}.csv"
    with open(failure_csv, "w", newline="", encoding="utf-8") as fcsv:
        csv_writer = csv.writer(fcsv)
        csv_writer.writerow([
            "timestamp",
            "chunk_idx",
            "input_file",
            "basename",
            "status",
            "error_type",
            "error_message",
            "traceback",
        ])
    processor.logger.info(f"Failure CSV report: {failure_csv}")
    result_handler = RunResultHandler(processor, failure_csv)

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
    for chunk_idx, chunk_start_idx in enumerate(range(0, len(work_queue), args.chunk_size), 1):
        file_chunk = work_queue[chunk_start_idx : chunk_start_idx + args.chunk_size]
        processor.logger.info(f"\nProcessing chunk {chunk_idx} ({len(file_chunk)} files)...")
        chunk_start = time.time()
        process_file = partial(
            process_buffer,
            processed_root=str(processor.processed_dir),
            do_dealiasing=processor.do_dealiasing,
            debug=processor.debug,
        )

        try:
            if args.backend == "dask":
                npartitions = min(max(args.n_workers, 1), len(file_chunk))
                bag = db.from_sequence(file_chunk, npartitions=npartitions)
                bag = bag.map(process_file)
                results = bag.compute(num_workers=args.n_workers, scheduler="processes")
            else:
                results = []
                if executor is None:
                    raise RuntimeError("Process pool executor is not initialized")
                future_to_input = {
                    executor.submit(process_file, input_file): input_file for input_file in file_chunk
                }
                for future in as_completed(future_to_input):
                    input_file = future_to_input[future]
                    try:
                        results.append(future.result())
                    except BrokenProcessPool:
                        # This is an infrastructure failure, not a file-level failure.
                        # Bubble up so we can recreate the pool and retry the chunk.
                        raise
                    except Exception as exc:
                        err = f"{type(exc).__name__}: {exc}"
                        results.append((input_file, "", "failed", err))

            processed_this_run += result_handler.handle_results(chunk_idx, chunk_start, results, retried_serial=False)
            if args.backend == "processpool":
                broken_pool_count = 0

        except Exception as e:
            processor.logger.error(f"Chunk {chunk_idx} failed with error: {e}\n{traceback.format_exc()}")
            processor.logger.error(
                f"Retrying chunk {chunk_idx} serially to isolate failing files and capture detailed errors..."
            )

            if args.backend == "processpool" and isinstance(e, BrokenProcessPool):
                broken_pool_count += 1
                processor.logger.error("Process pool is broken; recreating executor for next chunk.")
                if executor is not None:
                    try:
                        executor.shutdown(wait=True, cancel_futures=True)
                    except Exception:
                        pass

                # If fork repeatedly breaks, fall back to spawn for stability.
                if selected_start_method == "fork" and broken_pool_count >= 2:
                    processor.logger.warning(
                        "Repeated BrokenProcessPool with fork (%s times). Switching process start method to spawn.",
                        broken_pool_count,
                    )
                    pool_context, selected_start_method = diagnostics.get_mp_context("spawn")

                executor = ProcessPoolExecutor(max_workers=args.n_workers, mp_context=pool_context)

            retry_results = [process_file(input_file) for input_file in file_chunk]
            processed_this_run += result_handler.handle_results(chunk_idx, chunk_start, retry_results, retried_serial=True)

        if args.backend == "dask" and 'bag' in locals():
            del bag

    # Final summary
    processor.logger.info(f"\nFiles processed in this run: {processed_this_run}")
    processor.print_summary(start_time)

    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)

    return 0


if __name__ == "__main__":
    # Avoid per-process BLAS/OpenMP thread oversubscription when using many workers.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if "PYART_QUIET" not in os.environ:
        os.environ["PYART_QUIET"] = "1"  # Suppress Py-ART warnings
    try:
        unravel.warmup()
    except Exception as e:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
