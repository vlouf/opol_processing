"""
Distributed OPOL radar processing with resumable progress tracking.

This lean runner keeps only one parallel execution path:
- ProcessPoolExecutor backend
- Chunked execution
- JSON progress tracking (completed + failed)
- Run-level failure CSV report

Design goals:
- Keep behavior deterministic and easy to operate on HPC
- Avoid serial fallback when the process pool breaks
- Minimize code size and branching complexity
"""

import argparse
import csv
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import opol_processing
import unravel


def write_json_atomic(filepath: Path, data: Dict) -> None:
    """Atomically write JSON to avoid partial files on interruption."""
    temp_file = filepath.with_suffix(filepath.suffix + ".tmp")
    with open(temp_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    temp_file.replace(filepath)


def load_json(filepath: Path) -> Dict:
    """Load JSON dictionary or return empty dict if file is absent."""
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_date_from_filename(basename: str) -> Optional[str]:
    """Extract YYYYMMDD date from the first 8-digit group in a filename."""
    match = re.search(r"\d{8}", basename)
    return match.group(0) if match else None


def generate_output_filename(input_file: str, processed_root: str) -> str:
    """Build output path under processed_root/YYYYMMDD/*.cfradial.nc."""
    basename = os.path.basename(input_file)
    date_str = extract_date_from_filename(basename)
    if date_str is None:
        date_str = datetime.now().strftime("%Y%m%d")

    output_basename = basename.replace(".hdf", ".cfradial.nc")
    date_dir = Path(processed_root) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return str(date_dir / output_basename)


def process_buffer(
    input_file: str,
    processed_root: str,
    do_dealiasing: bool,
    debug: bool,
    profile_memory: bool,
) -> Tuple[str, str, str, Optional[str]]:
    """
    Process a single file and return (input_file, output_file, status, error_msg).

    Status values are: success, failed, skipped.
    """
    try:
        output_file = generate_output_filename(input_file, processed_root)

        result = opol_processing.process_and_save(
            input_file,
            output_filename=output_file,
            do_dealiasing=do_dealiasing,
            use_csu=True,
            debug=debug,
            profile_memory=profile_memory,
            exist_ok=True,
        )

        # process_and_save can return None for both success and skipped scans.
        if os.path.isfile(output_file):
            return (input_file, output_file, "success", None)

        if result is None:
            return (
                input_file,
                "",
                "skipped",
                "process_and_save returned None with no output file",
            )

        return (input_file, output_file, "success", None)

    except FileExistsError:
        output_file = generate_output_filename(input_file, processed_root)
        return (input_file, output_file, "success", None)
    except Exception as exc:
        error_msg = "%s: %s\n%s" % (type(exc).__name__, str(exc), traceback.format_exc())
        return (input_file, "", "failed", error_msg)
    finally:
        # Long-lived pool workers: release cyclic garbage (pyart radar objects
        # hold reference cycles) so RSS does not ratchet up between tasks.
        gc.collect()


def setup_logger(log_dir: Path) -> logging.Logger:
    """Configure file+console logging and return module logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / ("processing_%s.log" % timestamp)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_mp_context(start_method: str):
    """Return multiprocessing context for the selected start method."""
    if start_method == "auto":
        if sys.platform.startswith("linux"):
            try:
                return mp.get_context("fork"), "fork"
            except ValueError:
                pass
        return mp.get_context("spawn"), "spawn"
    return mp.get_context(start_method), start_method


def build_work_queue(
    input_dir: Path,
    completed: Dict,
    failed: Dict,
    retry_failed: bool,
    logger: logging.Logger,
) -> List[str]:
    """Build process queue with basename dedupe and resume filtering."""
    all_files = sorted(glob.glob(str(input_dir / "**/*.hdf"), recursive=True))

    unique_by_basename: Dict[str, str] = {}
    duplicate_count = 0
    for input_file in all_files:
        basename = os.path.basename(input_file)
        if basename in unique_by_basename:
            duplicate_count += 1
            continue
        unique_by_basename[basename] = input_file

    candidate_files = list(unique_by_basename.values())
    work_queue: List[str] = []

    skipped_completed = 0
    skipped_failed = 0

    for input_file in candidate_files:
        basename = os.path.basename(input_file)

        if basename in completed:
            skipped_completed += 1
            continue

        if basename in failed and not retry_failed:
            skipped_failed += 1
            continue

        work_queue.append(input_file)

    logger.info(
        "Found %s total files (%s unique basenames): %s to process, %s skipped completed, %s skipped failed, %s duplicate paths ignored",
        len(all_files),
        len(candidate_files),
        len(work_queue),
        skipped_completed,
        skipped_failed,
        duplicate_count,
    )

    if duplicate_count:
        logger.warning(
            "Duplicate basenames detected. Keeping first sorted path per basename; ignored duplicate paths=%s",
            duplicate_count,
        )

    return work_queue


def init_failure_csv(log_dir: Path) -> Path:
    """Create run-level failure CSV with header and return its path."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failure_csv = log_dir / ("failure_report_%s.csv" % run_timestamp)

    with open(failure_csv, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "timestamp",
                "chunk_idx",
                "input_file",
                "basename",
                "status",
                "error_type",
                "error_message",
                "traceback",
            ]
        )

    return failure_csv


def init_memory_profile_log(log_dir: Path) -> Path:
    """Create run-level JSONL memory profile log and return its path."""
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    profile_log = log_dir / ("memory_profile_%s.jsonl" % run_timestamp)
    with open(profile_log, "w", encoding="utf-8") as fobj:
        fobj.write("")
    return profile_log


def append_failure_csv_row(
    failure_csv: Path,
    chunk_idx: int,
    input_file: str,
    basename: str,
    status: str,
    error_type: str,
    error_message: str,
    tb: str,
) -> None:
    """Append one failure row."""
    with open(failure_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                datetime.now().isoformat(),
                chunk_idx,
                input_file,
                basename,
                status,
                error_type,
                error_message,
                tb,
            ]
        )


def run_processpool_chunk(
    executor: ProcessPoolExecutor,
    chunk_files: List[str],
    worker_func,
) -> List[Tuple[str, str, str, Optional[str]]]:
    """Run one chunk on a process pool and collect worker results."""
    results: List[Tuple[str, str, str, Optional[str]]] = []
    future_to_input = {executor.submit(worker_func, input_file): input_file for input_file in chunk_files}

    for future in as_completed(future_to_input):
        input_file = future_to_input[future]
        try:
            results.append(future.result())
        except BrokenProcessPool:
            raise
        except Exception as exc:
            err = "%s: %s" % (type(exc).__name__, exc)
            results.append((input_file, "", "failed", err))

    return results


def update_tracking_from_results(
    results: List[Tuple[str, str, str, Optional[str]]],
    completed: Dict,
    failed: Dict,
    failure_csv: Path,
    logger: logging.Logger,
    chunk_idx: int,
    chunk_elapsed: float,
    debug: bool,
) -> Tuple[int, int, int]:
    """Apply chunk results to tracking dicts and emit concise diagnostics."""
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for input_file, output_file, status, error_msg in results:
        basename = os.path.basename(input_file)

        if status == "success":
            completed[basename] = {
                "output": output_file,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            }
            failed.pop(basename, None)
            success_count += 1
            if debug:
                logger.info("SUCCESS %s -> %s", basename, output_file)
            continue

        if status == "skipped":
            completed[basename] = {
                "output": "",
                "status": "skipped",
                "reason": error_msg or "unknown",
                "timestamp": datetime.now().isoformat(),
            }
            failed.pop(basename, None)
            skipped_count += 1
            continue

        failed_error = error_msg or "Unknown error"
        failed[basename] = {
            "error": failed_error,
            "timestamp": datetime.now().isoformat(),
        }
        failed_count += 1

        head, _, tb = failed_error.partition("\n")
        error_type, _, error_message = head.partition(": ")
        if not error_type:
            error_type = "UnknownError"
            error_message = head or "Unknown error"

        append_failure_csv_row(
            failure_csv=failure_csv,
            chunk_idx=chunk_idx,
            input_file=input_file,
            basename=basename,
            status=status,
            error_type=error_type,
            error_message=error_message,
            tb=tb,
        )
        logger.error("FAILED %s: %s", basename, head)

    logger.info(
        "Chunk %s complete: %s success, %s failed, %s skipped (%.1fs)",
        chunk_idx,
        success_count,
        failed_count,
        skipped_count,
        chunk_elapsed,
    )

    return success_count, failed_count, skipped_count


def print_summary(
    logger: logging.Logger,
    input_dir: Path,
    output_dir: Path,
    completed: Dict,
    failed: Dict,
    start_time: float,
) -> None:
    """Print end-of-run summary using consolidated tracking state."""
    elapsed = time.time() - start_time

    completed_success = sum(1 for item in completed.values() if item.get("status") == "success")
    completed_skipped = sum(1 for item in completed.values() if item.get("status") == "skipped")
    failed_count = len(failed)

    logger.info("=" * 80)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 80)
    logger.info("Input: %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Total completed(success): %s", completed_success)
    logger.info("Total skipped: %s", completed_skipped)
    logger.info("Total failed: %s", failed_count)
    logger.info("Total elapsed: %.1f seconds (%.1f hours)", elapsed, elapsed / 3600)

    total_done = completed_success + completed_skipped
    if elapsed > 0 and total_done > 0:
        logger.info("Processing rate: %.2f files/second", total_done / elapsed)

    if completed_skipped > 0:
        reason_counts: Dict[str, int] = {}
        for item in completed.values():
            if item.get("status") != "skipped":
                continue
            reason = item.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_summary = ", ".join("%s=%s" % (key, value) for key, value in reason_counts.items())
        logger.info("Skip reasons: %s", reason_summary)

    logger.info("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="OPOL radar processing using a single ProcessPool backend."
    )
    parser.add_argument(
        "--input",
        dest="input_dir",
        type=str,
        required=True,
        help="Path to read-only input directory containing .hdf files",
    )
    parser.add_argument(
        "--output",
        dest="output_dir",
        type=str,
        required=True,
        help="Path to output directory for processed files and tracking",
    )
    parser.add_argument(
        "--chunk-size",
        dest="chunk_size",
        type=int,
        default=64,
        help="Number of files to process per chunk (default: 64)",
    )
    parser.add_argument(
        "--workers",
        dest="n_workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    parser.add_argument(
        "--do-dealiasing",
        dest="do_dealiasing",
        action="store_true",
        default=True,
        help="Enable velocity dealiasing (default: True)",
    )
    parser.add_argument(
        "--no-dealiasing",
        dest="do_dealiasing",
        action="store_false",
        help="Disable velocity dealiasing",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable verbose per-file success logging",
    )
    parser.add_argument(
        "--profile-memory",
        dest="profile_memory",
        action="store_true",
        help="Log per-stage worker memory diagnostics (RSS + peak RSS)",
    )
    parser.add_argument(
        "--retry-failed",
        dest="retry_failed",
        action="store_true",
        help="Retry files listed in failed.json instead of skipping them",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        dest="max_tasks_per_child",
        type=int,
        default=8,
        help=(
            "Recycle each worker process after N tasks to bound memory growth "
            "(0 disables recycling; requires Python >= 3.11 and spawn/forkserver; default: 8)"
        ),
    )
    parser.add_argument(
        "--mp-start-method",
        dest="mp_start_method",
        type=str,
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="Multiprocessing start method (default: auto)",
    )
    return parser


def main() -> int:
    """Main processing loop."""
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    processed_dir = output_dir / "processed"
    tracking_dir = output_dir / ".processing"
    logs_dir = tracking_dir / "logs"

    if not input_dir.exists():
        raise FileNotFoundError("Input directory does not exist: %s" % input_dir)
    if not os.access(input_dir, os.R_OK):
        raise PermissionError("Input directory is not readable: %s" % input_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not os.access(output_dir, os.W_OK):
        raise PermissionError("Output directory is not writable: %s" % output_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    tracking_dir.mkdir(parents=True, exist_ok=True)

    completed_file = tracking_dir / "completed.json"
    failed_file = tracking_dir / "failed.json"

    logger = setup_logger(logs_dir)

    completed = load_json(completed_file)
    failed = load_json(failed_file)

    failure_csv = init_failure_csv(logs_dir)
    memory_profile_log: Optional[Path] = None
    if args.profile_memory:
        memory_profile_log = init_memory_profile_log(logs_dir)
        os.environ["OPOL_MEMLOG_PATH"] = str(memory_profile_log)

    logger.info("=" * 80)
    logger.info("OPOL PROCESSPOOL PROCESSING")
    logger.info("=" * 80)
    logger.info("Input directory: %s", input_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Processed files: %s", processed_dir)
    logger.info("Tracking: %s", tracking_dir)
    logger.info("Failure CSV report: %s", failure_csv)
    if memory_profile_log is not None:
        logger.info("Memory profile log: %s", memory_profile_log)
    logger.info("Chunk size: %s", args.chunk_size)
    logger.info("Workers: %s", args.n_workers)
    logger.info("Dealiasing: %s", args.do_dealiasing)
    logger.info("Profile memory: %s", args.profile_memory)
    logger.info("Retry failed: %s", args.retry_failed)

    mp_context, selected_start_method = get_mp_context(args.mp_start_method)
    logger.info("Multiprocessing start method: %s", selected_start_method)

    executor_kwargs = {"max_workers": args.n_workers, "mp_context": mp_context}
    if args.max_tasks_per_child > 0:
        if sys.version_info >= (3, 11) and selected_start_method != "fork":
            executor_kwargs["max_tasks_per_child"] = args.max_tasks_per_child
            logger.info("Worker recycling: max_tasks_per_child=%s", args.max_tasks_per_child)
        else:
            logger.warning(
                "Worker recycling unavailable (needs Python >= 3.11 and spawn/forkserver start method). "
                "Worker RSS may grow across chunks and hit the job memory limit."
            )

    work_queue = build_work_queue(
        input_dir=input_dir,
        completed=completed,
        failed=failed,
        retry_failed=args.retry_failed,
        logger=logger,
    )

    if not work_queue:
        logger.info("No files to process. All files are completed or failed.")
        print_summary(logger, input_dir, output_dir, completed, failed, time.time())
        return 0

    start_time = time.time()
    processed_this_run = 0

    process_file = partial(
        process_buffer,
        processed_root=str(processed_dir),
        do_dealiasing=args.do_dealiasing,
        debug=args.debug,
        profile_memory=args.profile_memory,
    )

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        for chunk_idx, chunk_start_idx in enumerate(range(0, len(work_queue), args.chunk_size), 1):
            chunk_files = work_queue[chunk_start_idx : chunk_start_idx + args.chunk_size]
            chunk_start = time.time()
            logger.info("Processing chunk %s (%s files)", chunk_idx, len(chunk_files))

            try:
                results = run_processpool_chunk(executor, chunk_files, process_file)
            except BrokenProcessPool:
                logger.error(
                    "BrokenProcessPool in chunk %s; aborting immediately to avoid serial fallback.",
                    chunk_idx,
                )
                # Record the in-flight files so the failure CSV is not empty
                # when the pool dies (e.g. worker OOM-killed by the scheduler).
                for aborted_file in chunk_files:
                    append_failure_csv_row(
                        failure_csv=failure_csv,
                        chunk_idx=chunk_idx,
                        input_file=aborted_file,
                        basename=os.path.basename(aborted_file),
                        status="aborted",
                        error_type="BrokenProcessPool",
                        error_message="Worker pool died while this chunk was in flight (likely OOM kill).",
                        tb="",
                    )
                return 2
            except Exception as exc:
                logger.error(
                    "Chunk %s failed with infrastructure error %s: %s\n%s",
                    chunk_idx,
                    type(exc).__name__,
                    exc,
                    traceback.format_exc(),
                )
                return 2

            success_count, _, _ = update_tracking_from_results(
                results=results,
                completed=completed,
                failed=failed,
                failure_csv=failure_csv,
                logger=logger,
                chunk_idx=chunk_idx,
                chunk_elapsed=time.time() - chunk_start,
                debug=args.debug,
            )
            processed_this_run += success_count

            write_json_atomic(completed_file, completed)
            write_json_atomic(failed_file, failed)

    logger.info("Files processed successfully in this run: %s", processed_this_run)
    print_summary(logger, input_dir, output_dir, completed, failed, start_time)
    return 0


if __name__ == "__main__":
    # Keep thread caps in-script to avoid BLAS/OpenMP oversubscription per worker.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # Limit glibc malloc arenas and force freed memory back to the OS.
    # Repeated large numpy allocations otherwise fragment the heap and worker
    # RSS ratchets upward even though Python has freed the objects.
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "134217728")

    if "PYART_QUIET" not in os.environ:
        os.environ["PYART_QUIET"] = "1"

    try:
        unravel.warmup()
    except Exception:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sys.exit(main())
