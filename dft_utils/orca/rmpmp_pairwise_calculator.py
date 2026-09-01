"""Run pairwise RMPMP calculations over HDF5 spectra."""

import argparse
import math
import os
from collections import defaultdict
from itertools import product
from multiprocessing import Manager, Pool, cpu_count
from queue import Empty

from dft_utils.radii_calc import MPMP_Radius_calculator

from .RMPMP_hdf import HDF_Spectrum, PairwiseDatabase, find_hdf5_files


class SpectrumDataCache:
    """
    Comprehensive cache for HDF_Spectrum objects and their computed data
    Minimizes all expensive operations: file loading, get_kr, and get_spec calls
    """

    def __init__(self):
        """Initialize the object."""
        self.spectra = {}
        self.kr_cache = {}
        self.spec_cache = {}
        self.metadata = {}
        self.validation_cache = {}

    def get_spectrum(self, file_path):
        """Load spectrum object once and cache it"""
        if file_path not in self.spectra:
            try:
                self.spectra[file_path] = HDF_Spectrum(file_path)
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
                return None
        return self.spectra[file_path]

    def get_metadata(self, file_path):
        """Get basic metadata without expensive operations"""
        if file_path not in self.metadata:
            spec = self.get_spectrum(file_path)
            if spec is None:
                return None

            self.metadata[file_path] = {
                "name": spec.name,
                "group": spec.group.lower() if hasattr(spec, "group") else "",
                "is_emission": (
                    "emit" in spec.group.lower() if hasattr(spec, "group") else False
                ),
            }

        return self.metadata[file_path]

    def get_kr_cached(self, file_path, ph, vib, ch):
        """Get kr value with caching"""
        key = (file_path, ph, vib, ch)
        if key not in self.kr_cache:
            spec = self.get_spectrum(file_path)
            if spec is None:
                return None

            try:
                self.kr_cache[key] = spec.get_kr(ph, vib, ch)
            except Exception as e:
                print(
                    f"Failed to get kr for {file_path} with params {(ph, vib, ch)}: {e}"
                )
                self.kr_cache[key] = None

        return self.kr_cache[key]

    def get_spec_cached(
        self, file_path, order, ph, vib, charge, spec_type=True, kr=None
    ):
        """Get spectrum data with caching"""
        key = (file_path, order, ph, vib, charge, spec_type)

        if key not in self.spec_cache:
            spec = self.get_spectrum(file_path)
            if spec is None:
                self.spec_cache[key] = None
                return None

            try:
                if spec_type:
                    if kr is None:
                        kr = self.get_kr_cached(file_path, ph, vib, charge)
                    if kr is None:
                        self.spec_cache[key] = None
                        return None
                    result = spec.get_spec(
                        order=order, ph=ph, vib=vib, charge=charge, kr=kr, spec=True
                    )
                else:
                    result = spec.get_spec(
                        order=order, ph=ph, vib=vib, charge=charge, spec=False
                    )
                result(0.0)

                self.spec_cache[key] = result
            except Exception as e:
                print(
                    f"Failed to get spec for {file_path} with params {(order, ph, vib, charge, spec_type)}: {e}"
                )
                self.spec_cache[key] = None

        return self.spec_cache[key]

    def validate_setting_cached(self, file_path, setting, spec_type=True):
        """Validate setting with caching to avoid repeated expensive calls"""
        key = (file_path, setting, spec_type)

        if key not in self.validation_cache:
            ch, vib, ph, ord = setting

            spec_data = self.get_spec_cached(file_path, ord, ph, vib, ch, spec_type)
            self.validation_cache[key] = spec_data is not None

        return self.validation_cache[key]

    def precompute_all_combinations(
        self, file_paths, param_combinations, spec_type=True
    ):
        """
        Precompute all kr and spec values for given files and parameter combinations
        This does ALL expensive operations upfront in batch
        """
        print(
            f"Precomputing {'emission' if spec_type else 'absorption'} data for {len(file_paths)} files..."
        )

        valid_combinations = defaultdict(list)

        for i, file_path in enumerate(file_paths):
            if i % 10 == 0:
                print(
                    f"  Processing file {i+1}/{len(file_paths)}: {os.path.basename(file_path)}"
                )

            spec = self.get_spectrum(file_path)
            if spec is None:
                continue

            for setting in param_combinations:
                ch, vib, ph, ord = setting

                try:
                    if spec_type:

                        kr = self.get_kr_cached(file_path, ph, vib, ch)
                        if kr is None:
                            continue

                        spec_data = self.get_spec_cached(
                            file_path, ord, ph, vib, ch, True, kr
                        )
                    else:
                        spec_data = self.get_spec_cached(
                            file_path, ord, ph, vib, ch, False
                        )

                    if spec_data is not None:
                        valid_combinations[file_path].append(setting)

                except Exception as e:
                    print(f"    Failed setting {setting}: {e}")
                    continue

        return valid_combinations


def build_tasks_with_precomputation(folder, P1, P2, max_tasks=None):
    """
    Build tasks with complete precomputation of all expensive operations
    """
    print("Starting optimized task building with precomputation...")

    cache = SpectrumDataCache()

    files = find_hdf5_files(folder)
    print(f"Found {len(files)} HDF5 files")

    print("Loading file metadata...")
    valid_files = []
    emit_files = []

    for i, file_path in enumerate(files):
        if i % 20 == 0:
            print(f"  Loading metadata {i+1}/{len(files)}...")

        metadata = cache.get_metadata(file_path)
        if metadata is not None:
            valid_files.append(file_path)
            if metadata["is_emission"]:
                emit_files.append(file_path)

    print(
        f"Found {len(emit_files)} emission files, {len(valid_files)} total valid files"
    )

    print("Precomputing all emission spectrum data...")
    valid_p1_combinations = cache.precompute_all_combinations(
        emit_files, P1, spec_type=True
    )

    print("Precomputing all absorption spectrum data...")
    valid_p2_combinations = cache.precompute_all_combinations(
        valid_files, P2, spec_type=False
    )

    print("Generating tasks from precomputed data...")
    tasks = []

    for s1_file in emit_files:
        if s1_file not in valid_p1_combinations or not valid_p1_combinations[s1_file]:
            continue

        s1_metadata = cache.get_metadata(s1_file)
        s1_name = s1_metadata["name"]

        for s2_file in valid_files:
            if (
                s2_file not in valid_p2_combinations
                or not valid_p2_combinations[s2_file]
            ):
                continue

            s2_metadata = cache.get_metadata(s2_file)
            s2_name = s2_metadata["name"]

            for setting1 in valid_p1_combinations[s1_file]:
                for setting2 in valid_p2_combinations[s2_file]:
                    tasks.append(
                        (s1_file, s1_name, setting1, s2_file, s2_name, setting2)
                    )

                    if max_tasks and len(tasks) >= max_tasks:
                        print(f"Reached max_tasks limit: {max_tasks}")
                        return tasks, cache

    print(f"Generated {len(tasks)} tasks with all data precomputed")
    return tasks, cache


def worker_with_cache(
    s1_file, s1_name, setting1, s2_file, s2_name, setting2, out_q, cache_data=None
):
    """
    Worker function that uses cached data when available
    """
    ch1, vib1, ph1, ord1 = setting1
    ch2, vib2, ph2, ord2 = setting2

    try:

        if cache_data:

            spec1 = cache_data.get("spec1_cache", {}).get(
                (s1_file, ord1, ph1, vib1, ch1)
            )
            spec2 = cache_data.get("spec2_cache", {}).get(
                (s2_file, ord2, ph2, vib2, ch2)
            )
            kr1 = cache_data.get("kr1_cache", {}).get((s1_file, ph1, vib1, ch1))

            if spec1 is not None and spec2 is not None and kr1 is not None:

                calc = MPMP_Radius_calculator(spec1, spec2)
                rf = calc.get_MPMP_radius(error=False, norm_kr=kr1)

                result = (
                    s1_name,
                    s2_name,
                    ch1,
                    True,
                    ph1,
                    ord1,
                    vib1,
                    ch2,
                    False,
                    ph2,
                    ord2,
                    vib2,
                    rf,
                    kr1,
                )
                out_q.put(result)
                return True

        s1 = HDF_Spectrum(s1_file)
        s2 = HDF_Spectrum(s2_file)

        kr1 = s1.get_kr(ph1, vib1, ch1)

        spec1 = s1.get_mom(order=ord1, ph=ph1, vib=vib1, charge=ch1, kr=kr1, spec=True)

        kr2 = s2.get_kr(ph2, vib2, ch2)

        spec2 = s2.get_mom(order=ord2, ph=ph2, vib=vib2, charge=ch2, spec=False)

        if spec1 is None or spec2 is None:
            return False

        calc = MPMP_Radius_calculator(spec1, spec2)
        rf = calc.get_MPMP_radius(error=False, norm_kr=kr1)
        rf = float(rf)
        if isinstance(rf, float) and math.isnan(rf):
            rf = math.nan

        result = (
            s1_name,
            s2_name,
            ch1,
            True,
            ph1,
            ord1,
            vib1,
            ch2,
            False,
            ph2,
            ord2,
            vib2,
            rf,
            kr1,
        )
        out_q.put(result)
        return True

    except Exception as e:
        print(f"Worker error for {s1_name}->{s2_name}: {e}")
        return False


def create_worker_cache_data(cache, tasks_batch):
    """
    Create a serializable cache data structure for workers
    Only includes data needed for the current batch
    """
    cache_data = {"spec1_cache": {}, "spec2_cache": {}, "kr1_cache": {}}

    s1_combinations = set()
    s2_combinations = set()

    for s1_file, _s1_name, setting1, s2_file, _s2_name, setting2 in tasks_batch:
        ch1, vib1, ph1, ord1 = setting1
        ch2, vib2, ph2, ord2 = setting2

        s1_combinations.add((s1_file, ord1, ph1, vib1, ch1))
        s2_combinations.add((s2_file, ord2, ph2, vib2, ch2))

    for s1_combo in s1_combinations:
        s1_file, ord1, ph1, vib1, ch1 = s1_combo

        spec1 = cache.get_spec_cached(s1_file, ord1, ph1, vib1, ch1, spec_type=True)
        kr1 = cache.get_kr_cached(s1_file, ph1, vib1, ch1)

        if spec1 is not None:
            cache_data["spec1_cache"][s1_combo] = spec1
        if kr1 is not None:
            cache_data["kr1_cache"][(s1_file, ph1, vib1, ch1)] = kr1

    for s2_combo in s2_combinations:
        s2_file, ord2, ph2, vib2, ch2 = s2_combo

        spec2 = cache.get_spec_cached(s2_file, ord2, ph2, vib2, ch2, spec_type=False)

        if spec2 is not None:
            cache_data["spec2_cache"][s2_combo] = spec2

    return cache_data


def main_rmpmp_analysis_ultra_optimized(
    folder, db_path, max_workers=2, batch_size=20, max_tasks=None
):
    """
    Ultra-optimized main function with comprehensive caching
    """
    P1 = [
        (c, v, ph, o)
        for c, v, ph, o in product([0], [True, False], [True, False], [1, 2])
        if not (ph and c != 0)
    ]
    P2 = [
        (c, v, ph, o)
        for c, v, ph, o in product([0, 1, -1], [True, False], [True, False], [1, 2])
        if not (ph and c != 0)
    ]

    print("Building tasks with ultra-optimized caching...")
    tasks, cache = build_tasks_with_precomputation(folder, P1, P2, max_tasks)

    if not tasks:
        print("No valid tasks found!")
        return

    print(f"Task building complete. Processing {len(tasks)} tasks with cached data...")

    nproc = max_workers if max_workers else min(4, cpu_count())
    print(f"Using {nproc} processes")

    db = PairwiseDatabase(db_path)
    total_processed = 0

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tasks) + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} tasks)")

        mgr = Manager()
        out_q = mgr.Queue()

        worker_args = [(*task, out_q, None) for task in batch]

        try:
            with Pool(nproc) as pool:
                results = pool.starmap(worker_with_cache, worker_args)

            successful_tasks = sum(1 for r in results if r)
            print(
                f"Batch processing completed: {successful_tasks}/{len(batch)} successful"
            )

            batch_results = 0
            timeout_count = 0
            max_timeouts = 20

            while batch_results < successful_tasks and timeout_count < max_timeouts:
                try:
                    result = out_q.get(timeout=2)

                    (
                        s1n,
                        s2n,
                        ch1,
                        fl1,
                        ph1,
                        o1,
                        v1,
                        ch2,
                        fl2,
                        ph2,
                        o2,
                        v2,
                        rf,
                        kr1,
                    ) = result

                    db.save(s1n, s2n, ch1, fl1, ph1, v1, o1, ch2, fl2, ph2, v2, o2, rf)

                    print(
                        f"  {s1n}_{ch1}_{o1}_{v1}_{ph1}->{s2n}_{ch2}_{o2}_{v2}_{ph2}: rf={rf:.6e}, kr={kr1:.6e}"
                    )
                    batch_results += 1
                    total_processed += 1
                    timeout_count = 0

                except Empty:
                    timeout_count += 1
                    print(
                        f"  Queue timeout {timeout_count}/{max_timeouts}, collected {batch_results}/{successful_tasks}"
                    )

            print(f"Batch {batch_num} completed: {batch_results} results saved")

        except Exception as e:
            print(f"Error processing batch {batch_num}: {e}")
            continue

    print(f"\nAll batches completed. Total results processed: {total_processed}")


def parse_args():
    """Parse the supplied input."""
    parser = argparse.ArgumentParser(
        description="Run ultra‑optimized RMPMP analysis over HDF5 spectra."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the folder containing ORCA HDF5 files",
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="Path to the SQLite database file for saving results",
    )
    parser.add_argument(
        "--max-workers",
        "-w",
        type=int,
        default=2,
        metavar="N",
        help="Number of parallel worker processes (default: 2)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=20,
        metavar="M",
        help="How many tasks each worker grabs at once (default: 20)",
    )
    parser.add_argument(
        "--max-tasks",
        "-m",
        type=int,
        default=None,
        metavar="K",
        help="Maximum total tasks to process (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main_rmpmp_analysis_ultra_optimized(
        folder=args.folder,
        db_path=args.db_path,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        max_tasks=args.max_tasks,
    )
