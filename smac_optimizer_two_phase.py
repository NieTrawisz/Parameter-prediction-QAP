import traceback
from ctypes import *
import os
import re
import time
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from multiprocessing import Manager

import numpy as np
from ConfigSpace import (
    ConfigurationSpace,
    Categorical,
    Integer,
    Float,
)
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign
from smac.runhistory import TrialInfo, TrialValue

# Configuration constants
generations_per_dataset = 50
max_islands = 10

# Two-phase optimization settings
n_lhs_configs = 750      # Phase 1: LHS configurations
n_lhs_reps = 2            # Phase 1: Repetitions per LHS config
n_bo_trials = 1000        # Phase 2: SMAC BO trials
n_total = n_lhs_configs * n_lhs_reps + n_bo_trials  # 5000 total

# GPU parallelism thresholds
SMALL_THRESHOLD = 70   # < 50x50: 2 per GPU, else 1 per GPU

# Selection and succession types
succession_types = ["NoElitism", "Elitism", "SteadyState"]
selection_types = ["Roulette", "LinearRank", "NonlinearRank", "Tournament", "Threshold"]
script_dir = os.path.dirname(os.path.abspath(__file__))
executable = os.path.join(script_dir, 'gaqap_simplified_console')
env = os.environ.copy()

def get_gpu_count():
    """Get the number of available GPUs."""
    qap_dll = cdll.LoadLibrary(os.getcwd() + "/gaqap_simplified.so")
    qap_dll.getDeviceCount.restype = c_int
    return qap_dll.getDeviceCount()


def get_qap_dimension(dataset: str, data_dir: str) -> int:
    """
    Get the dimension of a QAP instance by reading the .dat file.
    QAP format typically has dimension as the first number in the file.
    """
    dat_file = Path(data_dir) / f"{dataset}.dat"
    
    try:
        with open(dat_file, 'r') as f:
            # Read first non-empty line and extract first number
            for line in f:
                line = line.strip()
                if line:
                    # Extract first integer from the line
                    match = re.search(r'\d+', line)
                    if match:
                        return int(match.group())
    except Exception as e:
        print(f"Warning: Could not read dimension for {dataset}: {e}")
    
    # Default to large if we can't determine
    return 999


def get_slots_for_dimension(dimension: int) -> int:
    """Determine how many optimization slots a dataset needs based on its dimension."""
    if dimension < SMALL_THRESHOLD:
        return 1  # Small: 2 per GPU, so needs 1 slot out of 2
    else:
        return 2  # Large: 1 per GPU, needs all 2 slots


def sort_datasets_by_priority(datasets: list, data_dir: str) -> list:
    """
    Sort datasets with priority:
    1. Datasets WITHOUT n(\d+)_ pattern first (legacy/priority datasets)
    2. Then by dimension (smaller first for faster initial results)
    3. Then alphabetically
    """
    def get_priority(dataset):
        # Check if it has n(\d+)_ pattern
        # has_n_pattern = bool(re.search(r"n(\d+)_", dataset))
        has_n_pattern = bool(re.search(r"drezner", dataset)) or bool(re.search(r"nugent", dataset))
        
        # Get dimension
        dimension = get_qap_dimension(dataset, data_dir)
        
        # Priority: (has_pattern, dimension, name)
        # False < True, so non-pattern datasets come first
        return (has_n_pattern, dimension, dataset)
    
    return sorted(datasets, key=get_priority)


class GPUWorkQueue:
    """
    Multiprocessing-safe work queue that manages dataset assignments to GPUs.
    Uses Manager for shared state across processes.
    """
    
    def __init__(self, datasets: list, data_dir: str, manager: mp.Manager):
        # Precompute dimensions and slots needed
        dimensions = {}
        slots = {}
        for ds in datasets:
            dim = get_qap_dimension(ds, data_dir)
            dimensions[ds] = dim
            slots[ds] = get_slots_for_dimension(dim)
        
        # Shared state via Manager
        self.datasets = manager.list(datasets)
        self.dimensions = manager.dict(dimensions)
        self.slots = manager.dict(slots)
        self.completed = manager.list()
        self.in_progress = manager.list()
        self.lock = manager.Lock()
    
    def get_next_dataset(self, available_slots: int) -> tuple:
        """
        Get the next dataset that fits in available slots.
        Returns (dataset, dimension, slots_needed) or (None, 0, 0) if no suitable dataset.
        """
        with self.lock:
            for ds in self.datasets:
                if ds in self.completed or ds in self.in_progress:
                    continue
                
                slots_needed = self.slots[ds]
                dimension = self.dimensions[ds]
                
                if slots_needed <= available_slots:
                    self.in_progress.append(ds)
                    return ds, dimension, slots_needed
            
            return None, 0, 0
    
    def mark_completed(self, dataset: str):
        """Mark a dataset as completed."""
        with self.lock:
            if dataset in self.in_progress:
                self.in_progress.remove(dataset)
            if dataset not in self.completed:
                self.completed.append(dataset)
    
    def mark_failed(self, dataset: str):
        """Mark a dataset as failed (remove from in_progress)."""
        with self.lock:
            if dataset in self.in_progress:
                self.in_progress.remove(dataset)
    
    def is_done(self) -> bool:
        """Check if all datasets are completed."""
        with self.lock:
            return len(self.completed) >= len(self.datasets)
    
    def get_progress(self) -> tuple:
        """Get (completed, in_progress, total) counts."""
        with self.lock:
            return len(self.completed), len(self.in_progress), len(self.datasets)


def create_config_space() -> ConfigurationSpace:
    """Create the hyperparameter configuration space for SMAC."""
    cs = ConfigurationSpace(seed=42)

    # Core hyperparameters
    succession = Categorical("succession", succession_types)
    reproduction = Categorical("reproduction", selection_types)
    islands_num = Integer("islands_num", (2, max_islands))

    # Mutation and crossover probabilities
    prob_nothing = Float("probNothing", (0.0, 1.0))
    prob_mutate_swap = Float("probMutateSwap", (0.0, 1.0))
    prob_mutate_shift = Float("probMutateShift", (0.0, 1.0))
    prob_mutate_inverse = Float("probMutateInverse", (0.0, 1.0))
    prob_pmx_crossover = Float("probPMXcrossover", (0.0, 1.0))
    prob_ox_crossover = Float("probOXcrossover", (0.0, 1.0))
    prob_cx_crossover = Float("probCXcrossover", (0.0, 1.0))

    # Migration parameters
    migration_batch = Float("migration_batch", (0.02, 0.4))
    migration_freq = Integer("migration_freq", (20, 500))
    migration = Categorical("migration", selection_types)
    release = Categorical("release", selection_types)

    # Add all hyperparameters
    cs.add([
        succession, reproduction, islands_num,
        prob_nothing, prob_mutate_swap, prob_mutate_shift, prob_mutate_inverse,
        prob_pmx_crossover, prob_ox_crossover, prob_cx_crossover,
        migration_batch, migration_freq, migration, release
    ])

    return cs


def sample_lhs_with_categoricals(cs: ConfigurationSpace, n_samples: int, seed: int = 42) -> list:
    """Sample configurations using LHS for numerical and uniform for categorical parameters."""
    cs.seed(seed)
    configs = cs.sample_configuration(n_samples)
    if n_samples == 1:
        configs = [configs]
    return configs


def create_objective_function(dataset: str, device: int, generations: int):
    """Create the objective function for SMAC optimization."""
    # Capture these at definition time, not runtime
    
    def objective(config, seed: int = 0) -> float:
        succession_idx = succession_types.index(config["succession"])
        reproduction_idx = selection_types.index(config["reproduction"])

        islands_num = config["islands_num"]
        prob_nothing = config["probNothing"]
        prob_mutate_swap = config["probMutateSwap"]
        prob_mutate_shift = config["probMutateShift"]
        prob_mutate_inverse = config["probMutateInverse"]
        prob_pmx_crossover = config["probPMXcrossover"]
        prob_ox_crossover = config["probOXcrossover"]
        prob_cx_crossover = config["probCXcrossover"]

        migration_batch = config["migration_batch"]
        migration_freq = config["migration_freq"]
        migration_idx = selection_types.index(config["migration"])
        release_idx = selection_types.index(config["release"])
        
        # cmd = [
        #     executable,
        #     dataset,           # <dataname>
        #     str(device),             # <device>
        #     str(generations),# <maxGenerations>
        #     str(seed),          # <seed>
        #     str(succession_idx),          # <succesion_idx>
        #     str(reproduction_idx),        # <reproduction_idx>
        #     str(1),      # <min_prob>
        #     str(50),      # <max_prob>
        #     str(0.5),     # <threshold>
        #     str(prob_nothing),            # <probNothing>
        #     str(prob_mutate_swap),        # <probMutateSwap>
        #     str(prob_mutate_shift),       # <probMutateShift>
        #     str(prob_mutate_inverse),     # <probMutateInverse>
        #     str(prob_pmx_crossover),      # <probPMX>
        #     str(prob_ox_crossover),       # <probOX>
        #     str(prob_cx_crossover),       # <probCX>
        #     str(islands_num),             # <islands_num>
        #     str(migration_freq),          # <migration_freq>
        #     str(migration_batch),         # <migration_batch>
        #     str(migration_idx),           # [migration_idx] - optional but provided
        #     str(release_idx)              # [release_idx] - optional but provided
        # ]
        
        # max_retries = 5
    
        # for attempt in range(1, max_retries + 1):
        #     try:
        #         result = subprocess.run(
        #             cmd,
        #             capture_output=True,
        #             text=True,
        #             cwd=script_dir,
        #             env=env
        #         )
                
        #         if result.returncode != 0:
        #             print(f"Attempt {attempt}/{max_retries}: Program exited with code {result.returncode}", file=sys.stderr)
        #             print(f"stderr: {result.stderr}", file=sys.stderr)
        #             if attempt == max_retries:
        #                 return float("inf")
        #             continue
                
        #         output = result.stdout
        #         match = re.search(r'Found value:(\d+)', output)
                
        #         if match:
        #             value = int(match.group(1))
        #             return value
        #         else:
        #             print(f"Attempt {attempt}/{max_retries}: 'Found value:<int>' pattern not found", file=sys.stderr)
        #             print(f"Full output:\n{output}", file=sys.stderr)
        #             if attempt == max_retries:
        #                 return float("inf")
        #             continue
                    
        #     except subprocess.TimeoutExpired:
        #         print(f"Attempt {attempt}/{max_retries}: Program execution timed out", file=sys.stderr)
        #     except FileNotFoundError:
        #         print(f"Error: Could not find executable at {cmd[0]}", file=sys.stderr)
        #         return float("inf")  # No retry - file won't appear
        #     except Exception as e:
        #         print(f"Attempt {attempt}/{max_retries}: Error running program: {e}", file=sys.stderr)
            
        #     if attempt == max_retries:
        #         return float("inf")
        
        # return float("inf")
        
        # Load DLL for this thread
        qap_dll = cdll.LoadLibrary(os.getcwd() + "/gaqap_simplified.so")
        qap_dll.run.restype = c_double
        
        name = c_char_p(dataset.encode("utf-8"))

        result = qap_dll.run(
            name,
            c_int(device),
            c_int(generations),
            c_int(seed),
            c_int(succession_idx),
            c_int(reproduction_idx),
            c_float(1.0),
            c_float(50.0),
            c_float(0.5),
            (c_float * max_islands)(*([prob_nothing] * max_islands)),
            (c_float * max_islands)(*([prob_mutate_swap] * max_islands)),
            (c_float * max_islands)(*([prob_mutate_shift] * max_islands)),
            (c_float * max_islands)(*([prob_mutate_inverse] * max_islands)),
            (c_float * max_islands)(*([prob_pmx_crossover] * max_islands)),
            (c_float * max_islands)(*([prob_ox_crossover] * max_islands)),
            (c_float * max_islands)(*([prob_cx_crossover] * max_islands)),
            c_int(islands_num),
            c_int(migration_freq),
            c_float(migration_batch),
            c_int(migration_idx),
            c_int(release_idx),
        )
        
        return float(result)

    return objective


def save_phase1_checkpoint(results: list, output_dir: Path, dataset: str):
    """Save Phase 1 results as checkpoint for recovery. Uses atomic write to prevent corruption."""
    checkpoint_file = output_dir / dataset / "phase1_checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first, then rename (atomic operation)
    temp_file = checkpoint_file.with_suffix('.json.tmp')
    
    serializable = [
        {"config": dict(config), "seed": seed, "cost": cost}
        for config, seed, cost in results
    ]
    
    try:
        with open(temp_file, "w") as f:
            json.dump(serializable, f, indent=2)
        
        # Atomic rename
        temp_file.replace(checkpoint_file)
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}")
        if temp_file.exists():
            temp_file.unlink()


def load_phase1_checkpoint(output_dir: Path, dataset: str, cs: ConfigurationSpace):
    """Load Phase 1 checkpoint if it exists."""
    checkpoint_file = output_dir / dataset / "phase1_checkpoint.json"
    
    if not checkpoint_file.exists():
        return None
    
    try:
        with open(checkpoint_file, "r") as f:
            content = f.read()
            if not content.strip():
                print(f"  Warning: Empty checkpoint file, starting fresh")
                return None
            data = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Corrupted checkpoint file ({e}), starting fresh")
        return None
    
    results = []
    for item in data:
        try:
            config = cs.sample_configuration()
            for key, val in item["config"].items():
                config[key] = val
            results.append((config, item["seed"], item["cost"]))
        except Exception as e:
            print(f"  Warning: Could not restore config from checkpoint: {e}")
            continue
    
    if not results:
        return None
    
    return results

def load_existing_trials(output_dir: Path, dataset: str, cs: ConfigurationSpace):
    """Load all existing trials from SMAC's runhistory file."""
    runhistory_file = output_dir / dataset / "runhistory.json"
    
    if not runhistory_file.exists():
        return []
    
    try:
        with open(runhistory_file, "r") as f:
            content = f.read()
            if not content.strip():
                return []
            data = json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not read runhistory: {e}")
        return []
    
    trials = []
    configs_data = data.get("configs", {})
    
    for trial_entry in data.get("data", []):
        try:
            config_id = str(trial_entry[0][0])  # config_id
            seed = trial_entry[0][2]             # seed
            cost = trial_entry[1][0]             # cost
            
            # Reconstruct config
            config_dict = configs_data.get(config_id, {})
            config = cs.sample_configuration()
            for key, val in config_dict.items():
                config[key] = val
            
            trials.append((config, seed, cost))
        except Exception as e:
            continue
    
    return trials

def is_dataset_complete_fast(output_dir: Path, dataset: str, seed: int = 42) -> bool:
    """Fast check if dataset is complete - just count entries in runhistory."""
    runhistory_file = output_dir / dataset / dataset / str(seed) / "runhistory.json"
    
    if not runhistory_file.exists():
        return False
    
    try:
        with open(runhistory_file, "r") as f:
            content = f.read()
            if not content.strip():
                return False
            data = json.loads(content)
            return len(data.get("data", [])) >= n_total
    except (json.JSONDecodeError, Exception):
        return False

def get_incomplete_datasets(datasets: list, output_dir: str = "./smac_output") -> list:
    """Fast filter to get only incomplete datasets."""
    output_path = Path(output_dir)
    incomplete = []
    skipped = 0
    
    for ds in datasets:
        if is_dataset_complete_fast(output_path, ds):
            skipped += 1
        else:
            incomplete.append(ds)
    
    if skipped > 0:
        print(f"Skipped {skipped} already completed datasets")
    
    return incomplete

def run_single_optimization(dataset: str, device: int, dimension: int, output_dir: str = "./smac_output"):
    """
    Run two-phase optimization for a single dataset.
    Returns (dataset, best_config, best_value) or (dataset, None, None) on failure.
    """
    print(f"[GPU {device}] Starting: {dataset} (dim={dimension})")
    
    output_path = Path(output_dir)
    cs = create_config_space()
    objective_fn = create_objective_function(dataset, device, generations_per_dataset)
    
    # Create scenario
    scenario = Scenario(
        configspace=cs,
        name=dataset,
        output_directory=output_path / dataset,
        deterministic=False,
        n_trials=n_total,
        seed=42,
    )
    
    existing_trials = load_existing_trials(output_path, dataset, cs)
    n_existing_from_file = len(existing_trials)
    
    # If we have existing trials but need to continue, use overwrite=True 
    # and manually reload all trials
    # if n_existing_from_file > 0 and n_existing_from_file < n_total:
    #     print(f"[GPU {device}] {dataset} Found {n_existing_from_file} existing trials, reloading...")
        
    #     smac = HyperparameterOptimizationFacade(
    #         scenario=scenario,
    #         target_function=objective_fn,
    #         initial_design=DefaultInitialDesign(scenario, n_configs=0),
    #         overwrite=True,  # Start fresh SMAC state
    #     )
        
    #     # Reload all existing trials
    #     for config, seed, cost in existing_trials:
    #         trial_info = TrialInfo(config=config, seed=seed)
    #         trial_value = TrialValue(cost=cost)
    #         smac.tell(trial_info, trial_value)
        
    #     print(f"[GPU {device}] {dataset} Reloaded {len(existing_trials)} trials")
    #     n_existing = len(existing_trials)
    # else:
    # Normal initialization
    try:
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=objective_fn,
            initial_design=DefaultInitialDesign(scenario, n_configs=0),
            overwrite=False,
        )
        n_existing = len(smac.runhistory)
    except Exception as e:
        print(f"[GPU {device}] {dataset} SMAC init failed ({e}), starting fresh")
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=objective_fn,
            initial_design=DefaultInitialDesign(scenario, n_configs=0),
            overwrite=True,
        )
        n_existing = 0
    
    if n_existing >= n_total:
        print(f"[GPU {device}] {dataset} already complete")
        incumbent = smac.intensifier.get_incumbent()
        return dataset, dict(incumbent), smac.runhistory.get_cost(incumbent)
    
    # =========================================================================
    # PHASE 1: Latin Hypercube Sampling
    # =========================================================================
    n_lhs_total = n_lhs_configs * n_lhs_reps
    
    if n_existing < n_lhs_total:
        print(f"[GPU {device}] {dataset} Phase 1: LHS ({n_existing}/{n_lhs_total} done)")
        
        phase1_results = load_phase1_checkpoint(output_path, dataset, cs)
        
        if phase1_results is None:
            phase1_results = []
            lhs_configs = sample_lhs_with_categoricals(cs, n_lhs_configs, seed=42)
            
            for i, config in enumerate(lhs_configs):
                for rep in range(n_lhs_reps):
                    seed = rep
                    try:
                        cost = objective_fn(config, seed=seed)
                        phase1_results.append((config, seed, cost))
                        
                        trial_info = TrialInfo(config=config, seed=seed)
                        trial_value = TrialValue(cost=cost)
                        smac.tell(trial_info, trial_value)
                        
                    except Exception as e:
                        print(f"[GPU {device}] {dataset} Error at config {i}, rep {rep}: {e}")
                        continue
                
                if (i + 1) % 100 == 0:
                    print(f"[GPU {device}] {dataset} LHS: {i + 1}/{n_lhs_configs}")
                    save_phase1_checkpoint(phase1_results, output_path, dataset)
            
            save_phase1_checkpoint(phase1_results, output_path, dataset)
        else:
            print(f"[GPU {device}] {dataset} Loading {len(phase1_results)} checkpoint results")
            for config, seed, cost in phase1_results:
                trial_info = TrialInfo(config=config, seed=seed)
                trial_value = TrialValue(cost=cost)
                smac.tell(trial_info, trial_value)
        
        best_cost = min(c for _, _, c in phase1_results)
        print(f"[GPU {device}] {dataset} Phase 1 done, best={best_cost}")
    
    # =========================================================================
    # PHASE 2: SMAC Bayesian Optimization
    # =========================================================================
    print(f"[GPU {device}] {dataset} Phase 2: Bayesian Optimization")
    
    try:
        incumbent = smac.optimize()
        best_value = smac.runhistory.get_cost(incumbent)
        print(f"[GPU {device}] {dataset} COMPLETE, best={best_value}")
        # Return dict instead of Configuration for serialization
        return dataset, dict(incumbent), best_value
        
    except KeyboardInterrupt:
        print(f"[GPU {device}] {dataset} interrupted")
        incumbent = smac.intensifier.get_incumbent()
        if incumbent is not None:
            return dataset, dict(incumbent), smac.runhistory.get_cost(incumbent)
        return dataset, None, None
        
    except Exception as e:
        print(f"[GPU {device}] {dataset} error: {e}")
        traceback.print_exc()
        return dataset, None, None


def gpu_worker_process(device: int, work_queue, output_dir: str, results_list):
    """
    Worker process for a single GPU.
    Pulls datasets from the work queue and runs optimizations sequentially.
    Each GPU gets its own process - no GIL contention.
    """
    slots_per_gpu = 2
    
    print(f"[GPU {device}] Worker process started (PID: {os.getpid()})")
    
    while not work_queue.is_done():
        # Get next dataset that fits
        dataset, dimension, slots_needed = work_queue.get_next_dataset(slots_per_gpu)
        
        if dataset is None:
            # No suitable work available, wait a bit
            time.sleep(0.5)
            continue
        
        try:
            result = run_single_optimization(dataset, device, dimension, output_dir)
            results_list.append(result)
            work_queue.mark_completed(dataset)
            
            # Print progress
            completed, in_progress, total = work_queue.get_progress()
            print(f"[Progress] {completed}/{total} completed, {in_progress} in progress")
            
        except Exception as e:
            print(f"[GPU {device}] {dataset} failed: {e}")
            traceback.print_exc()
            work_queue.mark_failed(dataset)
    
    print(f"[GPU {device}] Worker process finished")


def run_dynamic_optimization(datasets: list, data_dir: str, output_dir: str = "./smac_output"):
    """
    Run optimization with dynamic GPU assignment using separate processes per GPU.
    """
    gpu_count = get_gpu_count()
    print(f"Found {gpu_count} GPUs")
    
    # Sort datasets by priority
    sorted_datasets = sort_datasets_by_priority(datasets, data_dir)
    # sorted_datasets = sorted_datasets[::-1]
    
    print(f"\nDataset priority order (first 10):")
    for i, ds in enumerate(sorted_datasets[:10]):
        dim = get_qap_dimension(ds, data_dir)
        slots = get_slots_for_dimension(dim)
        pattern = "priority" if not re.search(r"n(\d+)_", ds) else "standard"
        print(f"  {i+1}. {ds} (dim={dim}, slots={slots}, {pattern})")
        
    if len(sorted_datasets) > 10:
        print(f"  ... and {len(sorted_datasets) - 10} more")
    
    # Create shared state via Manager
    manager = Manager()
    work_queue = GPUWorkQueue(sorted_datasets, data_dir, manager)
    results_list = manager.list()
    
    # Start GPU worker processes
    processes = []
    for device in range(gpu_count):
        p = mp.Process(
            target=gpu_worker_process,
            args=(device, work_queue, output_dir, results_list)
        )
        p.start()
        processes.append(p)
        print(f"Started process for GPU {device} (PID: {p.pid})")
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Convert manager list to regular list
    return list(results_list)


def get_completed_datasets(output_dir: str = "./smac_output") -> set:
    """Get the set of datasets that have already been fully optimized."""
    completed = set()
    output_path = Path(output_dir)
    
    if not output_path.exists():
        return completed
    
    for dataset_dir in output_path.iterdir():
        if dataset_dir.is_dir():
            runhistory_path = dataset_dir / "runhistory.json"
            if runhistory_path.exists():
                try:
                    with open(runhistory_path) as f:
                        content = f.read()
                        if not content.strip():
                            continue
                        data = json.loads(content)
                        if len(data.get("data", [])) >= n_total:
                            completed.add(dataset_dir.name)
                except (json.JSONDecodeError, Exception) as e:
                    print(f"Warning: Could not read runhistory for {dataset_dir.name}: {e}")
                    continue
    
    return completed


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Configuration
    # data_dir = r"/home/masarnia/Desktop/Datasets/QAP/qapgen/"
    data_dir = r"/home/masarnia/Desktop/Datasets/QAP/isa_instances/"
    output_dir = "./merged_full"
    
    # Get datasets from directory
    datasets = os.listdir(path=data_dir)
    datasets = [dataset for dataset in datasets if ".dat" in dataset]
    datasets = [dataset.replace(".dat", "") for dataset in datasets]
    
    print(f"Found {len(datasets)} datasets")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out already completed datasets
    # completed_datasets = get_completed_datasets(output_dir)
    # datasets = [d for d in datasets if d not in completed_datasets]
    datasets = get_incomplete_datasets(datasets, output_dir)
    
    if not datasets:
        print("All datasets have been optimized!")
        exit(0)
    
    print(f"\nTwo-Phase Optimization Settings:")
    print(f"  Phase 1 (LHS): {n_lhs_configs} configs × {n_lhs_reps} reps = {n_lhs_configs * n_lhs_reps} evals")
    print(f"  Phase 2 (BO):  {n_bo_trials} evals")
    print(f"  Total:         {n_total} evals per dataset")
    print(f"\nGPU Parallelism:")
    print(f"  dim < {SMALL_THRESHOLD}: 2 datasets per GPU")
    print(f"  dim >= {SMALL_THRESHOLD}: 1 dataset per GPU")
    print(f"\nRemaining datasets: {len(datasets)}")
    
    # Run optimization
    results = run_dynamic_optimization(datasets, data_dir, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    successful = [(ds, cfg, val) for ds, cfg, val in results if cfg is not None]
    failed = [ds for ds, cfg, val in results if cfg is None]
    
    print(f"\nSuccessful: {len(successful)}")
    for dataset, config, best_value in sorted(successful, key=lambda x: x[2]):
        print(f"  {dataset}: {best_value}")
    
    if failed:
        print(f"\nFailed/Incomplete: {len(failed)}")
        for dataset in failed:
            print(f"  {dataset}")