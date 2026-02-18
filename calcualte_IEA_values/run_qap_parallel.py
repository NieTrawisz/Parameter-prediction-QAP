#!/usr/bin/env python3
"""
Parallel QAP Runner with Queue-based Checkpointing
"""

import os
import sys
import csv
import json
import signal
import time
import threading
import queue
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
from datetime import datetime
from ctypes import cdll, c_char_p, c_int, c_float, c_double
import random
import multiprocessing as mp

# Configuration
NUM_SEEDS = 30
GENERATIONS = 50
MAX_ISLANDS = 10

SELECTION_TYPES = ["Roulette", "LinearRank", "NonlinearRank", "Threshold", "Tournament"]
SUCCESSION_TYPES = ["NoElitism", "Elitism", "SteadyState"]

shutdown_requested = threading.Event()

def signal_handler(signum, frame):
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Shutdown requested...")
    shutdown_requested.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def get_gpu_count_safe():
    """
    Get GPU count WITHOUT initializing CUDA in the parent process.
    Use nvidia-smi or environment variable instead of CUDA API.
    """
    # Try environment variable first
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        return len([d for d in devices if d.strip()])
    
    # Try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return len([line for line in result.stdout.strip().split("\n") if line])
    except Exception:
        pass
    
    return 1  # Default fallback


def load_checkpoint(checkpoint_file) -> set:
    """Load checkpoint as a set for fast lookup."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                completed = set(data.get("completed", []))
                # Normalize: ensure single underscore format (handles legacy double-underscore)
                normalized = set(tid.replace("__seed", "_seed") for tid in completed)
                return normalized
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return set()


def save_checkpoint_atomic(completed_set: set, checkpoint_file: str):
    """Save checkpoint atomically."""
    temp_file = checkpoint_file + ".tmp"
    with open(temp_file, 'w') as f:
        json.dump({"completed": list(completed_set)}, f)
    os.replace(temp_file, checkpoint_file)


def init_results_file(results_file):
    if not os.path.exists(results_file):
        headers = [
            "scenario_name", "seed", "result", 
            "device", "generations", "timestamp", "walltime_sec"
        ]
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)


def run_single_qap(dataset, device, generations, seed, config, max_islands=MAX_ISLANDS):
    """
    CRITICAL: This runs INSIDE the worker process, so CUDA init happens post-fork.
    """
    succession_idx = SUCCESSION_TYPES.index(config["succession"])
    reproduction_idx = SELECTION_TYPES.index(config["reproduction"])
    migration_idx = SELECTION_TYPES.index(config["migration"])
    release_idx = SELECTION_TYPES.index(config["release"])
    
    def make_array(val):
        return (c_float * max_islands)(*([val] * max_islands))
    
    # Load library and init CUDA HERE, not in parent
    qap_dll = cdll.LoadLibrary(os.getcwd() + "/gaqap_simplified.so")
    qap_dll.run.restype = c_double
    
    result = qap_dll.run(
        c_char_p(dataset.encode("utf-8")),
        c_int(device),
        c_int(generations),
        c_int(seed),
        c_int(succession_idx),
        c_int(reproduction_idx),
        c_float(1.0),
        c_float(50.0),
        c_float(0.5),
        make_array(config["probNothing"]),
        make_array(config["probMutateSwap"]),
        make_array(config["probMutateShift"]),
        make_array(config["probMutateInverse"]),
        make_array(config["probPMXcrossover"]),
        make_array(config["probOXcrossover"]),
        make_array(config["probCXcrossover"]),
        c_int(config["islands_num"]),
        c_int(config["migration_freq"]),
        c_float(config["migration_batch"]),
        c_int(migration_idx),
        c_int(release_idx),
    )
    return float(result)


def worker_task(args):
    """
    Worker function - returns result or None.
    NO file I/O here - just computation.
    CUDA context created fresh in this process.
    """
    scenario_name, config, seed, device, generations = args
    
    # FIX: Single underscore to match checkpoint format
    task_id = f"{scenario_name}_seed{seed}"
    start_time = time.time()
    
    try:
        result = run_single_qap(
            dataset=scenario_name,
            device=device,
            generations=generations,
            seed=seed,
            config=config
        )
        
        walltime = time.time() - start_time
        
        return {
            "task_id": task_id,
            "scenario_name": scenario_name,
            "seed": seed,
            "result": result,
            "device": device,
            "generations": generations,
            "timestamp": datetime.now().isoformat(),
            "walltime_sec": round(walltime, 3),
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "task_id": task_id,
            "success": False,
            "error": str(e),
            "scenario_name": scenario_name,
            "seed": seed,
        }


def checkpoint_writer(checkpoint_file: str, results_file: str, result_queue: mp.Queue, 
                      stop_event: threading.Event, total_tasks: int):
    """
    Dedicated thread for writing checkpoint and results.
    Single writer prevents file corruption and permission errors.
    """
    completed = load_checkpoint(checkpoint_file)
    last_save_count = len(completed)
    
    print(f"[Writer] Loaded {last_save_count} completed tasks")
    
    batch_results = []
    batch_size = 10  # Write every N results
    
    while not stop_event.is_set() or not result_queue.empty():
        try:
            result = result_queue.get(timeout=1.0)
            
            if result is None:  # Poison pill
                break
                
            if result["success"]:
                completed.add(result["task_id"])
                batch_results.append(result)
                
                # Write results CSV in batch
                if len(batch_results) >= batch_size:
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for r in batch_results:
                            writer.writerow([
                                r["scenario_name"],  r["seed"],
                                r["result"], r["device"], r["generations"],
                                r["timestamp"], r["walltime_sec"]
                            ])
                    batch_results = []
                
                # Save checkpoint periodically
                if len(completed) % 50 == 0 and len(completed) != last_save_count:
                    save_checkpoint_atomic(completed, checkpoint_file)
                    last_save_count = len(completed)
                    print(f"[Writer] Checkpoint saved: {len(completed)}/{total_tasks}")
                    
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Writer] Error: {e}")
    
    # Final flush
    if batch_results:
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for r in batch_results:
                writer.writerow([
                    r["scenario_name"], r["seed"],
                    r["result"], r["device"], r["generations"],
                    r["timestamp"], r["walltime_sec"]
                ])
    
    save_checkpoint_atomic(completed, checkpoint_file)
    print(f"[Writer] Final checkpoint: {len(completed)}/{total_tasks}")


def run_parallel_qap(best_configs_path: str, results_file: str, checkpoint_file: str, 
                     max_workers: Optional[int] = None):
    import pandas as pd
    
    configs_df = pd.read_csv(best_configs_path)
    
    # FIX: Use safe GPU detection that doesn't initialize CUDA
    gpu_count = get_gpu_count_safe()
    
    if max_workers is None:
        max_workers = gpu_count
    print(f"Detected {gpu_count} GPU(s), using {max_workers} workers")
    
    # Initialize files
    init_results_file(results_file)
    completed = load_checkpoint(checkpoint_file)
    
    # Build task list
    tasks = []
    for _, row in configs_df.iterrows():
        config = {
            "islands_num": int(row["islands_num"]),
            "migration": row["migration"],
            "migration_batch": float(row["migration_batch"]),
            "migration_freq": int(row["migration_freq"]),
            "probCXcrossover": float(row["probCXcrossover"]),
            "probMutateInverse": float(row["probMutateInverse"]),
            "probMutateShift": float(row["probMutateShift"]),
            "probMutateSwap": float(row["probMutateSwap"]),
            "probNothing": float(row["probNothing"]),
            "probOXcrossover": float(row["probOXcrossover"]),
            "probPMXcrossover": float(row["probPMXcrossover"]),
            "release": row["release"],
            "reproduction": row["reproduction"],
            "succession": row["succession"]
        }
        for seed in range(NUM_SEEDS):
            tasks.append((row["scenario_name"], config, seed))
    
    total_tasks = len(tasks)
    already_done = len(completed)
    remaining = total_tasks - already_done
    
    print(f"Total: {total_tasks}, Done: {already_done}, Remaining: {remaining}")
    
    if remaining == 0:
        print("All done!")
        return
    
    # Filter already done - FIX: Use single underscore format
    pending_tasks = []
    gpu_cycle = 0
    for scenario_name, config, seed in tasks:
        task_id = f"{scenario_name}_seed{seed}"  # Single underscore!
        if task_id not in completed:
            device = gpu_cycle % gpu_count
            gpu_cycle += 1
            pending_tasks.append((scenario_name, config, seed, device, GENERATIONS))
    
    # Setup queue and writer thread
    result_queue = mp.Queue()
    stop_event = threading.Event()
    
    writer_thread = threading.Thread(
        target=checkpoint_writer,
        args=(checkpoint_file, results_file, result_queue, stop_event, total_tasks)
    )
    writer_thread.start()
    
    completed_count = 0
    failed_count = 0
    
    # CRITICAL: Use spawn instead of fork to avoid CUDA issues
    mp_context = mp.get_context('spawn')
    
    try:
        # Use spawn context to ensure clean CUDA initialization in workers
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(worker_task, args): args 
                for args in pending_tasks
            }
            
            for future in as_completed(future_to_task):
                if shutdown_requested.is_set():
                    print("\nCancelling remaining tasks...")
                    for f in future_to_task:
                        if not f.done():
                            f.cancel()
                    break
                
                try:
                    result = future.result()
                    result_queue.put(result)
                    
                    if result["success"]:
                        completed_count += 1
                        print(f"  [DONE: {result['task_id']}: result={result['result']:.2f}, time={result['walltime_sec']:.1f}s]")
                    else:
                        failed_count += 1
                        print(f"  [ERROR: {result['task_id']}: {result['error']}]")
                        
                except Exception as e:
                    args = future_to_task[future]
                    scenario, config, seed, device, gens = args
                    # FIX: Correct argument unpacking for error message
                    print(f"  [FATAL ERROR: {scenario}_seed{seed}: {e}]")
                    failed_count += 1
                
                # Progress update
                if (completed_count + failed_count) % 10 == 0:
                    print(f"\nProgress: ~{already_done + completed_count}/{total_tasks} "
                          f"({100*(already_done + completed_count)/total_tasks:.1f}%) - "
                          f"This run: {completed_count}, Failed: {failed_count}\n")
    
    finally:
        # Signal writer to finish
        print("[Main] Shutting down writer...")
        result_queue.put(None)  # Poison pill
        stop_event.set()
        writer_thread.join(timeout=30)
        
        # Force final checkpoint reload to verify
        final_completed = load_checkpoint(checkpoint_file)
        print(f"\n{'='*60}")
        print(f"Run finished!")
        print(f"Checkpoint reports: {len(final_completed)}/{total_tasks}")
        print(f"This run processed: {completed_count} succeeded, {failed_count} failed")
        
        if len(final_completed) != already_done + completed_count:
            print(f"WARNING: Discrepancy detected!")
            print(f"  Expected: {already_done + completed_count}")
            print(f"  Actual: {len(final_completed)}")
            print(f"  Missing: {already_done + completed_count - len(final_completed)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="single_target_predictions.csv")
    parser.add_argument("--results", default="qap_single_predictions.csv")
    parser.add_argument("--checkpoint", default="single_predictions_checkpoint.json")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--generations", type=int, default=50)
    args = parser.parse_args()
    
    GENERATIONS = args.generations
    
    run_parallel_qap(args.configs, args.results, args.checkpoint, args.workers)