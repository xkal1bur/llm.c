import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Total floating-point operations per step (you specified this)
FLOP_PER_STEP = 3291048050688

def find_csv_files():
    """Find all OMP_*.csv files in the current directory."""
    return [f for f in os.listdir() if re.match(r'OMP_(\d+|seq)\.csv$', f)]

def extract_thread_count(filename):
    """Extract thread count from filename (OMP_16.csv → 16, OMP_seq.csv → 1)."""
    match = re.match(r'OMP_(\d+|seq)\.csv$', filename)
    if match:
        value = match.group(1)
        return 1 if value == "seq" else int(value)
    return None

def read_and_average_computation_time(filename):
    """Read CSV and compute average of computation_time_ms."""
    df = pd.read_csv(filename)
    return df["computation_time_ms"].mean()

def compute_gflops_cpu(filename):
    """Compute average GFLOP/s per step for a CPU CSV using FLOP_PER_STEP and per-step time."""
    df = pd.read_csv(filename)
    gflops_per_step = FLOP_PER_STEP / (df["computation_time_ms"] / 1000.0) / 1e9
    return gflops_per_step.mean()

def compute_gflops_gpu(filename):
    """Return average GFLOP/s per step from GPU CSV."""
    df = pd.read_csv(filename)
    return df["gflops_per_sec"].mean()

def main():
    files = find_csv_files()
    print("Found OMP CSV files:", files)
    
    time_data = []       # (threads, avg_time)
    gflops_data = []     # (threads, gflops)

    for f in files:
        threads = extract_thread_count(f)
        avg_time = read_and_average_computation_time(f)
        gflops = compute_gflops_cpu(f)
        time_data.append((threads, avg_time))
        gflops_data.append((threads, gflops))

    # Add GPU data (assumed 32 threads placeholder)
    gpu_filename = "training_metrics.csv"
    if os.path.exists(gpu_filename):
        try:
            gpu_avg_time = read_and_average_computation_time(gpu_filename)
            gpu_gflops = compute_gflops_gpu(gpu_filename)
            time_data.append((32, gpu_avg_time))
            gflops_data.append((32, gpu_gflops))
            print(f"Added GPU data: threads=32, avg_time={gpu_avg_time:.2f} ms, gflops={gpu_gflops:.2f}")
        except Exception as e:
            print(f"Failed to read GPU file: {e}")

    # ---- Plot 1: Computation Time ----
    time_data.sort()
    threads_time, avg_times = zip(*time_data)

    plt.figure(figsize=(10, 6))
    plt.plot(threads_time, avg_times, marker='o', linestyle='-', color='blue', label='OMP (CPU)')
    if 32 in threads_time:
        idx = threads_time.index(32)
        plt.plot(32, avg_times[idx], 'ro', label='GPU (Placeholder 32 Threads)', markersize=10)
    plt.xlabel("Number of Threads")
    plt.ylabel("Average Computation Time (ms)")
    plt.title("Average Computation Time vs Number of Threads (CPU vs GPU)")
    plt.grid(True)
    plt.xticks(sorted(set(threads_time)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("avg_computation_time_with_gpu.png")
    plt.show()

    # ---- Plot 2: GFLOP/s per Step ----
    gflops_data.sort()
    threads_gflops, gflops_values = zip(*gflops_data)

    plt.figure(figsize=(10, 6))
    plt.plot(threads_gflops, gflops_values, marker='s', linestyle='-', color='green', label='OMP (CPU)')
    if 32 in threads_gflops:
        idx = threads_gflops.index(32)
        plt.plot(32, gflops_values[idx], 'r^', label='GPU (Placeholder 32 Threads)', markersize=10)
    plt.xlabel("Number of Threads")
    plt.ylabel("GFLOP/s per Step")
    plt.title("GFLOP/s per Step vs Number of Threads (CPU vs GPU)")
    plt.grid(True)
    plt.xticks(sorted(set(threads_gflops)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("gflops_per_step_vs_threads.png")
    plt.show()

    # ---- Plot 3: Speedup vs Threads ----
    try:
        sequential_time = next(t for threads, t in time_data if threads == 1)
    except StopIteration:
        print("Error: No sequential data found. Cannot compute speedup.")
        return

    speedup_data = [(threads, sequential_time / t) for threads, t in time_data]
    speedup_data.sort()
    threads_speedup, speedups = zip(*speedup_data)

    plt.figure(figsize=(10, 6))
    plt.plot(threads_speedup, speedups, marker='^', linestyle='-', color='purple', label='Speedup')
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup (Sequential Time / Parallel Time)")
    plt.title("Speedup vs Number of Threads (CPU vs GPU)")
    plt.grid(True)
    plt.xticks(sorted(set(threads_speedup)))
    plt.legend()
    plt.tight_layout()
    plt.savefig("speedup_vs_threads.png")
    plt.show()

if __name__ == "__main__":
    main()
