import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Configuration
batch_sizes = [4, 8, 16]
thread_configs = ['seq', '2', '4', '8', '16']
thread_numbers = [1, 2, 4, 8, 16]  # seq = 1 thread

# Constants for theoretical GFLOP/s calculation
N = 124475904
T = 256
C = 768
L = 12

def load_data():
    """Load all CSV files and organize data by batch size and thread count."""
    data = {}
    
    for batch_size in batch_sizes:
        data[batch_size] = {}
        folder_path = f"batch{batch_size}"
        
        for i, thread_config in enumerate(thread_configs):
            file_path = f"{folder_path}/OMP_{thread_config}.csv"
            thread_num = thread_numbers[i]
            
            try:
                # Read CSV file
                df = pd.read_csv(file_path)
                
                # Calculate averages
                avg_computation_time = df['computation_time_ms'].mean()
                avg_gflops_per_sec = df['gflops_per_sec'].mean()
                
                data[batch_size][thread_num] = {
                    'avg_computation_time': avg_computation_time,
                    'avg_gflops_per_sec': avg_gflops_per_sec
                }
                
            except FileNotFoundError:
                print(f"Warning: File {file_path} not found")
                continue
    
    return data

def plot_average_iteration_time(data):
    """Plot 1: Average Iteration Time vs Number of threads."""
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        threads = []
        times = []
        
        for thread_num in sorted(data[batch_size].keys()):
            threads.append(thread_num)
            times.append(data[batch_size][thread_num]['avg_computation_time'])
        
        plt.plot(threads, times, marker='o', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Iteration Time (ms)')
    plt.title('Average Iteration Time vs Number of Threads')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_numbers, thread_numbers)
    plt.tight_layout()
    plt.savefig('average_iteration_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_average_gflops(data):
    """Plot 2: Average GFLOP/s per step vs Number of threads."""
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        threads = []
        gflops = []
        
        for thread_num in sorted(data[batch_size].keys()):
            threads.append(thread_num)
            gflops.append(data[batch_size][thread_num]['avg_gflops_per_sec'])
        
        plt.plot(threads, gflops, marker='o', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Average GFLOP/s')
    plt.title('Average GFLOP/s per Step vs Number of Threads')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_numbers, thread_numbers)
    plt.tight_layout()
    plt.savefig('average_gflops.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup(data):
    """Plot 3: Speedup vs Number of threads."""
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        threads = []
        speedups = []
        
        # Get sequential time (1 thread)
        seq_time = data[batch_size][1]['avg_computation_time']
        
        for thread_num in sorted(data[batch_size].keys()):
            threads.append(thread_num)
            parallel_time = data[batch_size][thread_num]['avg_computation_time']
            speedup = seq_time / parallel_time
            speedups.append(speedup)
        
        plt.plot(threads, speedups, marker='o', label=f'Batch Size {batch_size}')
    
    # Add ideal speedup line
    plt.plot(thread_numbers, thread_numbers, 'k--', alpha=0.5, label='Ideal Speedup')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Number of Threads')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_numbers, thread_numbers)
    plt.tight_layout()
    plt.savefig('speedup.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency(data):
    """Plot 5: Efficiency vs Number of threads (Speedup / Number of threads)."""
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        threads = []
        efficiencies = []
        
        # Get sequential time (1 thread)
        seq_time = data[batch_size][1]['avg_computation_time']
        
        for thread_num in sorted(data[batch_size].keys()):
            threads.append(thread_num)
            parallel_time = data[batch_size][thread_num]['avg_computation_time']
            speedup = seq_time / parallel_time
            efficiency = speedup / thread_num
            efficiencies.append(efficiency)
        
        plt.plot(threads, efficiencies, marker='o', label=f'Batch Size {batch_size}')
    
    # Add ideal efficiency line (should be 1.0)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Efficiency (Speedup / Threads)')
    plt.title('Efficiency vs Number of Threads')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_numbers, thread_numbers)
    plt.tight_layout()
    plt.savefig('efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_theoretical_gflops(data):
    """Plot 4: Theoretical GFLOP/s calculation using BT(6N + 6LCT) formula."""
    plt.figure(figsize=(10, 6))
    
    for batch_size in batch_sizes:
        B = batch_size
        threads = []
        theoretical_gflops = []
        
        for thread_num in sorted(data[batch_size].keys()):
            threads.append(thread_num)
            
            # Calculate number of floating point operations per step
            flops_per_step = B * T * (6 * N + 6 * L * C * T)
            
            # Get iteration time in seconds
            iteration_time_ms = data[batch_size][thread_num]['avg_computation_time']
            iteration_time_s = iteration_time_ms / 1000.0
            
            # Calculate GFLOP/s
            gflops = (flops_per_step / iteration_time_s) / 1e9
            theoretical_gflops.append(gflops)
        
        plt.plot(threads, theoretical_gflops, marker='o', label=f'Batch Size {batch_size}')
    
    plt.xlabel('Number of Threads')
    plt.ylabel('Theoretical GFLOP/s')
    plt.title('Theoretical GFLOP/s vs Number of Threads\n(Using BT(6N + 6LCT) formula)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(thread_numbers, thread_numbers)
    plt.tight_layout()
    plt.savefig('theoretical_gflops.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to load data and generate all plots."""
    print("Loading data from CSV files...")
    data = load_data()
    
    # Check if data was loaded successfully
    if not data:
        print("No data found. Please check if the CSV files exist in the expected locations.")
        return
    
    print("Generating plots...")
    
    # Generate all five plots
    plot_average_iteration_time(data)
    plot_average_gflops(data)
    plot_speedup(data)
    plot_efficiency(data)
    plot_theoretical_gflops(data)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()