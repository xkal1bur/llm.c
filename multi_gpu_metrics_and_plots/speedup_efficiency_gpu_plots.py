import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

# Constants for theoretical GFLOP/s calculation
N = 124475904
T = 256
C = 768
L = 12

def extract_gpu_and_batch_info(filename):
    """Extract GPU count and batch size from filename."""
    gpu_match = re.search(r'multigpu(\d+)_', filename)
    batch_match = re.search(r'_b(\d+)_', filename)
    
    gpu_count = int(gpu_match.group(1)) if gpu_match else None
    batch_size = int(batch_match.group(1)) if batch_match else None
    
    return gpu_count, batch_size

def get_metrics_mean(filepath, effective_batch_size):
    """Get mean values for all metrics, excluding first row."""
    df = pd.read_csv(filepath)
    # Skip first row (step 1) and calculate means
    df_mean = df.iloc[1:].mean()
    
    return {
        'total_time': df_mean['total_time_ms'] / 1000,  # Convert to seconds
        'computation_time': df_mean['pure_computation_time'] / 1000,
        'communication_time': df_mean['total_pure_comm_time_ms'] / 1000,
        'gflops_per_sec': df_mean['gflops_per_sec']
    }

def load_multi_gpu_data():
    """Load all multi-GPU CSV files and organize data."""
    singlegpu_data = {}
    multigpu2_data = {}
    multigpu4_data = {}
    multigpu8_data = {}
    
    # Process all multigpu CSV files
    for file in glob.glob('multigpu*.csv'):
        gpu_count, batch_size = extract_gpu_and_batch_info(file)
        if gpu_count and batch_size:
            effective_batch = gpu_count * batch_size  # GPU count * batch_size
            metrics = get_metrics_mean(file, effective_batch)
            
            if gpu_count == 1:
                singlegpu_data[effective_batch] = metrics
            elif gpu_count == 2:
                multigpu2_data[effective_batch] = metrics
            elif gpu_count == 4:
                multigpu4_data[effective_batch] = metrics
            elif gpu_count == 8:
                multigpu8_data[effective_batch] = metrics
    
    return singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data

def plot_speedup_total_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data):
    """Plot speedup vs number of GPUs using total time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlegpu data
    effective_batches = sorted(singlegpu_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlegpu_data:
            gpus = [1]  # Start with single GPU
            speedups = [1.0]  # Baseline speedup is 1
            
            singlegpu_time = singlegpu_data[effective_batch]['total_time']
            
            # Add 2-GPU if available
            if effective_batch in multigpu2_data:
                gpus.append(2)
                multigpu2_time = multigpu2_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu2_time
                speedups.append(speedup)
            
            # Add 4-GPU if available
            if effective_batch in multigpu4_data:
                gpus.append(4)
                multigpu4_time = multigpu4_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu4_time
                speedups.append(speedup)
            
            # Add 8-GPU if available
            if effective_batch in multigpu8_data:
                gpus.append(8)
                multigpu8_time = multigpu8_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu8_time
                speedups.append(speedup)
            
            plt.plot(gpus, speedups, marker='o', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal speedup line
    max_gpus = 8
    ideal_gpus = [1, 2, 4, 8]
    plt.plot(ideal_gpus, ideal_gpus, 'k--', alpha=0.5, label='Ideal Speedup', linewidth=2)
    
    plt.xlabel('Number of GPUs', fontweight='bold', fontsize=12)
    plt.ylabel('Speedup (Total Time)', fontweight='bold', fontsize=12)
    plt.title('Speedup vs Number of GPUs (Total Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4, 8])
    plt.tight_layout()
    plt.savefig('speedup_total_time_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup_computation_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data):
    """Plot speedup vs number of GPUs using computation time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlegpu data
    effective_batches = sorted(singlegpu_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlegpu_data:
            gpus = [1]  # Start with single GPU
            speedups = [1.0]  # Baseline speedup is 1
            
            singlegpu_time = singlegpu_data[effective_batch]['computation_time']
            
            # Add 2-GPU if available
            if effective_batch in multigpu2_data:
                gpus.append(2)
                multigpu2_time = multigpu2_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu2_time
                speedups.append(speedup)
            
            # Add 4-GPU if available
            if effective_batch in multigpu4_data:
                gpus.append(4)
                multigpu4_time = multigpu4_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu4_time
                speedups.append(speedup)
            
            # Add 8-GPU if available
            if effective_batch in multigpu8_data:
                gpus.append(8)
                multigpu8_time = multigpu8_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu8_time
                speedups.append(speedup)
            
            plt.plot(gpus, speedups, marker='s', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal speedup line
    ideal_gpus = [1, 2, 4, 8]
    plt.plot(ideal_gpus, ideal_gpus, 'k--', alpha=0.5, label='Ideal Speedup', linewidth=2)
    
    plt.xlabel('Number of GPUs', fontweight='bold', fontsize=12)
    plt.ylabel('Speedup (Computation Time)', fontweight='bold', fontsize=12)
    plt.title('Speedup vs Number of GPUs (Computation Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4, 8])
    plt.tight_layout()
    plt.savefig('speedup_computation_time_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_total_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data):
    """Plot efficiency vs number of GPUs using total time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlegpu data
    effective_batches = sorted(singlegpu_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlegpu_data:
            gpus = [1]  # Start with single GPU
            efficiencies = [1.0]  # Baseline efficiency is 1
            
            singlegpu_time = singlegpu_data[effective_batch]['total_time']
            
            # Add 2-GPU if available
            if effective_batch in multigpu2_data:
                gpus.append(2)
                multigpu2_time = multigpu2_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu2_time
                efficiency = speedup / 2  # 2 GPUs
                efficiencies.append(efficiency)
            
            # Add 4-GPU if available
            if effective_batch in multigpu4_data:
                gpus.append(4)
                multigpu4_time = multigpu4_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu4_time
                efficiency = speedup / 4  # 4 GPUs
                efficiencies.append(efficiency)
            
            # Add 8-GPU if available
            if effective_batch in multigpu8_data:
                gpus.append(8)
                multigpu8_time = multigpu8_data[effective_batch]['total_time']
                speedup = singlegpu_time / multigpu8_time
                efficiency = speedup / 8  # 8 GPUs
                efficiencies.append(efficiency)
            
            plt.plot(gpus, efficiencies, marker='^', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal efficiency line (should be 1.0)
    ideal_gpus = [1, 2, 4, 8]
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)', linewidth=2)
    
    plt.xlabel('Number of GPUs', fontweight='bold', fontsize=12)
    plt.ylabel('Efficiency (Speedup / GPUs)', fontweight='bold', fontsize=12)
    plt.title('Efficiency vs Number of GPUs (Total Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, 1.2)  # Set y-axis limit for better visualization
    plt.tight_layout()
    plt.savefig('efficiency_total_time_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_computation_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data):
    """Plot efficiency vs number of GPUs using computation time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlegpu data
    effective_batches = sorted(singlegpu_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlegpu_data:
            gpus = [1]  # Start with single GPU
            efficiencies = [1.0]  # Baseline efficiency is 1
            
            singlegpu_time = singlegpu_data[effective_batch]['computation_time']
            
            # Add 2-GPU if available
            if effective_batch in multigpu2_data:
                gpus.append(2)
                multigpu2_time = multigpu2_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu2_time
                efficiency = speedup / 2  # 2 GPUs
                efficiencies.append(efficiency)
            
            # Add 4-GPU if available
            if effective_batch in multigpu4_data:
                gpus.append(4)
                multigpu4_time = multigpu4_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu4_time
                efficiency = speedup / 4  # 4 GPUs
                efficiencies.append(efficiency)
            
            # Add 8-GPU if available
            if effective_batch in multigpu8_data:
                gpus.append(8)
                multigpu8_time = multigpu8_data[effective_batch]['computation_time']
                speedup = singlegpu_time / multigpu8_time
                efficiency = speedup / 8  # 8 GPUs
                efficiencies.append(efficiency)
            
            plt.plot(gpus, efficiencies, marker='D', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal efficiency line (should be 1.0)
    ideal_gpus = [1, 2, 4, 8]
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)', linewidth=2)
    
    plt.xlabel('Number of GPUs', fontweight='bold', fontsize=12)
    plt.ylabel('Efficiency (Speedup / GPUs)', fontweight='bold', fontsize=12)
    plt.title('Efficiency vs Number of GPUs (Computation Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, 1.2)  # Set y-axis limit for better visualization
    plt.tight_layout()
    plt.savefig('efficiency_computation_time_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_communication_overhead(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data):
    """Plot communication overhead vs number of GPUs."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in any GPU configuration
    all_batches = set(singlegpu_data.keys()) | set(multigpu2_data.keys()) | set(multigpu4_data.keys()) | set(multigpu8_data.keys())
    effective_batches = sorted(all_batches)
    
    for effective_batch in effective_batches:
        gpus = []
        comm_ratios = []
        
        # Check each GPU configuration
        for gpu_count, data in [(1, singlegpu_data), (2, multigpu2_data), (4, multigpu4_data), (8, multigpu8_data)]:
            if effective_batch in data:
                gpus.append(gpu_count)
                comm_time = data[effective_batch]['communication_time']
                total_time = data[effective_batch]['total_time']
                comm_ratio = comm_time / total_time if total_time > 0 else 0
                comm_ratios.append(comm_ratio)
        
        if len(gpus) > 1:  # Only plot if we have multiple GPU configurations
            plt.plot(gpus, comm_ratios, marker='o', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    plt.xlabel('Number of GPUs', fontweight='bold', fontsize=12)
    plt.ylabel('Communication Overhead Ratio', fontweight='bold', fontsize=12)
    plt.title('Communication Overhead vs Number of GPUs', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4, 8])
    plt.ylim(0, None)  # Start y-axis from 0
    plt.tight_layout()
    plt.savefig('communication_overhead_gpu.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to load data and generate all speedup and efficiency plots."""
    print("Loading multi-GPU data from CSV files...")
    singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data = load_multi_gpu_data()
    
    # Check if data was loaded successfully
    if not singlegpu_data:
        print("No single GPU data found. Please check if the CSV files exist.")
        return
    
    print("Generating speedup and efficiency plots...")

    # --- DEBUGGING ---
    print("Single-GPU data:", singlegpu_data)
    print("Multi-GPU 2 data:", multigpu2_data)
    print("Multi-GPU 4 data:", multigpu4_data)
    print("Multi-GPU 8 data:", multigpu8_data)
    
    # Generate all plots
    plot_speedup_total_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data)
    plot_speedup_computation_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data)
    plot_efficiency_total_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data)
    plot_efficiency_computation_time(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data)
    plot_communication_overhead(singlegpu_data, multigpu2_data, multigpu4_data, multigpu8_data)
    
    print("All speedup and efficiency plots generated successfully!")
    print("Generated files:")
    print("- speedup_total_time_gpu.png")
    print("- speedup_computation_time_gpu.png")
    print("- efficiency_total_time_gpu.png")
    print("- efficiency_computation_time_gpu.png")
    print("- communication_overhead_gpu.png")

if __name__ == "__main__":
    main() 