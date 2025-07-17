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
    """Get mean values for all metrics, excluding first row"""
    df = pd.read_csv(filepath)
    # Skip first row (step 1) and calculate means
    df_mean = df.iloc[1:].mean()
    
    # Calculate theoretical GFLOP/s using BT(6N + 6LCT) formula
    B = effective_batch_size
    flops_per_step = B * T * (6 * N + 6 * L * C * T)
    computation_time_s = df_mean['pure_computation_time'] / 1000  # Convert to seconds
    theoretical_gflops = (flops_per_step / computation_time_s) / 1e9
    
    # Calculate GFLOP/s based on computation time only
    computation_gflops = (flops_per_step / computation_time_s) / 1e9
    
    return {
        'total_time': df_mean['total_time_ms'] / 1000,  # Convert to seconds
        'computation_time': df_mean['pure_computation_time'] / 1000,
        'communication_time': df_mean['total_pure_comm_time_ms'] / 1000,
        'pure_communication_time': df_mean['total_pure_comm_time_ms'] / 1000,
        'gflops_per_sec': df_mean['gflops_per_sec'],
        'theoretical_gflops': theoretical_gflops,
        'computation_gflops': computation_gflops
    }

# Read data
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

# Get all effective batch sizes and sort
all_batch_sizes = sorted(set(
    list(singlegpu_data.keys()) + 
    list(multigpu2_data.keys()) + 
    list(multigpu4_data.keys()) + 
    list(multigpu8_data.keys())
))

# Create subplot figure with 2x4 grid for 8 plots
fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 10))
fig.suptitle('Multi-GPU Performance Analysis', fontsize=16, fontweight='bold')

# Metrics to plot
metrics = ['total_time', 'computation_time', 'communication_time', 'pure_communication_time',
           'gflops_per_sec', 'theoretical_gflops', 'computation_gflops']
titles = ['Total Time', 'Computation Time', 'Communication Time', 'Pure Communication Time',
          'GFLOPS per Second', 'Theoretical GFLOPS', 'Computation GFLOPS']
ylabels = ['Time (s)', 'Time (s)', 'Time (s)', 'Time (s)', 'GFLOPS/s', 'GFLOPS/s', 'GFLOPS/s']
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

# Add an 8th plot for communication overhead
metrics.append('comm_overhead_ratio')
titles.append('Communication Overhead Ratio')
ylabels.append('Ratio')

for i, (metric, title, ylabel, ax) in enumerate(zip(metrics, titles, ylabels, axes)):
    # Prepare data for plotting
    singlegpu_values = []
    multigpu2_values = []
    multigpu4_values = []
    multigpu8_values = []
    
    for bs in all_batch_sizes:
        # Handle communication overhead ratio calculation
        if metric == 'comm_overhead_ratio':
            singlegpu_val = singlegpu_data.get(bs, {}).get('communication_time', 0) / singlegpu_data.get(bs, {}).get('total_time', 1) if bs in singlegpu_data else None
            multigpu2_val = multigpu2_data.get(bs, {}).get('communication_time', 0) / multigpu2_data.get(bs, {}).get('total_time', 1) if bs in multigpu2_data else None
            multigpu4_val = multigpu4_data.get(bs, {}).get('communication_time', 0) / multigpu4_data.get(bs, {}).get('total_time', 1) if bs in multigpu4_data else None
            multigpu8_val = multigpu8_data.get(bs, {}).get('communication_time', 0) / multigpu8_data.get(bs, {}).get('total_time', 1) if bs in multigpu8_data else None
        else:
            singlegpu_val = singlegpu_data.get(bs, {}).get(metric)
            multigpu2_val = multigpu2_data.get(bs, {}).get(metric)
            multigpu4_val = multigpu4_data.get(bs, {}).get(metric)
            multigpu8_val = multigpu8_data.get(bs, {}).get(metric)
        
        singlegpu_values.append(singlegpu_val)
        multigpu2_values.append(multigpu2_val)
        multigpu4_values.append(multigpu4_val)
        multigpu8_values.append(multigpu8_val)
    
    # Filter out None values
    singlegpu_x = [bs for bs, val in zip(all_batch_sizes, singlegpu_values) if val is not None]
    singlegpu_y = [val for val in singlegpu_values if val is not None]
    
    multigpu2_x = [bs for bs, val in zip(all_batch_sizes, multigpu2_values) if val is not None]
    multigpu2_y = [val for val in multigpu2_values if val is not None]
    
    multigpu4_x = [bs for bs, val in zip(all_batch_sizes, multigpu4_values) if val is not None]
    multigpu4_y = [val for val in multigpu4_values if val is not None]
    
    multigpu8_x = [bs for bs, val in zip(all_batch_sizes, multigpu8_values) if val is not None]
    multigpu8_y = [val for val in multigpu8_values if val is not None]
    
    # Plot lines
    if singlegpu_x:
        ax.plot(singlegpu_x, singlegpu_y, marker='o', label='Single GPU (1 GPU)', 
                linewidth=2.5, markersize=8, color='#F18F01')
    if multigpu2_x:
        ax.plot(multigpu2_x, multigpu2_y, marker='s', label='Multi-GPU (2 GPUs)', 
                linewidth=2.5, markersize=8, color='#2E86AB')
    if multigpu4_x:
        ax.plot(multigpu4_x, multigpu4_y, marker='^', label='Multi-GPU (4 GPUs)', 
                linewidth=2.5, markersize=8, color='#A23B72')
    if multigpu8_x:
        ax.plot(multigpu8_x, multigpu8_y, marker='D', label='Multi-GPU (8 GPUs)', 
                linewidth=2.5, markersize=8, color='#C73E1D')
    
    ax.set_xlabel('Effective Batch Size (GPUs × batch)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks
    all_x = sorted(set(singlegpu_x + multigpu2_x + multigpu4_x + multigpu8_x))
    if all_x:
        ax.set_xticks(all_x)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.93)

# Save figure in multiple formats
plt.savefig('multi_gpu_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('multi_gpu_performance_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('multi_gpu_performance_analysis.svg', dpi=300, bbox_inches='tight')

print("Figure saved as:")
print("- multi_gpu_performance_analysis.png")
print("- multi_gpu_performance_analysis.pdf") 
print("- multi_gpu_performance_analysis.svg")

plt.show() 