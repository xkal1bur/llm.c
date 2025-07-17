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

def extract_batch_size(filename):
    match = re.search(r'_b(\d+)_', filename)
    return int(match.group(1)) if match else None

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
        'gflops_per_sec': df_mean['gflops_per_sec'],
        'theoretical_gflops': theoretical_gflops,
        'computation_gflops': computation_gflops
    }

# Read data
singlenode_data = {}
binode_data = {}
tetranode_data = {}

# Process singlenode files (1 node)
for file in glob.glob('singlenode_*.csv'):
    batch_size = extract_batch_size(file)
    if batch_size:
        effective_batch = 1 * batch_size  # 1 node * batch_size
        metrics = get_metrics_mean(file, effective_batch)
        singlenode_data[effective_batch] = metrics

# Process binode files (2 nodes)
for file in glob.glob('binode_*.csv'):
    batch_size = extract_batch_size(file)
    if batch_size:
        effective_batch = 2 * batch_size  # 2 nodes * batch_size
        metrics = get_metrics_mean(file, effective_batch)
        binode_data[effective_batch] = metrics

# Process tetranode files (4 nodes)
for file in glob.glob('tetranode_*.csv'):
    batch_size = extract_batch_size(file)
    if batch_size:
        effective_batch = 4 * batch_size  # 4 nodes * batch_size  
        metrics = get_metrics_mean(file, effective_batch)
        tetranode_data[effective_batch] = metrics

# Get all effective batch sizes and sort
all_batch_sizes = sorted(set(list(singlenode_data.keys()) + list(binode_data.keys()) + list(tetranode_data.keys())))

# Create subplot figure with 2x3 grid for 6 plots
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Multi-Node Performance Analysis', fontsize=16, fontweight='bold')

# Metrics to plot
metrics = ['total_time', 'computation_time', 'communication_time', 'gflops_per_sec', 'theoretical_gflops', 'computation_gflops']
titles = ['Total Time', 'Computation Time', 'Communication Time', 'GFLOPS per Second', 'Theoretical GFLOPS', 'Computation GFLOPS']
ylabels = ['Time (s)', 'Time (s)', 'Time (s)', 'GFLOPS/s', 'GFLOPS/s', 'GFLOPS/s']
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

for i, (metric, title, ylabel, ax) in enumerate(zip(metrics, titles, ylabels, axes)):
    # Prepare data for plotting
    singlenode_values = [singlenode_data.get(bs, {}).get(metric) for bs in all_batch_sizes]
    binode_values = [binode_data.get(bs, {}).get(metric) for bs in all_batch_sizes]
    tetranode_values = [tetranode_data.get(bs, {}).get(metric) for bs in all_batch_sizes]
    
    # Filter out None values
    singlenode_x = [bs for bs, val in zip(all_batch_sizes, singlenode_values) if val is not None]
    singlenode_y = [val for val in singlenode_values if val is not None]
    
    binode_x = [bs for bs, val in zip(all_batch_sizes, binode_values) if val is not None]
    binode_y = [val for val in binode_values if val is not None]
    
    tetranode_x = [bs for bs, val in zip(all_batch_sizes, tetranode_values) if val is not None]
    tetranode_y = [val for val in tetranode_values if val is not None]
    
    # Plot lines
    ax.plot(singlenode_x, singlenode_y, marker='o', label='Singlenode (1 node)', 
            linewidth=2.5, markersize=8, color='#F18F01')
    ax.plot(binode_x, binode_y, marker='s', label='Binode (2 nodes)', 
            linewidth=2.5, markersize=8, color='#2E86AB')
    ax.plot(tetranode_x, tetranode_y, marker='^', label='Tetranode (4 nodes)', 
            linewidth=2.5, markersize=8, color='#A23B72')
    
    ax.set_xlabel('Effective Batch Size (nodes × batch)', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=10)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis ticks
    if singlenode_x or binode_x or tetranode_x:
        all_x = sorted(set(singlenode_x + binode_x + tetranode_x))
        ax.set_xticks(all_x)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('multi_nodes_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show() 