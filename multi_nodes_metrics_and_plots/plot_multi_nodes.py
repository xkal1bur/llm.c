import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np

def extract_batch_size(filename):
    match = re.search(r'_b(\d+)_', filename)
    return int(match.group(1)) if match else None

def get_metrics_mean(filepath):
    """Get mean values for all metrics, excluding first row"""
    df = pd.read_csv(filepath)
    # Skip first row (step 1) and calculate means
    df_mean = df.iloc[1:].mean()
    return {
        'total_time': df_mean['total_time_ms'] / 1000,  # Convert to seconds
        'computation_time': df_mean['computation_time_ms'] / 1000,
        'communication_time': df_mean['communication_time_ms'] / 1000,
        'gflops_per_sec': df_mean['gflops_per_sec']
    }

# Read data
binode_data = {}
tetranode_data = {}

# Process binode files (2 nodes)
for file in glob.glob('binode_*.csv'):
    batch_size = extract_batch_size(file)
    if batch_size:
        effective_batch = 2 * batch_size  # 2 nodes * batch_size
        metrics = get_metrics_mean(file)
        binode_data[effective_batch] = metrics

# Process tetranode files (4 nodes)
for file in glob.glob('tetranode_*.csv'):
    batch_size = extract_batch_size(file)
    if batch_size:
        effective_batch = 4 * batch_size  # 4 nodes * batch_size  
        metrics = get_metrics_mean(file)
        tetranode_data[effective_batch] = metrics

# Get all effective batch sizes and sort
all_batch_sizes = sorted(set(list(binode_data.keys()) + list(tetranode_data.keys())))

# Create subplot figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Multi-Node Performance Analysis', fontsize=16, fontweight='bold')

# Metrics to plot
metrics = ['total_time', 'computation_time', 'communication_time', 'gflops_per_sec']
titles = ['Total Time', 'Computation Time', 'Communication Time', 'GFLOPS per Second']
ylabels = ['Time (s)', 'Time (s)', 'Time (s)', 'GFLOPS/s']
axes = [ax1, ax2, ax3, ax4]

for i, (metric, title, ylabel, ax) in enumerate(zip(metrics, titles, ylabels, axes)):
    # Prepare data for plotting
    binode_values = [binode_data.get(bs, {}).get(metric) for bs in all_batch_sizes]
    tetranode_values = [tetranode_data.get(bs, {}).get(metric) for bs in all_batch_sizes]
    
    # Filter out None values
    binode_x = [bs for bs, val in zip(all_batch_sizes, binode_values) if val is not None]
    binode_y = [val for val in binode_values if val is not None]
    
    tetranode_x = [bs for bs, val in zip(all_batch_sizes, tetranode_values) if val is not None]
    tetranode_y = [val for val in tetranode_values if val is not None]
    
    # Plot lines
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
    if binode_x or tetranode_x:
        all_x = sorted(set(binode_x + tetranode_x))
        ax.set_xticks(all_x)

# Adjust layout and save
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('multi_nodes_performance_analysis.png', dpi=300, bbox_inches='tight')
plt.show() 