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
    """Extract batch size from filename."""
    match = re.search(r'_b(\d+)_', filename)
    return int(match.group(1)) if match else None

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

def load_multi_node_data():
    """Load all multi-node CSV files and organize data."""
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
    
    return singlenode_data, binode_data, tetranode_data

def plot_speedup_total_time(singlenode_data, binode_data, tetranode_data):
    """Plot speedup vs number of nodes using total time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlenode data
    effective_batches = sorted(singlenode_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlenode_data:
            nodes = [1]  # Start with singlenode
            speedups = [1.0]  # Baseline speedup is 1
            
            singlenode_time = singlenode_data[effective_batch]['total_time']
            
            # Add binode if available
            if effective_batch in binode_data:
                nodes.append(2)
                binode_time = binode_data[effective_batch]['total_time']
                speedup = singlenode_time / binode_time
                speedups.append(speedup)
            
            # Add tetranode if available
            if effective_batch in tetranode_data:
                nodes.append(4)
                tetranode_time = tetranode_data[effective_batch]['total_time']
                speedup = singlenode_time / tetranode_time
                speedups.append(speedup)
            
            plt.plot(nodes, speedups, marker='o', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal speedup line
    max_nodes = 4
    ideal_nodes = [1, 2, 4]
    plt.plot(ideal_nodes, ideal_nodes, 'k--', alpha=0.5, label='Ideal Speedup', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    plt.ylabel('Speedup (Total Time)', fontweight='bold', fontsize=12)
    plt.title('Speedup vs Number of Nodes (Total Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4])
    plt.tight_layout()
    plt.savefig('speedup_total_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_speedup_computation_time(singlenode_data, binode_data, tetranode_data):
    """Plot speedup vs number of nodes using computation time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlenode data
    effective_batches = sorted(singlenode_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlenode_data:
            nodes = [1]  # Start with singlenode
            speedups = [1.0]  # Baseline speedup is 1
            
            singlenode_time = singlenode_data[effective_batch]['computation_time']
            
            # Add binode if available
            if effective_batch in binode_data:
                nodes.append(2)
                binode_time = binode_data[effective_batch]['computation_time']
                speedup = singlenode_time / binode_time
                speedups.append(speedup)
            
            # Add tetranode if available
            if effective_batch in tetranode_data:
                nodes.append(4)
                tetranode_time = tetranode_data[effective_batch]['computation_time']
                speedup = singlenode_time / tetranode_time
                speedups.append(speedup)
            
            plt.plot(nodes, speedups, marker='s', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal speedup line
    ideal_nodes = [1, 2, 4]
    plt.plot(ideal_nodes, ideal_nodes, 'k--', alpha=0.5, label='Ideal Speedup', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    plt.ylabel('Speedup (Computation Time)', fontweight='bold', fontsize=12)
    plt.title('Speedup vs Number of Nodes (Computation Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4])
    plt.tight_layout()
    plt.savefig('speedup_computation_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_total_time(singlenode_data, binode_data, tetranode_data):
    """Plot efficiency vs number of nodes using total time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlenode data
    effective_batches = sorted(singlenode_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlenode_data:
            nodes = [1]  # Start with singlenode
            efficiencies = [1.0]  # Baseline efficiency is 1
            
            singlenode_time = singlenode_data[effective_batch]['total_time']
            
            # Add binode if available
            if effective_batch in binode_data:
                nodes.append(2)
                binode_time = binode_data[effective_batch]['total_time']
                speedup = singlenode_time / binode_time
                efficiency = speedup / 2  # 2 nodes
                efficiencies.append(efficiency)
            
            # Add tetranode if available
            if effective_batch in tetranode_data:
                nodes.append(4)
                tetranode_time = tetranode_data[effective_batch]['total_time']
                speedup = singlenode_time / tetranode_time
                efficiency = speedup / 4  # 4 nodes
                efficiencies.append(efficiency)
            
            plt.plot(nodes, efficiencies, marker='^', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal efficiency line (should be 1.0)
    ideal_nodes = [1, 2, 4]
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    plt.ylabel('Efficiency (Speedup / Nodes)', fontweight='bold', fontsize=12)
    plt.title('Efficiency vs Number of Nodes (Total Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4])
    plt.ylim(0, 1.2)  # Set y-axis limit for better visualization
    plt.tight_layout()
    plt.savefig('efficiency_total_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_efficiency_computation_time(singlenode_data, binode_data, tetranode_data):
    """Plot efficiency vs number of nodes using computation time."""
    plt.figure(figsize=(12, 8))
    
    # Get all effective batch sizes that exist in singlenode data
    effective_batches = sorted(singlenode_data.keys())
    
    for effective_batch in effective_batches:
        if effective_batch in singlenode_data:
            nodes = [1]  # Start with singlenode
            efficiencies = [1.0]  # Baseline efficiency is 1
            
            singlenode_time = singlenode_data[effective_batch]['computation_time']
            
            # Add binode if available
            if effective_batch in binode_data:
                nodes.append(2)
                binode_time = binode_data[effective_batch]['computation_time']
                speedup = singlenode_time / binode_time
                efficiency = speedup / 2  # 2 nodes
                efficiencies.append(efficiency)
            
            # Add tetranode if available
            if effective_batch in tetranode_data:
                nodes.append(4)
                tetranode_time = tetranode_data[effective_batch]['computation_time']
                speedup = singlenode_time / tetranode_time
                efficiency = speedup / 4  # 4 nodes
                efficiencies.append(efficiency)
            
            plt.plot(nodes, efficiencies, marker='D', label=f'Effective Batch {effective_batch}', 
                    linewidth=2.5, markersize=8)
    
    # Add ideal efficiency line (should be 1.0)
    ideal_nodes = [1, 2, 4]
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal Efficiency (1.0)', linewidth=2)
    
    plt.xlabel('Number of Nodes', fontweight='bold', fontsize=12)
    plt.ylabel('Efficiency (Speedup / Nodes)', fontweight='bold', fontsize=12)
    plt.title('Efficiency vs Number of Nodes (Computation Time)', fontweight='bold', fontsize=14)
    plt.legend(frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks([1, 2, 4])
    plt.ylim(0, 1.2)  # Set y-axis limit for better visualization
    plt.tight_layout()
    plt.savefig('efficiency_computation_time.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to load data and generate all speedup and efficiency plots."""
    print("Loading multi-node data from CSV files...")
    singlenode_data, binode_data, tetranode_data = load_multi_node_data()
    
    # Check if data was loaded successfully
    if not singlenode_data:
        print("No singlenode data found. Please check if the CSV files exist.")
        return
    
    print("Generating speedup and efficiency plots...")

    # --- DEBUGGING ---
    print("Single-node data:", singlenode_data)
    print("Bi-node data:", binode_data)
    print("Tetra-node data:", tetranode_data)
    
    # Generate all four plots
    plot_speedup_total_time(singlenode_data, binode_data, tetranode_data)
    plot_speedup_computation_time(singlenode_data, binode_data, tetranode_data)
    plot_efficiency_total_time(singlenode_data, binode_data, tetranode_data)
    plot_efficiency_computation_time(singlenode_data, binode_data, tetranode_data)
    
    print("All speedup and efficiency plots generated successfully!")

if __name__ == "__main__":
    main() 