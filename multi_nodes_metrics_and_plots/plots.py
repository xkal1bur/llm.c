import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_metrics_data(filename):
    """Read the multi-node training metrics CSV file."""
    df = pd.read_csv(filename)
    return df

def plot_computation_time_analysis(df):
    """Plot computation time analysis similar to CPU avg_computation_time plot."""
    plt.figure(figsize=(12, 8))
    
    # Create subplots for detailed analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Average times comparison
    avg_total = df['total_time_ms'].mean()
    avg_comp = df['computation_time_ms'].mean() 
    avg_comm = df['communication_time_ms'].mean()
    avg_pure_nccl = df['total_pure_comm_time_ms'].mean()
    
    categories = ['Total Time', 'Computation', 'Communication\n(Total)', 'Pure NCCL\nComm']
    times = [avg_total, avg_comp, avg_comm, avg_pure_nccl]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    bars = ax1.bar(categories, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Average Time (ms)')
    ax1.set_title('Multi-Node Training: Average Time Breakdown per Step')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Time evolution over training steps
    ax2.plot(df['step'], df['total_time_ms'], 'b-', linewidth=2, alpha=0.8, label='Total Time')
    ax2.plot(df['step'], df['computation_time_ms'], 'g-', linewidth=2, alpha=0.8, label='Computation Time')
    ax2.plot(df['step'], df['communication_time_ms'], 'r-', linewidth=2, alpha=0.8, label='Communication Time')
    
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Time Evolution During Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_node_computation_time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_gflops_performance(df):
    """Plot GFLOP/s performance similar to CPU gflops_per_step plot."""
    plt.figure(figsize=(12, 8))
    
    # Main performance plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: GFLOP/s over steps
    ax1.plot(df['step'], df['gflops_per_sec'], 'g-', linewidth=3, alpha=0.8, marker='o', markersize=4)
    avg_gflops = df['gflops_per_sec'].mean()
    ax1.axhline(y=avg_gflops, color='red', linestyle='--', linewidth=2, alpha=0.8,
                label=f'Average: {avg_gflops:.2f} GFLOP/s')
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('GFLOP/s')
    ax1.set_title('Multi-Node Computational Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance efficiency (GFLOP/s vs Communication Overhead)
    comm_overhead = (df['communication_time_ms'] / df['total_time_ms']) * 100
    
    # Create scatter plot with color mapping
    scatter = ax2.scatter(comm_overhead, df['gflops_per_sec'], 
                         c=df['step'], cmap='viridis', alpha=0.7, s=50)
    ax2.set_xlabel('Communication Overhead (%)')
    ax2.set_ylabel('GFLOP/s')
    ax2.set_title('Performance vs Communication Overhead')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for step information
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Training Step')
    
    plt.tight_layout()
    plt.savefig('multi_node_gflops_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_communication_efficiency_analysis(df):
    """Plot communication efficiency analysis (similar to speedup concept for multi-node)."""
    plt.figure(figsize=(12, 10))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Communication Breakdown
    avg_grad_reduce = df['gradient_reduce_time_ms'].mean()
    avg_loss_reduce = df['loss_reduce_time_ms'].mean()
    avg_other_comm = df['communication_time_ms'].mean() - df['total_pure_comm_time_ms'].mean()
    
    comm_types = ['Gradient\nReduction', 'Loss\nReduction', 'Sync\nOverhead']
    comm_times = [avg_grad_reduce, avg_loss_reduce, avg_other_comm]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    wedges, texts, autotexts = ax1.pie(comm_times, labels=comm_types, colors=colors, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 10})
    ax1.set_title('Communication Time Breakdown\n(Average per Step)')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    
    # Plot 2: Communication Efficiency Trend
    comm_efficiency = (df['computation_time_ms'] / df['total_time_ms']) * 100
    ax2.plot(df['step'], comm_efficiency, 'purple', linewidth=2, marker='s', markersize=4, alpha=0.8)
    avg_efficiency = comm_efficiency.mean()
    ax2.axhline(y=avg_efficiency, color='orange', linestyle='--', linewidth=2,
                label=f'Average: {avg_efficiency:.1f}%')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Compute Efficiency (%)')
    ax2.set_title('Computation Efficiency Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Gradient Operations Analysis
    ax3.bar(range(len(df)), df['num_gradient_reductions'], color='brown', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Training Step')
    ax3.set_ylabel('Number of Gradient Reductions')
    ax3.set_title('Gradient Reduction Operations per Step')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Communication vs Computation Ratio
    comm_comp_ratio = df['communication_time_ms'] / df['computation_time_ms']
    ax4.plot(df['step'], comm_comp_ratio, 'red', linewidth=2, marker='^', markersize=4, alpha=0.8)
    avg_ratio = comm_comp_ratio.mean()
    ax4.axhline(y=avg_ratio, color='blue', linestyle='--', linewidth=2,
                label=f'Average Ratio: {avg_ratio:.2f}')
    ax4.set_xlabel('Training Step')
    ax4.set_ylabel('Communication/Computation Ratio')
    ax4.set_title('Communication to Computation Time Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_node_communication_efficiency.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_scaling_analysis(df):
    """Plot scaling analysis for multi-node setup."""
    plt.figure(figsize=(12, 8))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Performance Stability
    performance_stability = df['gflops_per_sec'].std() / df['gflops_per_sec'].mean() * 100
    time_stability = df['total_time_ms'].std() / df['total_time_ms'].mean() * 100
    comm_stability = df['communication_time_ms'].std() / df['communication_time_ms'].mean() * 100
    
    metrics = ['GFLOP/s\nStability', 'Total Time\nStability', 'Communication\nStability']
    stability_scores = [performance_stability, time_stability, comm_stability]
    colors = ['green', 'blue', 'red']
    
    bars = ax1.bar(metrics, stability_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Coefficient of Variation (%)')
    ax1.set_title('Multi-Node Training Stability\n(Lower is Better)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, stability_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Communication Efficiency vs Performance
    # This is like a "scaling efficiency" - how well communication scales with performance
    comm_overhead_pct = (df['communication_time_ms'] / df['total_time_ms']) * 100
    
    ax2.scatter(df['gflops_per_sec'], comm_overhead_pct, alpha=0.7, s=60, c='purple')
    
    # Add trend line
    z = np.polyfit(df['gflops_per_sec'], comm_overhead_pct, 1)
    p = np.poly1d(z)
    ax2.plot(df['gflops_per_sec'], p(df['gflops_per_sec']), "r--", alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('GFLOP/s')
    ax2.set_ylabel('Communication Overhead (%)')
    ax2.set_title('Communication Overhead vs Performance\n(Multi-Node Scaling)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('multi_node_scaling_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_communication_analysis(df):
    """Print detailed communication analysis similar to summary statistics."""
    print("=" * 70)
    print("MULTI-NODE COMMUNICATION ANALYSIS")
    print("=" * 70)
    print(f"Training Configuration: {len(df)} steps analyzed")
    print()
    
    # Performance Metrics
    print("🚀 COMPUTATIONAL PERFORMANCE:")
    print(f"  Average GFLOP/s: {df['gflops_per_sec'].mean():.2f}")
    print(f"  Peak GFLOP/s: {df['gflops_per_sec'].max():.2f}")
    print(f"  Performance stability: {(df['gflops_per_sec'].std()/df['gflops_per_sec'].mean()*100):.1f}% CV")
    print()
    
    # Time Analysis
    print("⏱️  TIMING BREAKDOWN (per step):")
    print(f"  Average total time: {df['total_time_ms'].mean():.1f} ms")
    print(f"  Average computation time: {df['computation_time_ms'].mean():.1f} ms")
    print(f"  Average communication time: {df['communication_time_ms'].mean():.1f} ms")
    compute_pct = (df['computation_time_ms'].mean() / df['total_time_ms'].mean()) * 100
    comm_pct = (df['communication_time_ms'].mean() / df['total_time_ms'].mean()) * 100
    print(f"  Computation efficiency: {compute_pct:.1f}%")
    print(f"  Communication overhead: {comm_pct:.1f}%")
    print()
    
    # Communication Breakdown
    print("📡 COMMUNICATION BREAKDOWN:")
    print(f"  Gradient reduction time: {df['gradient_reduce_time_ms'].mean():.1f} ms")
    print(f"  Loss reduction time: {df['loss_reduce_time_ms'].mean():.1f} ms")
    print(f"  Pure NCCL time: {df['total_pure_comm_time_ms'].mean():.1f} ms")
    sync_overhead = df['communication_time_ms'].mean() - df['total_pure_comm_time_ms'].mean()
    print(f"  Synchronization overhead: {sync_overhead:.1f} ms")
    print(f"  Average gradient reductions per step: {df['num_gradient_reductions'].mean():.1f}")
    print()
    
    # Efficiency Analysis
    print("📊 MULTI-NODE EFFICIENCY:")
    grad_reduce_pct = (df['gradient_reduce_time_ms'].mean() / df['total_pure_comm_time_ms'].mean()) * 100
    loss_reduce_pct = (df['loss_reduce_time_ms'].mean() / df['total_pure_comm_time_ms'].mean()) * 100
    print(f"  Gradient sync dominance: {grad_reduce_pct:.1f}% of pure NCCL time")
    print(f"  Loss sync contribution: {loss_reduce_pct:.1f}% of pure NCCL time")
    comm_comp_ratio = df['communication_time_ms'].mean() / df['computation_time_ms'].mean()
    print(f"  Communication/Computation ratio: {comm_comp_ratio:.2f}")
    print()
    
    # Scaling Assessment
    print("🎯 SCALING ASSESSMENT:")
    if comm_pct < 20:
        scaling_grade = "EXCELLENT"
    elif comm_pct < 35:
        scaling_grade = "GOOD"
    elif comm_pct < 50:
        scaling_grade = "MODERATE"
    else:
        scaling_grade = "POOR"
    print(f"  Multi-node scaling efficiency: {scaling_grade}")
    print(f"  Recommendation: {'Excellent for scaling' if comm_pct < 20 else 'Consider optimization' if comm_pct > 35 else 'Good balance'}")
    print("=" * 70)

def main():
    """Main function to run all multi-node analysis."""
    filename = "b2_t256_d1024.csv"
    print(f"🔍 Analyzing multi-node training metrics from {filename}...")
    
    try:
        df = read_metrics_data(filename)
        print(f"✅ Successfully loaded {len(df)} training steps")
        print()
        
        # Print detailed communication analysis
        print_communication_analysis(df)
        print()
        
        # Generate all plots
        print("📈 Generating multi-node performance plots...")
        
        plot_computation_time_analysis(df)
        print("✅ Computation time analysis saved")
        
        plot_gflops_performance(df)
        print("✅ GFLOP/s performance analysis saved")
        
        plot_communication_efficiency_analysis(df)
        print("✅ Communication efficiency analysis saved")
        
        plot_scaling_analysis(df)
        print("✅ Scaling analysis saved")
        
        print()
        print("🎉 All multi-node analysis plots generated successfully!")
        print("📁 Generated files:")
        print("   📊 multi_node_computation_time_analysis.png")
        print("   🚀 multi_node_gflops_performance.png") 
        print("   📡 multi_node_communication_efficiency.png")
        print("   📈 multi_node_scaling_analysis.png")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {filename}")
        print("   Make sure the CSV file is in the current directory.")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    main()
