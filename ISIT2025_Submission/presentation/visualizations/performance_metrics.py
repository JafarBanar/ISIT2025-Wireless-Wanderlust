import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Create output directory
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

# Performance data
tasks = ['Task 1\n(Vanilla)', 'Task 2\n(Trajectory)', 'Task 3\n(Grant-Free)', 'Task 4\n(Joint)']
metrics = {
    'MAE (m)': [0.42, 0.35, 0.38, 0.36],
    'R90 (m)': [0.85, 0.72, 0.78, 0.74],
    'Combined': [0.56, 0.47, 0.51, 0.49],
    'Transmission Rate (%)': [100, 100, 40, 35],
    'Inference Time (ms)': [5, 8, 7, 9],
    'Model Size (MB)': [12, 15, 18, 20]
}

# Create DataFrame
df = pd.DataFrame(metrics, index=tasks)

def plot_metrics_comparison():
    """Plot bar chart comparing MAE, R90, and Combined metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tasks))
    width = 0.25
    
    ax.bar(x - width, df['MAE (m)'], width, label='MAE', color='#2ecc71')
    ax.bar(x, df['R90 (m)'], width, label='R90', color='#e74c3c')
    ax.bar(x + width, df['Combined'], width, label='Combined', color='#3498db')
    
    ax.set_ylabel('Error (meters)')
    ax.set_title('Performance Metrics Across Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    
    # Add value labels
    for i, v in enumerate(df['MAE (m)']):
        ax.text(i - width, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(df['R90 (m)']):
        ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(df['Combined']):
        ax.text(i + width, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart():
    """Plot radar chart showing performance trade-offs."""
    # Normalize metrics for radar chart
    normalized = df.copy()
    for col in ['MAE (m)', 'R90 (m)', 'Combined']:
        normalized[col] = 1 - (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    for col in ['Transmission Rate (%)', 'Inference Time (ms)', 'Model Size (MB)']:
        normalized[col] = (df[col].max() - df[col]) / (df[col].max() - df[col].min())
    
    # Number of variables
    categories = list(normalized.columns)
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"], color="grey", size=8)
    plt.ylim(0, 1)
    
    # Plot each task
    for i, task in enumerate(tasks):
        values = normalized.loc[task].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=task)
        ax.fill(angles, values, alpha=0.1)
    
    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
    plt.title('Performance Trade-offs Across Tasks')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_bandwidth_vs_accuracy():
    """Plot bandwidth reduction vs. accuracy trade-off."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot transmission rate
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Transmission Rate (%)', color='#e74c3c')
    line1 = ax1.plot(tasks, df['Transmission Rate (%)'], 'o-', color='#e74c3c', label='Transmission Rate')
    ax1.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Create second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('MAE (m)', color='#2ecc71')
    line2 = ax2.plot(tasks, df['MAE (m)'], 's-', color='#2ecc71', label='MAE')
    ax2.tick_params(axis='y', labelcolor='#2ecc71')
    
    # Add value labels
    for i, v in enumerate(df['Transmission Rate (%)']):
        ax1.text(i, v, f'{v}%', ha='center', va='bottom', color='#e74c3c')
    for i, v in enumerate(df['MAE (m)']):
        ax2.text(i, v, f'{v:.2f}m', ha='center', va='top', color='#2ecc71')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Bandwidth Reduction vs. Accuracy Trade-off')
    plt.tight_layout()
    plt.savefig(output_dir / 'bandwidth_vs_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_resource_usage():
    """Plot model size and inference time comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Model size
    ax1.bar(tasks, df['Model Size (MB)'], color='#3498db')
    ax1.set_ylabel('Model Size (MB)')
    ax1.set_title('Model Size Comparison')
    for i, v in enumerate(df['Model Size (MB)']):
        ax1.text(i, v, f'{v}MB', ha='center', va='bottom')
    
    # Inference time
    ax2.bar(tasks, df['Inference Time (ms)'], color='#9b59b6')
    ax2.set_ylabel('Inference Time (ms)')
    ax2.set_title('Inference Time Comparison')
    for i, v in enumerate(df['Inference Time (ms)']):
        ax2.text(i, v, f'{v}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating performance visualization plots...")
    plot_metrics_comparison()
    plot_radar_chart()
    plot_bandwidth_vs_accuracy()
    plot_resource_usage()
    print(f"Plots saved to {output_dir}") 