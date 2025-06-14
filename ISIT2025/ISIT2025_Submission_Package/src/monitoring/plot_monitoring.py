import json
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from src.config.server_config import MONITORING_CONFIG

def load_metrics():
    """Load all metrics files and combine into a DataFrame."""
    metrics_files = glob.glob(str(MONITORING_CONFIG["log_dir"] / "metrics_*.json"))
    all_metrics = []
    
    for file in metrics_files:
        try:
            with open(file, 'r') as f:
                metrics = json.load(f)
                # Extract the latest values from each metric
                latest_metrics = {
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(file)),
                    'memory_usage': metrics['memory_usage'][-1]['memory_usage'] if metrics['memory_usage'] else 0,
                    'available_memory': metrics['memory_usage'][-1]['available_memory'] if metrics['memory_usage'] else 0,
                    'total_memory': metrics['memory_usage'][-1]['total_memory'] if metrics['memory_usage'] else 0,
                    'cpu_usage': metrics['cpu_usage'][-1]['cpu_usage'] if metrics['cpu_usage'] else 0,
                    'cpu_count': metrics['cpu_usage'][-1]['cpu_count'] if metrics['cpu_usage'] else 0,
                    'gpu_usage': metrics['gpu_usage'][-1]['gpu_usage'] if metrics['gpu_usage'] else 0,
                    'gpu_available': metrics['gpu_usage'][-1]['gpu_available'] if metrics['gpu_usage'] else 0,
                    'error_count': len(metrics['errors']),
                    'inference_count': len(metrics['predictions']),
                    'avg_inference_time': sum(m['inference_time'] for m in metrics['inference_times']) / len(metrics['inference_times']) if metrics['inference_times'] else 0,
                    'max_inference_time': max(m['inference_time'] for m in metrics['inference_times']) if metrics['inference_times'] else 0,
                    'min_inference_time': min(m['inference_time'] for m in metrics['inference_times']) if metrics['inference_times'] else 0,
                    'throughput': metrics['throughput'][-1]['inferences_per_second'] if metrics['throughput'] else 0,
                    'model_size_mb': metrics['model_size'] / (1024 * 1024) if 'model_size' in metrics else 0
                }
                all_metrics.append(latest_metrics)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_metrics:
        print("No metrics files found!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_metrics)
    df.set_index('timestamp', inplace=True)
    return df

def plot_metrics(df):
    """Create plots for different metrics."""
    if df is None:
        return
    
    # Create figure with subplots
    fig = plt.figure(figsize=MONITORING_CONFIG["plot_size"])
    gs = fig.add_gridspec(3, 2)
    
    # Plot inference metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df.index, df['inference_count'], 'b-', label='Total Inferences')
    ax1.set_title('Inference Count')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True)
    
    # Plot inference time
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df.index, df['avg_inference_time'], 'r-', label='Average Time')
    ax2.plot(df.index, df['max_inference_time'], 'g--', label='Max Time')
    ax2.plot(df.index, df['min_inference_time'], 'y--', label='Min Time')
    ax2.set_title('Inference Time (ms)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Time (ms)')
    ax2.legend()
    ax2.grid(True)
    
    # Plot memory usage
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(df.index, df['memory_usage'], 'm-', label='Memory Usage %')
    ax3.plot(df.index, df['available_memory'] / df['total_memory'] * 100, 'c--', label='Available Memory %')
    ax3.set_title('Memory Usage')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Percentage')
    ax3.legend()
    ax3.grid(True)
    
    # Plot CPU/GPU usage
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(df.index, df['cpu_usage'], 'c-', label='CPU')
    if df['gpu_available'].any():
        ax4.plot(df.index, df['gpu_usage'], 'y-', label='GPU')
    ax4.set_title('CPU/GPU Usage (%)')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Usage (%)')
    ax4.legend()
    ax4.grid(True)
    
    # Plot throughput
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(df.index, df['throughput'], 'g-', label='Throughput')
    ax5.set_title('Model Throughput')
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Inferences/Second')
    ax5.legend()
    ax5.grid(True)
    
    # Plot error count
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(df.index, df['error_count'], 'r-', label='Errors')
    ax6.set_title('Error Count')
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Count')
    ax6.legend()
    ax6.grid(True)
    
    # Add overall title
    fig.suptitle('Model Server Monitoring Metrics', fontsize=16)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = MONITORING_CONFIG["log_dir"] / "metrics_plot.png"
    plt.savefig(plot_path, dpi=MONITORING_CONFIG["plot_dpi"], bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    
    # Show plot
    plt.show()

def main():
    # Load and plot metrics
    df = load_metrics()
    plot_metrics(df)

if __name__ == "__main__":
    main() 