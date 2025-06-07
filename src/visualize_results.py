import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from src.utils.metrics import calculate_metrics

def load_metrics(model_name):
    """Load metrics from saved file"""
    metrics_path = Path(f'results/{model_name}/metrics.npy')
    if metrics_path.exists():
        return np.load(metrics_path, allow_pickle=True).item()
    return None

def plot_training_history(metrics, save_path):
    """Plot training history for loss and MAE"""
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['history']['loss'], label='Training Loss')
    plt.plot(metrics['history']['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(metrics['history']['mae'], label='Training MAE')
    plt.plot(metrics['history']['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path / 'training_history.png')
    plt.close()

def plot_metrics_comparison(basic_metrics, improved_metrics, save_path):
    """Plot comparison of key metrics between models"""
    metrics = ['test_loss', 'test_mae']
    models = ['Basic Model', 'Improved Model']
    
    # Prepare data
    data = {
        'test_loss': [basic_metrics['test_loss'], improved_metrics['test_loss']],
        'test_mae': [basic_metrics['test_mae'], improved_metrics['test_mae']]
    }
    
    # Create bar plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, data[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width/2, models)
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path / 'metrics_comparison.png')
    plt.close()

def plot_error_distribution(model, test_dataset, save_path):
    """Plot error distribution for predictions"""
    predictions = model.predict(test_dataset)
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    
    errors = predictions - true_labels
    
    plt.figure(figsize=(12, 5))
    
    # Plot error distribution
    plt.subplot(1, 2, 1)
    sns.histplot(errors[:, 0], kde=True)
    plt.title('X-coordinate Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    sns.histplot(errors[:, 1], kde=True)
    plt.title('Y-coordinate Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path / 'error_distribution.png')
    plt.close()

def create_visualization_dashboard():
    """Create comprehensive visualization dashboard"""
    # Create output directory
    output_dir = Path('results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics
    basic_metrics = load_metrics('basic_localization')
    improved_metrics = load_metrics('improved_localization')
    
    if basic_metrics:
        # Plot training history for basic model
        plot_training_history(basic_metrics, output_dir)
    
    if basic_metrics and improved_metrics:
        # Plot metrics comparison
        plot_metrics_comparison(basic_metrics, improved_metrics, output_dir)
    
    # Create summary report
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Model Performance Summary\n")
        f.write("=======================\n\n")
        
        if basic_metrics:
            f.write("Basic Model:\n")
            f.write(f"Test Loss: {basic_metrics['test_loss']:.4f}\n")
            f.write(f"Test MAE: {basic_metrics['test_mae']:.4f}\n")
            if 'training_time' in basic_metrics:
                f.write(f"Training Time: {basic_metrics['training_time']:.2f} seconds\n")
            f.write("\n")
        
        if improved_metrics:
            f.write("Improved Model:\n")
            f.write(f"Test Loss: {improved_metrics['test_loss']:.4f}\n")
            f.write(f"Test MAE: {improved_metrics['test_mae']:.4f}\n")
            if 'training_time' in improved_metrics:
                f.write(f"Training Time: {improved_metrics['training_time']:.2f} seconds\n")
            f.write("\n")
        
        if basic_metrics and improved_metrics:
            f.write("Comparison:\n")
            f.write(f"Loss Difference: {improved_metrics['test_loss'] - basic_metrics['test_loss']:.4f}\n")
            f.write(f"MAE Difference: {improved_metrics['test_mae'] - basic_metrics['test_mae']:.4f}\n")

def main():
    create_visualization_dashboard()
    print("Visualization dashboard created in results/visualizations/")

if __name__ == '__main__':
    main() 