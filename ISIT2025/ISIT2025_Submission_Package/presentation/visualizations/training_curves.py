import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')  # Using a specific seaborn style version
sns.set_theme()  # Set seaborn theme

# Create output directory
output_dir = Path(__file__).parent / 'output'
output_dir.mkdir(exist_ok=True)

# Simulated training data (replace with actual data)
epochs = np.arange(1, 101)
tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4']

# Generate synthetic training curves
np.random.seed(42)

def generate_learning_curves():
    """Generate synthetic learning curves for each task."""
    curves = {}
    for task in tasks:
        # Generate base curves
        train_loss = 0.8 * np.exp(-0.05 * epochs) + 0.2 + 0.1 * np.random.randn(len(epochs))
        val_loss = 0.9 * np.exp(-0.04 * epochs) + 0.25 + 0.15 * np.random.randn(len(epochs))
        
        # Add task-specific characteristics
        if task == 'Task 2':
            train_loss *= 0.8  # Better convergence
            val_loss *= 0.85
        elif task == 'Task 3':
            train_loss *= 0.9
            val_loss *= 0.95
        elif task == 'Task 4':
            train_loss *= 0.85
            val_loss *= 0.9
        
        curves[task] = {
            'train_loss': train_loss,
            'val_loss': val_loss
        }
    return curves

def generate_metric_curves():
    """Generate synthetic metric curves for each task."""
    metrics = {}
    for task in tasks:
        # Generate base curves
        mae = 0.6 * np.exp(-0.03 * epochs) + 0.4 + 0.05 * np.random.randn(len(epochs))
        r90 = 1.2 * np.exp(-0.025 * epochs) + 0.8 + 0.1 * np.random.randn(len(epochs))
        combined = 0.7 * mae + 0.3 * r90
        
        # Add task-specific characteristics
        if task == 'Task 2':
            mae *= 0.8
            r90 *= 0.85
        elif task == 'Task 3':
            mae *= 0.9
            r90 *= 0.95
        elif task == 'Task 4':
            mae *= 0.85
            r90 *= 0.9
        
        metrics[task] = {
            'mae': mae,
            'r90': r90,
            'combined': 0.7 * mae + 0.3 * r90
        }
    return metrics

def plot_training_curves(curves):
    """Plot training and validation loss curves for each task."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        train_loss = curves[task]['train_loss']
        val_loss = curves[task]['val_loss']
        
        ax.plot(epochs, train_loss, label='Training Loss', color='#2ecc71')
        ax.plot(epochs, val_loss, label='Validation Loss', color='#e74c3c')
        
        ax.set_title(f'{task} Training Curves')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)
        
        # Add final values
        ax.text(0.02, 0.98, f'Final Train Loss: {train_loss[-1]:.3f}',
                transform=ax.transAxes, va='top')
        ax.text(0.02, 0.93, f'Final Val Loss: {val_loss[-1]:.3f}',
                transform=ax.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_curves(metrics):
    """Plot metric progression curves for each task."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        mae = metrics[task]['mae']
        r90 = metrics[task]['r90']
        combined = metrics[task]['combined']
        
        ax.plot(epochs, mae, label='MAE', color='#2ecc71')
        ax.plot(epochs, r90, label='R90', color='#e74c3c')
        ax.plot(epochs, combined, label='Combined', color='#3498db')
        
        ax.set_title(f'{task} Metric Progression')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error (meters)')
        ax.legend()
        ax.grid(True)
        
        # Add final values
        ax.text(0.02, 0.98, f'Final MAE: {mae[-1]:.3f}m',
                transform=ax.transAxes, va='top')
        ax.text(0.02, 0.93, f'Final R90: {r90[-1]:.3f}m',
                transform=ax.transAxes, va='top')
        ax.text(0.02, 0.88, f'Final Combined: {combined[-1]:.3f}',
                transform=ax.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate_schedule():
    """Plot learning rate scheduling visualization."""
    # Generate learning rate schedule
    initial_lr = 0.001
    epochs = np.arange(1, 101)
    
    # Simulate different scheduling strategies
    constant = np.ones_like(epochs) * initial_lr
    step_decay = initial_lr * np.power(0.1, np.floor(epochs / 30))
    exponential = initial_lr * np.exp(-0.01 * epochs)
    cosine = initial_lr * (1 + np.cos(np.pi * epochs / 100)) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, constant, label='Constant', color='#2ecc71')
    plt.plot(epochs, step_decay, label='Step Decay', color='#e74c3c')
    plt.plot(epochs, exponential, label='Exponential', color='#3498db')
    plt.plot(epochs, cosine, label='Cosine', color='#9b59b6')
    
    plt.title('Learning Rate Scheduling Strategies')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_rate_schedule.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating training visualization plots...")
    
    # Generate and plot training curves
    curves = generate_learning_curves()
    plot_training_curves(curves)
    
    # Generate and plot metric curves
    metrics = generate_metric_curves()
    plot_metric_curves(metrics)
    
    # Plot learning rate schedule
    plot_learning_rate_schedule()
    
    print(f"Plots saved to {output_dir}") 