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

# Simulated data (replace with actual data)
np.random.seed(42)

def generate_position_data(n_samples=1000):
    """Generate synthetic position data for visualization."""
    # Generate true positions in a grid-like pattern
    grid_size = int(np.sqrt(n_samples))
    x = np.linspace(0, 10, grid_size)
    y = np.linspace(0, 10, grid_size)
    X, Y = np.meshgrid(x, y)
    true_positions = np.column_stack((X.flatten(), Y.flatten()))
    
    # Ensure we have exactly n_samples points
    if len(true_positions) > n_samples:
        indices = np.random.choice(len(true_positions), n_samples, replace=False)
        true_positions = true_positions[indices]
    elif len(true_positions) < n_samples:
        # Pad with random points if we have fewer than n_samples
        extra_points = np.random.uniform(0, 10, (n_samples - len(true_positions), 2))
        true_positions = np.vstack((true_positions, extra_points))
    
    # Generate predicted positions with task-specific error patterns
    predictions = {}
    for task in ['Task 1', 'Task 2', 'Task 3', 'Task 4']:
        if task == 'Task 1':
            # Random errors
            error = np.random.normal(0, 0.5, (n_samples, 2))
        elif task == 'Task 2':
            # Correlated errors (better in trajectory)
            error = np.random.normal(0, 0.3, (n_samples, 2))
            # Reshape the sine term to broadcast correctly
            sine_term = np.sin(true_positions[:, 0] * np.pi / 5).reshape(-1, 1)
            error = error * (1 + 0.5 * sine_term)
        elif task == 'Task 3':
            # Spatial pattern errors
            error = np.random.normal(0, 0.4, (n_samples, 2))
            # Reshape the exponential term to broadcast correctly
            exp_term = np.exp(-0.1 * np.sum(true_positions**2, axis=1)).reshape(-1, 1)
            error *= exp_term
        else:  # Task 4
            # Mixed pattern
            error = np.random.normal(0, 0.35, (n_samples, 2))
            # Reshape the sine term to broadcast correctly
            sine_term = np.sin(true_positions[:, 0] * np.pi / 5).reshape(-1, 1)
            error *= (1 + 0.3 * sine_term)
        
        predictions[task] = true_positions + error
    
    return true_positions, predictions

def plot_position_scatter(true_pos, predictions):
    """Plot scatter plots of true vs. predicted positions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (task, pred_pos) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Plot true positions
        ax.scatter(true_pos[:, 0], true_pos[:, 1], 
                  c='#2ecc71', alpha=0.3, label='True Positions')
        
        # Plot predictions
        ax.scatter(pred_pos[:, 0], pred_pos[:, 1], 
                  c='#e74c3c', alpha=0.3, label='Predictions')
        
        # Plot error lines
        for i in range(0, len(true_pos), 10):  # Plot every 10th point
            ax.plot([true_pos[i, 0], pred_pos[i, 0]], 
                   [true_pos[i, 1], pred_pos[i, 1]], 
                   'k-', alpha=0.1)
        
        ax.set_title(f'{task} Position Predictions')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        
        # Calculate and display error statistics
        errors = np.linalg.norm(pred_pos - true_pos, axis=1)
        mae = np.mean(errors)
        r90 = np.percentile(errors, 90)
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}m\nR90: {r90:.3f}m',
                transform=ax.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'position_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_heatmap(true_pos, predictions):
    """Plot error heatmaps for each task."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (task, pred_pos) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Calculate errors
        errors = np.linalg.norm(pred_pos - true_pos, axis=1)
        
        # Create 2D histogram of errors
        h, xedges, yedges, im = ax.hist2d(true_pos[:, 0], true_pos[:, 1], 
                                         weights=errors, bins=20, cmap='viridis')
        
        ax.set_title(f'{task} Error Heatmap')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        plt.colorbar(im, ax=ax, label='Average Error (m)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectory_comparison():
    """Plot trajectory comparison for a sample path."""
    # Generate a sample trajectory
    t = np.linspace(0, 10, 100)
    true_traj = np.column_stack((
        5 + 2 * np.sin(t),  # X coordinate
        5 + 2 * np.cos(t)   # Y coordinate
    ))
    
    # Generate predicted trajectories with different error patterns
    predictions = {}
    for task in ['Task 1', 'Task 2', 'Task 3', 'Task 4']:
        if task == 'Task 1':
            error = np.random.normal(0, 0.5, true_traj.shape)
        elif task == 'Task 2':
            error = np.random.normal(0, 0.3, true_traj.shape)
            # Fix broadcasting: reshape sine term
            sine_term = np.sin(t).reshape(-1, 1)
            error *= (1 + 0.2 * sine_term)
        elif task == 'Task 3':
            error = np.random.normal(0, 0.4, true_traj.shape)
            exp_term = np.exp(-0.1 * np.sum(true_traj**2, axis=1)).reshape(-1, 1)
            error *= exp_term
        else:  # Task 4
            error = np.random.normal(0, 0.35, true_traj.shape)
            sine_term = np.sin(t).reshape(-1, 1)
            error *= (1 + 0.15 * sine_term)
        
        predictions[task] = true_traj + error
    
    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (task, pred_traj) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Plot true trajectory
        ax.plot(true_traj[:, 0], true_traj[:, 1], 
                'b-', label='True Trajectory', alpha=0.7)
        
        # Plot predicted trajectory
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 
                'r-', label='Predicted Trajectory', alpha=0.7)
        
        # Plot error regions
        ax.fill_between(true_traj[:, 0], 
                       true_traj[:, 1] - 0.5, 
                       true_traj[:, 1] + 0.5, 
                       alpha=0.1, color='b')
        
        ax.set_title(f'{task} Trajectory Comparison')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True)
        
        # Calculate and display error statistics
        errors = np.linalg.norm(pred_traj - true_traj, axis=1)
        mae = np.mean(errors)
        r90 = np.percentile(errors, 90)
        ax.text(0.02, 0.98, f'MAE: {mae:.3f}m\nR90: {r90:.3f}m',
                transform=ax.transAxes, va='top')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_distribution(true_pos, predictions):
    """Plot error distribution histograms for each task."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, (task, pred_pos) in enumerate(predictions.items()):
        ax = axes[idx]
        
        # Calculate errors
        errors = np.linalg.norm(pred_pos - true_pos, axis=1)
        
        # Plot histogram
        sns.histplot(errors, bins=30, ax=ax, color='#3498db')
        
        # Add vertical lines for MAE and R90
        mae = np.mean(errors)
        r90 = np.percentile(errors, 90)
        ax.axvline(mae, color='#e74c3c', linestyle='--', label=f'MAE: {mae:.3f}m')
        ax.axvline(r90, color='#2ecc71', linestyle='--', label=f'R90: {r90:.3f}m')
        
        ax.set_title(f'{task} Error Distribution')
        ax.set_xlabel('Error (m)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating error analysis plots...")
    
    # Generate position data
    true_positions, predictions = generate_position_data()
    
    # Generate plots
    plot_position_scatter(true_positions, predictions)
    plot_error_heatmap(true_positions, predictions)
    plot_trajectory_comparison()
    plot_error_distribution(true_positions, predictions)
    
    print(f"Plots saved to {output_dir}") 