import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('default')  # Use default style instead of seaborn
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def plot_training_history(history, output_dir: str, metrics: Optional[List[str]] = None):
    """Plot training history metrics."""
    setup_plotting_style()
    
    # Convert history to dict if it's a History object
    history_dict = history.history if hasattr(history, 'history') else history
    
    # Get metrics to plot
    if metrics is None:
        metrics = [m for m in history_dict.keys() if not m.startswith('val_')]
    
    # Create figure with subplots
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    # Plot each metric
    for ax, metric in zip(axes, metrics):
        ax.plot(history_dict[metric], label='train')
        if f'val_{metric}' in history_dict:
            ax.plot(history_dict[f'val_{metric}'], label='validation')
        ax.set_title(f'{metric} over epochs')
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_prediction_scatter(true_positions: np.ndarray,
                          pred_positions: np.ndarray,
                          save_path: Optional[str] = None):
    """
    Create scatter plot comparing true and predicted positions.
    
    Args:
        true_positions: Ground truth positions (N, 2)
        pred_positions: Predicted positions (N, 2)
        save_path: Path to save the plot
        
    Raises:
        ValueError: If input arrays have invalid shapes or mismatched dimensions
    """
    if true_positions.shape != pred_positions.shape:
        raise ValueError("True and predicted positions must have the same shape")
    
    if len(true_positions.shape) != 2 or true_positions.shape[1] != 2:
        raise ValueError("Positions must be 2D arrays with shape (N, 2)")
    
    setup_plotting_style()
    
    plt.figure(figsize=(12, 12))
    plt.scatter(true_positions[:, 0], true_positions[:, 1], 
               alpha=0.5, label='True Positions')
    plt.scatter(pred_positions[:, 0], pred_positions[:, 1], 
               alpha=0.5, label='Predicted Positions')
    
    for i in range(len(true_positions)):
        plt.plot([true_positions[i, 0], pred_positions[i, 0]],
                [true_positions[i, 1], pred_positions[i, 1]],
                'k-', alpha=0.1)
    
    plt.title('True vs Predicted Positions')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_predictions(true_positions: np.ndarray,
                    pred_positions: np.ndarray,
                    output_dir: str):
    """
    Generate prediction visualization plots.
    
    Args:
        true_positions: Ground truth positions (N, 2)
        pred_positions: Predicted positions (N, 2)
        output_dir: Directory to save plots
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot scatter comparison
    plot_prediction_scatter(
        true_positions,
        pred_positions,
        save_path=str(Path(output_dir) / 'prediction_scatter.png')
    )
    
    # Plot trajectory comparison if we have sequential data
    if len(true_positions) > 1:
        timestamps = np.arange(len(true_positions))
        plot_trajectory(
            true_positions,
            timestamps,
            save_path=str(Path(output_dir) / 'trajectory.png')
        )
        
        # Plot predicted trajectory
        plot_trajectory(
            pred_positions,
            timestamps,
            save_path=str(Path(output_dir) / 'predicted_trajectory.png')
        )

def plot_error_distribution(true_positions: np.ndarray,
                          pred_positions: np.ndarray,
                          output_dir: str):
    """
    Plot distribution of prediction errors.
    
    Args:
        true_positions: Ground truth positions (N, 2)
        pred_positions: Predicted positions (N, 2)
        output_dir: Directory to save plots
    """
    # Calculate errors
    errors = np.sqrt(np.sum((true_positions - pred_positions) ** 2, axis=1))
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot error distribution
    plot_error_distribution_base(
        errors,
        save_path=str(Path(output_dir) / 'error_distribution.png')
    )

def plot_error_distribution_base(errors: np.ndarray,
                               save_path: Optional[str] = None,
                               bins: int = 50):
    """
    Plot distribution of prediction errors.
    
    Args:
        errors: Array of prediction errors
        save_path: Path to save the plot
        bins: Number of histogram bins
        
    Raises:
        ValueError: If errors array is empty or contains invalid values
    """
    if len(errors) == 0:
        raise ValueError("Errors array cannot be empty")
    
    if not np.isfinite(errors).all():
        raise ValueError("Errors array contains invalid values (inf or nan)")
    
    setup_plotting_style()
    
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, bins=bins, kde=True)
    plt.axvline(np.mean(errors), color='r', linestyle='--', 
                label=f'Mean Error: {np.mean(errors):.4f}')
    plt.axvline(np.percentile(errors, 90), color='g', linestyle='--',
                label=f'90th Percentile (R90): {np.percentile(errors, 90):.4f}')
    
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Error Distance')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(feature_scores: np.ndarray,
                          feature_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          top_k: Optional[int] = None):
    """
    Plot feature importance scores.
    
    Args:
        feature_scores: Array of feature importance scores
        feature_names: List of feature names
        save_path: Path to save the plot
        top_k: Number of top features to show
        
    Raises:
        ValueError: If inputs have invalid dimensions or values
    """
    if len(feature_scores) == 0:
        raise ValueError("Feature scores array cannot be empty")
    
    if feature_names is not None and len(feature_names) != len(feature_scores):
        raise ValueError("Number of feature names must match number of scores")
    
    if top_k is not None and (top_k <= 0 or top_k > len(feature_scores)):
        raise ValueError("top_k must be positive and not larger than number of features")
    
    setup_plotting_style()
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(feature_scores))]
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_scores)
    if top_k:
        sorted_idx = sorted_idx[-top_k:]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(sorted_idx)), feature_scores[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title('Feature Importance Scores')
    plt.xlabel('Importance Score')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_trajectory(positions: np.ndarray,
                   timestamps: np.ndarray,
                   save_path: Optional[str] = None):
    """
    Plot trajectory in 2D with time-based coloring.
    
    Args:
        positions: Array of positions (N, 2)
        timestamps: Array of timestamps (N,)
        save_path: Path to save the plot
        
    Raises:
        ValueError: If inputs have invalid dimensions or values
    """
    if len(positions.shape) != 2 or positions.shape[1] != 2:
        raise ValueError("Positions must be a 2D array with shape (N, 2)")
    
    if len(timestamps) != positions.shape[0]:
        raise ValueError("Number of timestamps must match number of positions")
    
    if not np.all(np.diff(timestamps) >= 0):
        raise ValueError("Timestamps must be monotonically increasing")
    
    setup_plotting_style()
    
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(positions[:, 0], positions[:, 1],
                         c=timestamps, cmap='viridis')
    plt.colorbar(scatter, label='Time')
    
    # Plot lines connecting consecutive points
    plt.plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3)
    
    plt.title('Trajectory Over Time')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def create_performance_report(model_name: str,
                            metrics: Dict[str, float],
                            history: Dict[str, List[float]],
                            true_positions: np.ndarray,
                            pred_positions: np.ndarray,
                            output_dir: str):
    """
    Create comprehensive performance report with plots.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of evaluation metrics
        history: Training history
        true_positions: Ground truth positions
        pred_positions: Predicted positions
        output_dir: Directory to save report files
        
    Raises:
        ValueError: If any inputs are invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise ValueError("Model name must be a non-empty string")
    
    if not metrics:
        raise ValueError("Metrics dictionary cannot be empty")
    
    if not history:
        raise ValueError("History dictionary cannot be empty")
    
    if true_positions.shape != pred_positions.shape:
        raise ValueError("True and predicted positions must have the same shape")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training history
    plot_training_history(
        history,
        output_dir=str(output_dir)
    )
    
    # Plot predictions
    plot_predictions(
        true_positions,
        pred_positions,
        output_dir=str(output_dir)
    )
    
    # Plot error distribution
    plot_error_distribution(
        true_positions,
        pred_positions,
        output_dir=str(output_dir)
    )
    
    # Save metrics summary
    with open(output_dir / f'{model_name}_metrics_summary.txt', 'w') as f:
        f.write(f"Performance Metrics for {model_name}\n")
        f.write("=" * 50 + "\n\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n") 