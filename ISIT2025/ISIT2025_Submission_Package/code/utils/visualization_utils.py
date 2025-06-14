import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class VisualizationUtils:
    """Utility class for creating various visualizations."""
    
    def __init__(self, output_dir: str = 'results/visualizations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set default style
        plt.style.use('seaborn')
        sns.set_palette('husl')
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            metrics: List[str] = None,
                            save_path: Optional[Union[str, Path]] = None):
        """Plot training history metrics."""
        if metrics is None:
            metrics = list(history.keys())
        
        plt.figure(figsize=(12, 6))
        
        for metric in metrics:
            if metric in history:
                plt.plot(history[metric], label=metric)
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_distribution(self, errors: np.ndarray,
                              save_path: Optional[Union[str, Path]] = None):
        """Plot error distribution histogram."""
        plt.figure(figsize=(10, 6))
        
        sns.histplot(errors, kde=True)
        plt.axvline(np.mean(errors), color='r', linestyle='--',
                   label=f'Mean: {np.mean(errors):.2f}')
        plt.axvline(np.median(errors), color='g', linestyle='--',
                   label=f'Median: {np.median(errors):.2f}')
        
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_spatial_errors(self, predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          errors: np.ndarray,
                          save_path: Optional[Union[str, Path]] = None):
        """Plot spatial distribution of errors."""
        plt.figure(figsize=(12, 8))
        
        scatter = plt.scatter(ground_truth[:, 0], ground_truth[:, 1],
                            c=errors, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Error')
        
        # Plot predictions with arrows
        for pred, gt in zip(predictions, ground_truth):
            plt.arrow(gt[0], gt[1],
                     pred[0] - gt[0], pred[1] - gt[1],
                     color='red', alpha=0.3,
                     head_width=0.1, head_length=0.1)
        
        plt.title('Spatial Error Distribution')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_channel_quality(self, channel_quality: np.ndarray,
                           timestamps: Optional[np.ndarray] = None,
                           save_path: Optional[Union[str, Path]] = None):
        """Plot channel quality over time."""
        plt.figure(figsize=(12, 6))
        
        if timestamps is not None:
            plt.plot(timestamps, channel_quality)
            plt.xlabel('Time')
        else:
            plt.plot(channel_quality)
            plt.xlabel('Sample')
        
        plt.title('Channel Quality Over Time')
        plt.ylabel('Quality')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_vs_quality(self, errors: np.ndarray,
                            channel_quality: np.ndarray,
                            save_path: Optional[Union[str, Path]] = None):
        """Plot error vs channel quality correlation."""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(channel_quality, errors, alpha=0.5)
        
        # Add trend line
        z = np.polyfit(channel_quality, errors, 1)
        p = np.poly1d(z)
        plt.plot(channel_quality, p(channel_quality), "r--", alpha=0.8)
        
        plt.title('Error vs Channel Quality')
        plt.xlabel('Channel Quality')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            labels: Optional[List[str]] = None,
                            save_path: Optional[Union[str, Path]] = None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importance(self, feature_importance: np.ndarray,
                              feature_names: List[str],
                              save_path: Optional[Union[str, Path]] = None):
        """Plot feature importance."""
        plt.figure(figsize=(12, 6))
        
        # Sort features by importance
        indices = np.argsort(feature_importance)
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_interactive_plot(self, data: Dict[str, np.ndarray],
                              plot_type: str = 'scatter',
                              save_path: Optional[Union[str, Path]] = None):
        """Create an interactive plot using Plotly."""
        if plot_type == 'scatter':
            fig = px.scatter(data)
        elif plot_type == 'line':
            fig = px.line(data)
        elif plot_type == 'bar':
            fig = px.bar(data)
        elif plot_type == 'heatmap':
            fig = px.imshow(data)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
        else:
            fig.show()
    
    def create_subplots(self, data: Dict[str, np.ndarray],
                       plot_types: List[str],
                       rows: int, cols: int,
                       save_path: Optional[Union[str, Path]] = None):
        """Create subplots with different plot types."""
        fig = make_subplots(rows=rows, cols=cols)
        
        for i, (name, plot_type) in enumerate(zip(data.keys(), plot_types)):
            row = i // cols + 1
            col = i % cols + 1
            
            if plot_type == 'scatter':
                fig.add_trace(go.Scatter(y=data[name], name=name),
                            row=row, col=col)
            elif plot_type == 'line':
                fig.add_trace(go.Scatter(y=data[name], name=name, mode='lines'),
                            row=row, col=col)
            elif plot_type == 'bar':
                fig.add_trace(go.Bar(y=data[name], name=name),
                            row=row, col=col)
            elif plot_type == 'heatmap':
                fig.add_trace(go.Heatmap(z=data[name], name=name),
                            row=row, col=col)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
        
        fig.update_layout(height=300*rows, width=400*cols,
                         title_text="Multiple Subplots")
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(save_path))
        else:
            fig.show()
    
    def plot_trajectory(self, trajectory: np.ndarray,
                       predictions: Optional[np.ndarray] = None,
                       save_path: Optional[Union[str, Path]] = None):
        """Plot trajectory with optional predictions."""
        plt.figure(figsize=(12, 8))
        
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Ground Truth')
        
        if predictions is not None:
            plt.plot(predictions[:, 0], predictions[:, 1], 'r--', label='Predictions')
        
        plt.title('Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_error_heatmap(self, errors: np.ndarray,
                          grid_size: Tuple[int, int],
                          save_path: Optional[Union[str, Path]] = None):
        """Plot error heatmap on a grid."""
        plt.figure(figsize=(12, 8))
        
        error_grid = errors.reshape(grid_size)
        sns.heatmap(error_grid, cmap='viridis', annot=True, fmt='.2f')
        
        plt.title('Error Heatmap')
        plt.xlabel('X Grid')
        plt.ylabel('Y Grid')
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_learning_curves(self, train_scores: np.ndarray,
                           val_scores: np.ndarray,
                           save_path: Optional[Union[str, Path]] = None):
        """Plot learning curves."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(train_scores, label='Training')
        plt.plot(val_scores, label='Validation')
        
        plt.title('Learning Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 