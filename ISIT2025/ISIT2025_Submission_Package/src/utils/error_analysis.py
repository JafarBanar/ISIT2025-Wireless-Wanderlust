import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
from datetime import datetime
import json
from pathlib import Path
import logging

class ErrorAnalyzer:
    """Analyze prediction errors and generate comprehensive reports."""
    
    def __init__(self, output_dir: str = 'results/analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for plots
        plt.style.use('default')  # Use default style instead of seaborn
        plt.rcParams['figure.figsize'] = [10, 6]  # Set default figure size
        plt.rcParams['axes.grid'] = True  # Enable grid by default
        plt.rcParams['grid.alpha'] = 0.3  # Set grid transparency
    
    def calculate_errors(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """Calculate various error metrics."""
        # Calculate Euclidean distances
        errors = np.sqrt(np.sum(np.square(predictions - ground_truth), axis=1))
        
        # Calculate metrics
        metrics = {
            'mean_error': np.mean(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors),
            'max_error': np.max(errors),
            'min_error': np.min(errors),
            'p95_error': np.percentile(errors, 95),
            'p99_error': np.percentile(errors, 99),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'mae': np.mean(np.abs(errors))
        }
        
        return metrics, errors
    
    def analyze_spatial_distribution(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze spatial distribution of errors."""
        # Calculate errors
        errors = np.sqrt(np.sum(np.square(predictions - ground_truth), axis=1))
        
        # Create spatial grid
        x_min, x_max = np.min(ground_truth[:, 0]), np.max(ground_truth[:, 0])
        y_min, y_max = np.min(ground_truth[:, 1]), np.max(ground_truth[:, 1])
        
        grid_size = 20
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        
        # Calculate error statistics for each grid cell
        error_grid = np.zeros((grid_size-1, grid_size-1))
        count_grid = np.zeros((grid_size-1, grid_size-1))
        
        for i in range(len(errors)):
            x_idx = np.searchsorted(x_grid, ground_truth[i, 0]) - 1
            y_idx = np.searchsorted(y_grid, ground_truth[i, 1]) - 1
            
            if 0 <= x_idx < grid_size-1 and 0 <= y_idx < grid_size-1:
                error_grid[x_idx, y_idx] += errors[i]
                count_grid[x_idx, y_idx] += 1
        
        # Calculate average errors
        mask = count_grid > 0
        error_grid[mask] /= count_grid[mask]
        
        return {
            'error_grid': error_grid,
            'count_grid': count_grid,
            'x_grid': x_grid,
            'y_grid': y_grid
        }
    
    def analyze_temporal_patterns(self, errors: np.ndarray, timestamps: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze temporal patterns in errors."""
        # Calculate moving averages
        window_size = min(100, len(errors))
        moving_avg = np.convolve(errors, np.ones(window_size)/window_size, mode='valid')
        
        # Calculate error rate of change
        error_diff = np.diff(errors)
        
        return {
            'moving_avg': moving_avg,
            'error_diff': error_diff,
            'timestamps': timestamps
        }
    
    def analyze_channel_impact(self, errors: np.ndarray, channel_quality: np.ndarray) -> Dict[str, float]:
        """Analyze impact of channel quality on errors."""
        # Calculate correlation
        correlation = np.corrcoef(errors, channel_quality)[0, 1]
        
        # Calculate error statistics for different channel quality ranges
        quality_ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        range_stats = {}
        
        for q_min, q_max in quality_ranges:
            mask = (channel_quality >= q_min) & (channel_quality < q_max)
            if np.any(mask):
                range_stats[f'quality_{q_min:.2f}_{q_max:.2f}'] = {
                    'mean_error': np.mean(errors[mask]),
                    'std_error': np.std(errors[mask]),
                    'count': np.sum(mask)
                }
        
        return {
            'correlation': correlation,
            'range_stats': range_stats
        }
    
    def generate_report(self, model_name: str, metrics: Dict[str, float],
                       errors: np.ndarray, predictions: np.ndarray,
                       ground_truth: np.ndarray, channel_quality: np.ndarray = None):
        """Generate comprehensive error analysis report."""
        # Create plots
        self._plot_error_distribution(errors, model_name)
        self._plot_spatial_errors(predictions, ground_truth, errors, model_name)
        if channel_quality is not None:
            self._plot_channel_correlation(errors, channel_quality, model_name)
        
        # Save metrics
        metrics_path = self.output_dir / f'{model_name}_metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Error Analysis Report for {model_name}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Error Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.4f}\n")
            
            if channel_quality is not None:
                f.write("\nChannel Quality Analysis:\n")
                f.write("-" * 20 + "\n")
                channel_impact = self.analyze_channel_impact(errors, channel_quality)
                f.write(f"Error-Channel Correlation: {channel_impact['correlation']:.4f}\n")
        
        logging.info(f"Error analysis report generated: {metrics_path}")
    
    def _plot_error_distribution(self, errors: np.ndarray, model_name: str):
        """Plot error distribution."""
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50, density=True, alpha=0.7)
        plt.title(f'Error Distribution - {model_name}')
        plt.xlabel('Error (m)')
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f'{model_name}_error_dist.png')
        plt.close()
    
    def _plot_spatial_errors(self, predictions: np.ndarray, ground_truth: np.ndarray,
                           errors: np.ndarray, model_name: str):
        """Plot spatial distribution of errors."""
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(ground_truth[:, 0], ground_truth[:, 1],
                            c=errors, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Error (m)')
        plt.title(f'Spatial Error Distribution - {model_name}')
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f'{model_name}_spatial_errors.png')
        plt.close()
    
    def _plot_channel_correlation(self, errors: np.ndarray, channel_quality: np.ndarray,
                                model_name: str):
        """Plot channel quality vs error correlation."""
        plt.figure(figsize=(10, 6))
        plt.scatter(channel_quality, errors, alpha=0.5)
        plt.title(f'Channel Quality vs Error - {model_name}')
        plt.xlabel('Channel Quality')
        plt.ylabel('Error (m)')
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f'{model_name}_channel_correlation.png')
        plt.close()

def main():
    # Example usage
    analyzer = ErrorAnalyzer()
    
    # Simulate some data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate ground truth positions
    ground_truth = np.random.uniform(-10, 10, (n_samples, 2))
    
    # Generate predictions with some error
    errors = np.random.normal(0, 1, n_samples)
    angles = np.random.uniform(0, 2*np.pi, n_samples)
    error_vectors = np.column_stack((
        errors * np.cos(angles),
        errors * np.sin(angles)
    ))
    predictions = ground_truth + error_vectors
    
    # Calculate metrics
    metrics, errors = analyzer.calculate_errors(predictions, ground_truth)
    
    # Generate report
    analyzer.generate_report(
        model_name="example_model",
        metrics=metrics,
        errors=errors,
        predictions=predictions,
        ground_truth=ground_truth
    )

if __name__ == '__main__':
    main() 