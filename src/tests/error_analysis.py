import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

class ErrorAnalyzer:
    def __init__(self, model, test_data, test_labels):
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.predictions = None
        self.errors = None
        self.error_stats = {}
        
    def compute_predictions(self):
        """Generate predictions and compute errors"""
        self.predictions = self.model.predict(self.test_data)
        self.errors = np.linalg.norm(self.predictions - self.test_labels, axis=1)
        
    def analyze_error_distribution(self) -> Dict[str, float]:
        """Analyze error distribution statistics"""
        if self.errors is None:
            self.compute_predictions()
            
        stats = {
            'mean_error': np.mean(self.errors),
            'median_error': np.median(self.errors),
            'std_error': np.std(self.errors),
            'max_error': np.max(self.errors),
            'min_error': np.min(self.errors),
            'r90': np.percentile(self.errors, 90)
        }
        self.error_stats.update(stats)
        return stats
    
    def identify_worst_cases(self, n: int = 10) -> List[Tuple[int, float]]:
        """Identify worst performing cases"""
        if self.errors is None:
            self.compute_predictions()
            
        worst_indices = np.argsort(self.errors)[-n:][::-1]
        return [(idx, self.errors[idx]) for idx in worst_indices]
    
    def analyze_spatial_distribution(self) -> Dict[str, np.ndarray]:
        """Analyze error distribution in spatial coordinates"""
        x_coords = self.test_labels[:, 0]
        y_coords = self.test_labels[:, 1]
        
        # Create spatial grid
        x_bins = np.linspace(min(x_coords), max(x_coords), 20)
        y_bins = np.linspace(min(y_coords), max(y_coords), 20)
        
        spatial_errors = np.zeros((len(x_bins)-1, len(y_bins)-1))
        counts = np.zeros_like(spatial_errors)
        
        for i in range(len(self.errors)):
            x_idx = np.digitize(x_coords[i], x_bins) - 1
            y_idx = np.digitize(y_coords[i], y_bins) - 1
            if 0 <= x_idx < len(x_bins)-1 and 0 <= y_idx < len(y_bins)-1:
                spatial_errors[x_idx, y_idx] += self.errors[i]
                counts[x_idx, y_idx] += 1
                
        # Avoid division by zero
        counts[counts == 0] = 1
        avg_spatial_errors = spatial_errors / counts
        
        return {
            'spatial_errors': avg_spatial_errors,
            'x_bins': x_bins,
            'y_bins': y_bins
        }
    
    def generate_report(self, output_dir: str = 'results/'):
        """Generate comprehensive error analysis report"""
        # Basic statistics
        stats = self.analyze_error_distribution()
        
        # Worst cases
        worst_cases = self.identify_worst_cases()
        
        # Spatial analysis
        spatial_analysis = self.analyze_spatial_distribution()
        
        # Generate plots
        plt.figure(figsize=(15, 5))
        
        # Error histogram
        plt.subplot(131)
        plt.hist(self.errors, bins=50)
        plt.title('Error Distribution')
        plt.xlabel('Error Magnitude')
        plt.ylabel('Count')
        
        # Spatial heatmap
        plt.subplot(132)
        sns.heatmap(spatial_analysis['spatial_errors'])
        plt.title('Spatial Error Distribution')
        
        # Error vs. Position
        plt.subplot(133)
        plt.scatter(self.test_labels[:, 0], self.test_labels[:, 1], 
                   c=self.errors, cmap='viridis')
        plt.colorbar()
        plt.title('Error vs. Position')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/error_analysis.png')
        plt.close()
        
        # Save statistics to file
        with open(f'{output_dir}/error_stats.txt', 'w') as f:
            f.write('Error Statistics:\n')
            for key, value in stats.items():
                f.write(f'{key}: {value:.4f}\n')
            
            f.write('\nWorst Cases:\n')
            for idx, error in worst_cases:
                f.write(f'Index {idx}: Error = {error:.4f}\n')
                
    def analyze_feature_importance(self) -> np.ndarray:
        """Analyze feature importance using gradient-based methods"""
        input_data = tf.convert_to_tensor(self.test_data)
        
        with tf.GradientTape() as tape:
            tape.watch(input_data)
            predictions = self.model(input_data)
            loss = tf.reduce_mean(tf.square(predictions - self.test_labels))
            
        gradients = tape.gradient(loss, input_data)
        feature_importance = np.mean(np.abs(gradients.numpy()), axis=0)
        
        return feature_importance
    
    def cross_validation_analysis(self, k_folds: int = 5) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation analysis"""
        dataset_size = len(self.test_data)
        fold_size = dataset_size // k_folds
        metrics = {
            'mae': [],
            'mse': [],
            'r90': []
        }
        
        for i in range(k_folds):
            start_idx = i * fold_size
            end_idx = start_idx + fold_size
            
            # Create fold datasets
            val_data = self.test_data[start_idx:end_idx]
            val_labels = self.test_labels[start_idx:end_idx]
            
            # Compute metrics
            predictions = self.model.predict(val_data)
            errors = np.linalg.norm(predictions - val_labels, axis=1)
            
            metrics['mae'].append(np.mean(errors))
            metrics['mse'].append(np.mean(np.square(errors)))
            metrics['r90'].append(np.percentile(errors, 90))
            
        return metrics 