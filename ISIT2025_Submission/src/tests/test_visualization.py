import unittest
import numpy as np
import os
from pathlib import Path
from ..utils.visualization import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_trajectory_predictions,
    plot_feature_importance,
    create_performance_report
)

class TestVisualization(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path('test_output')
        self.test_dir.mkdir(exist_ok=True)
        
        # Sample data for testing
        self.history = {
            'loss': [0.5, 0.4, 0.3],
            'val_loss': [0.6, 0.5, 0.4],
            'mae': [0.3, 0.2, 0.1],
            'val_mae': [0.4, 0.3, 0.2]
        }
        
        self.true_positions = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        
        self.pred_positions = np.array([
            [0.1, 0.1],
            [1.1, 0.9],
            [1.9, 2.1]
        ])
        
        self.feature_scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        self.timestamps = np.array([0, 1, 2])
    
    def tearDown(self):
        # Clean up test files
        if self.test_dir.exists():
            for file in self.test_dir.glob('*'):
                file.unlink()
            self.test_dir.rmdir()
    
    def test_plot_training_history(self):
        output_path = self.test_dir / 'training_history.png'
        plot_training_history(self.history, str(output_path))
        self.assertTrue(output_path.exists())
    
    def test_plot_predictions(self):
        output_path = self.test_dir / 'predictions.png'
        plot_predictions(
            self.true_positions,
            self.pred_positions,
            str(output_path)
        )
        self.assertTrue(output_path.exists())
    
    def test_plot_error_distribution(self):
        output_path = self.test_dir / 'errors.png'
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        plot_error_distribution(errors, str(output_path))
        self.assertTrue(output_path.exists())
    
    def test_plot_feature_importance(self):
        output_path = self.test_dir / 'feature_importance.png'
        plot_feature_importance(
            self.feature_scores,
            feature_names=['F1', 'F2', 'F3', 'F4', 'F5'],
            save_path=str(output_path)
        )
        self.assertTrue(output_path.exists())
    
    def test_plot_trajectory_predictions(self):
        output_path = self.test_dir / 'trajectory.png'
        plot_trajectory_predictions(
            self.true_positions,
            self.timestamps,
            str(output_path)
        )
        self.assertTrue(output_path.exists())
    
    def test_create_performance_report(self):
        metrics = {
            'mae': 0.1,
            'r90': 0.2,
            'loss': 0.3
        }
        
        create_performance_report(
            'test_model',
            metrics,
            self.history,
            self.true_positions,
            self.pred_positions,
            str(self.test_dir)
        )
        
        expected_files = [
            'test_model_training_history.png',
            'test_model_predictions.png',
            'test_model_error_distribution.png',
            'test_model_metrics_summary.txt'
        ]
        
        for file in expected_files:
            self.assertTrue((self.test_dir / file).exists())
            
    def test_invalid_inputs(self):
        # Test with empty data
        with self.assertRaises(ValueError):
            plot_training_history({})
        
        # Test with mismatched dimensions
        with self.assertRaises(ValueError):
            plot_predictions(
                np.array([[0, 0]]),
                np.array([[0, 0], [1, 1]])
            )
        
        # Test with invalid feature names
        with self.assertRaises(ValueError):
            plot_feature_importance(
                np.array([0.1, 0.2]),
                feature_names=['F1']  # Mismatched length
            )

if __name__ == '__main__':
    unittest.main() 