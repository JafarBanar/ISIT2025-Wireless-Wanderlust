import unittest
import numpy as np
import tensorflow as tf
from src.isit2025.utils.competition_metrics import (
    R90Metric,
    CombinedCompetitionMetric,
    calculate_competition_metrics,
    evaluate_trajectory_metrics,
    calculate_feature_importance_metrics
)

class TestCompetitionMetrics(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.y_true = np.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4]
        ])
        
        self.y_pred = np.array([
            [0.1, 0.1],
            [1.1, 0.9],
            [1.9, 2.1],
            [3.2, 2.8],
            [3.8, 4.2]
        ])
        
        # Sample trajectory data
        self.true_trajectories = np.array([
            [[0, 0], [1, 1], [2, 2]],
            [[1, 1], [2, 2], [3, 3]],
            [[2, 2], [3, 3], [4, 4]]
        ])
        
        self.pred_trajectories = np.array([
            [[0.1, 0.1], [1.1, 0.9], [1.9, 2.1]],
            [[1.1, 0.9], [1.9, 2.1], [3.2, 2.8]],
            [[1.9, 2.1], [3.2, 2.8], [3.8, 4.2]]
        ])
        
        self.time_steps = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2]
        ])
        
        # Sample feature data
        self.feature_scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2])
        self.selected_features = np.array([0, 1, 0, 1, 0])
    
    def test_r90_metric(self):
        metric = R90Metric()
        
        # Test single update
        metric.update_state(self.y_true, self.y_pred)
        r90 = metric.result().numpy()
        
        # Verify r90 is reasonable
        self.assertGreater(r90, 0)
        self.assertLess(r90, np.max(np.abs(self.y_true - self.y_pred)))
        
        # Test reset
        metric.reset_state()
        self.assertEqual(metric.error_accumulator.shape[0], 0)
    
    def test_combined_competition_metric(self):
        metric = CombinedCompetitionMetric()
        
        # Test single update
        metric.update_state(self.y_true, self.y_pred)
        combined = metric.result().numpy()
        
        # Calculate expected result
        mae = np.mean(np.abs(self.y_true - self.y_pred))
        errors = np.sqrt(np.sum((self.y_true - self.y_pred) ** 2, axis=1))
        r90 = np.percentile(errors, 90)
        expected = 0.7 * mae + 0.3 * r90
        
        # Verify combined metric
        self.assertAlmostEqual(combined, expected, places=5)
        
        # Test reset
        metric.reset_state()
        self.assertEqual(metric.mae_accumulator.numpy(), 0)
        self.assertEqual(metric.count.numpy(), 0)
    
    def test_calculate_competition_metrics(self):
        metrics = calculate_competition_metrics(self.y_true, self.y_pred)
        
        # Verify all required metrics are present
        self.assertIn('mae', metrics)
        self.assertIn('r90', metrics)
        self.assertIn('combined_metric', metrics)
        
        # Verify metric values are reasonable
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['r90'], 0)
        self.assertGreater(metrics['combined_metric'], 0)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            calculate_competition_metrics(
                np.array([[0, 0]]),
                np.array([[0, 0], [1, 1]])
            )
    
    def test_evaluate_trajectory_metrics(self):
        metrics = evaluate_trajectory_metrics(
            self.true_trajectories,
            self.pred_trajectories,
            self.time_steps
        )
        
        # Verify all required metrics are present
        required_metrics = [
            'trajectory_mae',
            'trajectory_r90',
            'endpoint_mae',
            'endpoint_r90',
            'smoothness',
            'time_weighted_mae'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreater(metrics[metric], 0)
        
        # Test without time steps
        metrics_no_time = evaluate_trajectory_metrics(
            self.true_trajectories,
            self.pred_trajectories
        )
        self.assertNotIn('time_weighted_mae', metrics_no_time)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            evaluate_trajectory_metrics(
                self.true_trajectories,
                self.pred_trajectories[:-1]  # Mismatched shape
            )
    
    def test_calculate_feature_importance_metrics(self):
        metrics = calculate_feature_importance_metrics(
            self.feature_scores,
            self.selected_features
        )
        
        # Verify all required metrics are present
        required_metrics = [
            'mean_selected_importance',
            'selection_ratio',
            'importance_concentration'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Verify metric values
        self.assertGreater(metrics['mean_selected_importance'], 0)
        self.assertEqual(metrics['selection_ratio'], 0.4)  # 2/5 features selected
        self.assertGreater(metrics['importance_concentration'], 0)
        self.assertLessEqual(metrics['importance_concentration'], 1)
        
        # Test invalid inputs
        with self.assertRaises(ValueError):
            calculate_feature_importance_metrics(
                self.feature_scores,
                self.selected_features[:-1]  # Mismatched shape
            )

if __name__ == '__main__':
    unittest.main() 