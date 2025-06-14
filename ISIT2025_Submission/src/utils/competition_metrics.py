import numpy as np
import tensorflow as tf
from typing import Dict, Tuple, Optional

class R90Metric(tf.keras.metrics.Metric):
    """
    R90 metric - radius containing 90% of prediction errors.
    Implements the competition's R90 metric as a Keras metric.
    """
    
    def __init__(self, name='r90', **kwargs):
        super().__init__(name=name, **kwargs)
        self.error_accumulator = self.add_weight(
            name='error_accumulator',
            initializer='zeros',
            shape=(),
            dtype=tf.float32
        )
        self.count = self.add_weight(
            name='count',
            initializer='zeros',
            dtype=tf.float32
        )
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Calculate Euclidean distances
        errors = tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=1))
        
        # Update running statistics
        batch_r90 = tf.numpy_function(
            lambda x: np.float32(np.percentile(x, 90)),
            [errors],
            tf.float32
        )
        self.error_accumulator.assign_add(batch_r90)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.error_accumulator / self.count
    
    def reset_state(self):
        self.error_accumulator.assign(0.0)
        self.count.assign(0.0)

class CombinedCompetitionMetric(tf.keras.metrics.Metric):
    """
    Combined competition metric incorporating both MAE and R90.
    Formula: 0.7 * MAE + 0.3 * R90
    """
    
    def __init__(self, name='combined_competition_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mae_accumulator = self.add_weight(name='mae_acc', initializer='zeros')
        self.r90_metric = R90Metric()
        self.count = self.add_weight(name='count', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        mae = tf.reduce_mean(tf.abs(y_true - y_pred))
        self.mae_accumulator.assign_add(mae)
        self.r90_metric.update_state(y_true, y_pred)
        self.count.assign_add(1.0)
        
    def result(self):
        mae = self.mae_accumulator / self.count
        r90 = self.r90_metric.result()
        return 0.7 * mae + 0.3 * r90
    
    def reset_state(self):
        self.mae_accumulator.assign(0.0)
        self.r90_metric.reset_state()
        self.count.assign(0.0)

def calculate_competition_metrics(y_true: np.ndarray,
                               y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate all competition metrics for given predictions.
    
    Args:
        y_true: Ground truth positions (N, 2)
        y_pred: Predicted positions (N, 2)
        
    Returns:
        Dictionary containing MAE, R90, and combined metric values
    """
    # Input validation
    if y_true.shape != y_pred.shape:
        raise ValueError("True and predicted positions must have the same shape")
    if len(y_true.shape) != 2 or y_true.shape[1] != 2:
        raise ValueError("Positions must be 2D arrays with shape (N, 2)")
    
    # Calculate MAE
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Calculate R90
    errors = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    r90 = np.percentile(errors, 90)
    
    # Calculate combined metric
    combined = 0.7 * mae + 0.3 * r90
    
    return {
        'mae': float(mae),
        'r90': float(r90),
        'combined_metric': float(combined)
    }

def evaluate_trajectory_metrics(true_trajectories: np.ndarray,
                              pred_trajectories: np.ndarray,
                              time_steps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate metrics for trajectory predictions.
    
    Args:
        true_trajectories: Ground truth trajectories (N, T, 2)
        pred_trajectories: Predicted trajectories (N, T, 2)
        time_steps: Optional time steps for temporal weighting (N, T)
        
    Returns:
        Dictionary containing trajectory-specific metrics
    """
    # Input validation
    if true_trajectories.shape != pred_trajectories.shape:
        raise ValueError("True and predicted trajectories must have the same shape")
    if len(true_trajectories.shape) != 3 or true_trajectories.shape[2] != 2:
        raise ValueError("Trajectories must be 3D arrays with shape (N, T, 2)")
    
    N, T, _ = true_trajectories.shape
    
    # Initialize metrics
    metrics = {}
    
    # Calculate point-wise metrics
    errors = np.sqrt(np.sum((true_trajectories - pred_trajectories) ** 2, axis=2))
    
    # Basic metrics
    metrics['trajectory_mae'] = float(np.mean(errors))
    metrics['trajectory_r90'] = float(np.percentile(errors, 90))
    
    # Endpoint error
    endpoint_errors = np.sqrt(np.sum(
        (true_trajectories[:, -1] - pred_trajectories[:, -1]) ** 2,
        axis=1
    ))
    metrics['endpoint_mae'] = float(np.mean(endpoint_errors))
    metrics['endpoint_r90'] = float(np.percentile(endpoint_errors, 90))
    
    # Smoothness metric (average acceleration)
    if T >= 3:
        acc = np.diff(np.diff(pred_trajectories, axis=1), axis=1)
        metrics['smoothness'] = float(np.mean(np.sqrt(np.sum(acc ** 2, axis=2))))
    
    # Time-weighted metrics if time_steps provided
    if time_steps is not None:
        if time_steps.shape != (N, T):
            raise ValueError("Time steps must have shape (N, T)")
        
        # Normalize weights
        weights = time_steps / np.sum(time_steps, axis=1, keepdims=True)
        
        # Calculate weighted MAE
        weighted_mae = np.mean(np.sum(errors * weights, axis=1))
        metrics['time_weighted_mae'] = float(weighted_mae)
    
    return metrics

def calculate_feature_importance_metrics(feature_scores: np.ndarray,
                                      selected_features: np.ndarray) -> Dict[str, float]:
    """
    Calculate metrics for feature selection performance.
    
    Args:
        feature_scores: Importance scores for each feature (N,)
        selected_features: Binary array indicating selected features (N,)
        
    Returns:
        Dictionary containing feature selection metrics
    """
    # Input validation
    if feature_scores.shape != selected_features.shape:
        raise ValueError("Feature scores and selection mask must have the same shape")
    
    metrics = {}
    
    # Calculate average importance of selected features
    metrics['mean_selected_importance'] = float(
        np.mean(feature_scores[selected_features > 0])
    )
    
    # Calculate selection ratio
    metrics['selection_ratio'] = float(
        np.sum(selected_features > 0) / len(selected_features)
    )
    
    # Calculate importance concentration (ratio of total importance in selected features)
    total_importance = np.sum(feature_scores)
    selected_importance = np.sum(feature_scores[selected_features > 0])
    metrics['importance_concentration'] = float(
        selected_importance / total_importance if total_importance > 0 else 0
    )
    
    return metrics 