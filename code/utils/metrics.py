import numpy as np
import tensorflow as tf

def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error between predicted and true locations.
    
    Args:
        y_true (np.ndarray): Ground truth locations, shape (N, 2) for x,y coordinates
        y_pred (np.ndarray): Predicted locations, shape (N, 2) for x,y coordinates
        
    Returns:
        float: Mean Absolute Error
    """
    return np.mean(np.abs(y_true - y_pred))

def calculate_r90(y_true, y_pred):
    """
    Calculate R90 metric - the radius of the circle centered at the true location 
    that contains 90% of the predicted locations.
    
    Args:
        y_true (np.ndarray): Ground truth locations, shape (N, 2) for x,y coordinates
        y_pred (np.ndarray): Predicted locations, shape (N, 2) for x,y coordinates
        
    Returns:
        float: R90 value
    """
    # Calculate Euclidean distances between true and predicted locations
    distances = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
    
    # Sort distances and find the 90th percentile
    r90 = np.percentile(distances, 90)
    return r90

def combined_loss(y_true, y_pred):
    """Combined loss function for the competition."""
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return mse + 0.5 * mae

class CombinedMetric(tf.keras.metrics.Metric):
    """Custom metric combining MAE and R90."""
    
    def __init__(self, name='combined_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mae = tf.keras.metrics.MeanAbsoluteError()
        self.r90 = tf.keras.metrics.Mean()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update MAE
        self.mae.update_state(y_true, y_pred, sample_weight)
        
        # Calculate R90
        errors = tf.abs(y_true - y_pred)
        r90 = tf.reduce_mean(tf.nn.top_k(errors, k=tf.cast(tf.shape(errors)[0] * 0.9, tf.int32)).values)
        self.r90.update_state(r90)
        
    def result(self):
        return 0.7 * self.mae.result() + 0.3 * self.r90.result()
    
    def reset_states(self):
        self.mae.reset_states()
        self.r90.reset_states()

def calculate_combined_score(y_true, y_pred):
    """Calculate combined competition score."""
    mae = calculate_mae(y_true, y_pred)
    r90 = calculate_r90(y_true, y_pred)
    return 0.7 * mae + 0.3 * r90 