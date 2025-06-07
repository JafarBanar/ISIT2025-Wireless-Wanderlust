import os
import numpy as np
import tensorflow as tf
from models.csi_localization import create_csi_localization_model
from data.csi_preprocessing import preprocess_csi_data, create_sequence_data

def load_model(model_path, input_shape, num_classes=2):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the saved model
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Loaded model
    """
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        # If loading fails, create a new model and load weights
        model = create_csi_localization_model(input_shape, num_classes)
        model.load_weights(model_path)
    
    return model

def evaluate_model(model, test_data, test_labels):
    """
    Evaluate model performance.
    
    Args:
        model (tf.keras.Model): Trained model
        test_data (numpy.ndarray): Test data
        test_labels (numpy.ndarray): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    predictions = model.predict(test_data)
    
    # Calculate metrics
    mae = np.mean(np.abs(predictions - test_labels))
    mse = np.mean(np.square(predictions - test_labels))
    
    # Calculate R90 metric
    errors = np.sqrt(np.sum(np.square(predictions - test_labels), axis=1))
    r90 = np.percentile(errors, 90)
    
    # Calculate trajectory metrics
    trajectory_smoothness = calculate_trajectory_smoothness(predictions)
    
    return {
        'mae': mae,
        'mse': mse,
        'r90': r90,
        'trajectory_smoothness': trajectory_smoothness
    }

def calculate_trajectory_smoothness(predictions):
    """
    Calculate trajectory smoothness metric.
    
    Args:
        predictions (numpy.ndarray): Model predictions
        
    Returns:
        float: Trajectory smoothness score
    """
    # Calculate differences between consecutive points
    diffs = np.diff(predictions, axis=0)
    
    # Calculate angles between consecutive movements
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    
    # Calculate angle differences
    angle_diffs = np.diff(angles)
    
    # Calculate smoothness as inverse of angle changes
    smoothness = 1.0 / (1.0 + np.mean(np.abs(angle_diffs)))
    
    return smoothness

def generate_predictions(model, csi_data, sequence_length=10):
    """
    Generate predictions for new CSI data.
    
    Args:
        model (tf.keras.Model): Trained model
        csi_data (numpy.ndarray): CSI data
        sequence_length (int): Length of sequences
        
    Returns:
        numpy.ndarray: Predicted locations
    """
    # Preprocess data
    processed_data = preprocess_csi_data(csi_data)
    
    # Create sequences
    sequence_data, _ = create_sequence_data(
        processed_data, 
        np.zeros((len(processed_data), 2)),  # Dummy labels
        sequence_length
    )
    
    # Generate predictions
    predictions = model.predict(sequence_data)
    
    return predictions

def main():
    # Configuration
    MODEL_PATH = 'models/csi_localization/best_model.h5'
    TEST_DATA_PATH = 'data/test_csi_data.npy'
    TEST_LABELS_PATH = 'data/test_labels.npy'
    INPUT_SHAPE = (32, 32, 2)  # Adjust based on your data
    NUM_CLASSES = 2
    SEQUENCE_LENGTH = 10
    
    # Load model
    model = load_model(MODEL_PATH, INPUT_SHAPE, NUM_CLASSES)
    
    # Load test data
    test_data = np.load(TEST_DATA_PATH)
    test_labels = np.load(TEST_LABELS_PATH)
    
    # Evaluate model
    metrics = evaluate_model(model, test_data, test_labels)
    
    # Print results
    print("\nEvaluation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"R90: {metrics['r90']:.4f}")
    print(f"Trajectory Smoothness: {metrics['trajectory_smoothness']:.4f}")
    
    # Generate predictions for competition
    predictions = generate_predictions(model, test_data, SEQUENCE_LENGTH)
    
    # Save predictions
    np.save('submission/predictions.npy', predictions)
    print("\nPredictions saved to submission/predictions.npy")

if __name__ == "__main__":
    main() 