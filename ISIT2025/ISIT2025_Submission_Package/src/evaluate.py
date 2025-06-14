import os
import numpy as np
import tensorflow as tf
import json
import argparse
from pathlib import Path
from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import FeatureSelectionModel
from src.data_loader import CSIDataLoader
from src.data.csi_preprocessing import preprocess_csi_data, create_sequence_data
from src.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging(__name__)

def load_test_data(data_pattern):
    """Load test data using CSIDataLoader."""
    try:
        logger.info(f"Loading test data from pattern: {data_pattern}")
        data_loader = CSIDataLoader(data_pattern)
        test_dataset, _ = data_loader.load_dataset(is_training=False)
        logger.info("Test data loaded successfully")
        return test_dataset
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        raise

def evaluate_model(model_type, model_path, data_dir, output_dir='evaluation_results'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_type (str): Type of model to evaluate ('vanilla', 'trajectory', 'feature_selection')
        model_path (str): Path to the saved model file
        data_dir (str): Directory containing test data
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation results
    """
    logger.info(f"Starting evaluation for {model_type} model from {model_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model with custom objects
    try:
        if model_type == 'vanilla':
            model = VanillaLocalizationModel.load_model(
                model_path,
                input_shape=(32, 1024, 2),
                num_classes=2
            )
        elif model_type == 'trajectory':
            model = TrajectoryAwareLocalizationModel.load_model(
                model_path,
                input_shape=(32, 1024, 2),
                num_classes=2
            )
        elif model_type == 'feature_selection':
            model = FeatureSelectionModel.load_model(
                model_path,
                input_shape=(32, 1024, 2),
                num_classes=2
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        logger.info(f"Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Load test data
    test_data = load_test_data(os.path.join(data_dir, '*.tfrecords'))
    
    # Evaluate model
    try:
        logger.info("Starting model evaluation")
        metrics = model.evaluate(test_data)
        logger.info("Model evaluation completed")
    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise
    
    # Save evaluation results
    results = {
        'model_type': model_type,
        'model_path': str(model_path),
        'metrics': {
            'loss': float(metrics[0]),
            'combined_metric': float(metrics[1]),
            'r90_metric': float(metrics[2])
        }
    }
    
    output_file = output_dir / f'{model_type}_evaluation.json'
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {str(e)}")
        raise
    
    return results

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate CSI localization models')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vanilla', 'trajectory', 'feature_selection'],
                      help='Type of model to evaluate')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved model file')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--sequence_length', type=int, default=10,
                      help='Length of sequences for prediction')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
    # Evaluate model
        results = evaluate_model(
            args.model_type,
            args.model_path,
            args.data_dir,
            args.output_dir
        )
    
    # Print results
        logger.info("\nEvaluation Results:")
        logger.info(f"Loss: {results['metrics']['loss']:.4f}")
        logger.info(f"Combined Metric: {results['metrics']['combined_metric']:.4f}")
        logger.info(f"R90 Metric: {results['metrics']['r90_metric']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 