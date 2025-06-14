"""Validate all models against competition requirements."""

import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any
import pandas as pd

from .config.competition_config import (
    PATHS,
    VANILLA_MODEL_CONFIG,
    TRAJECTORY_MODEL_CONFIG,
    FEATURE_SELECTION_CONFIG,
    TRAINING_CONFIG,
    DEADLINES,
    COMPETITION_METRICS,
    COMPETITION_MODELS
)
from .models.vanilla_localization import VanillaLocalizationModel
from .models.trajectory_model import TrajectoryAwareModel
from .models.feature_selection import GrantFreeAccessModel
from src.utils.competition_metrics import calculate_competition_metrics
from .data_processing.generate_competition_data import generate_competition_data

def load_competition_data(data_dir: str) -> Dict[str, np.ndarray]:
    """Load or generate competition data."""
    if not os.path.exists(os.path.join(data_dir, 'csi_features.npy')):
        print("Generating new competition data...")
        return generate_competition_data(data_dir)
    
    print("Loading existing competition data...")
    return {
        'csi_features': np.load(os.path.join(data_dir, 'csi_features.npy')),
        'positions': np.load(os.path.join(data_dir, 'positions.npy')),
        'dataset_info': np.load(os.path.join(data_dir, 'dataset_info.npy'), allow_pickle=True).item()
    }

def create_dataset(csi_features: np.ndarray,
                  positions: np.ndarray,
                  sequence_length: int = None) -> tf.data.Dataset:
    """Create TensorFlow dataset."""
    if sequence_length is None:
        # For vanilla model
        return tf.data.Dataset.from_tensor_slices({
            'csi_features': csi_features,
            'position': positions
        }).batch(TRAINING_CONFIG['batch_size'])
    else:
        # For sequence models (trajectory and feature selection)
        n_sequences = len(csi_features) - sequence_length + 1
        feature_sequences = np.zeros((n_sequences, sequence_length, *csi_features.shape[1:]))
        position_sequences = np.zeros((n_sequences, sequence_length, 2))
        
        for i in range(n_sequences):
            feature_sequences[i] = csi_features[i:i + sequence_length]
            position_sequences[i] = positions[i:i + sequence_length]
        
        return tf.data.Dataset.from_tensor_slices({
            'csi_features': feature_sequences,
            'prev_positions': position_sequences[:, :-1],
            'position': position_sequences[:, -1]
        }).batch(TRAINING_CONFIG['batch_size'])

def validate_vanilla_model(data: Dict[str, np.ndarray],
                         output_dir: str) -> Dict[str, float]:
    """Validate vanilla localization model."""
    print("\nValidating Vanilla Localization Model...")
    
    # Create model
    model = VanillaLocalizationModel(**VANILLA_MODEL_CONFIG)
    
    # Create dataset
    dataset = create_dataset(data['csi_features'], data['positions'])
    
    # Make predictions
    predictions = model.predict(dataset)
    
    # Calculate metrics
    metrics = calculate_competition_metrics(data['positions'], predictions)
    
    # Save results
    np.save(os.path.join(output_dir, 'vanilla_predictions.npy'), predictions)
    with open(os.path.join(output_dir, 'vanilla_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def validate_trajectory_model(data: Dict[str, np.ndarray],
                           output_dir: str) -> Dict[str, float]:
    """Validate trajectory-aware model."""
    print("\nValidating Trajectory-Aware Model...")
    
    # Create model
    model = TrajectoryAwareModel(**TRAJECTORY_MODEL_CONFIG)
    
    # Create dataset
    dataset = create_dataset(
        data['csi_features'],
        data['positions'],
        sequence_length=TRAJECTORY_MODEL_CONFIG['sequence_length']
    )
    
    # Make predictions
    predictions = model.predict(dataset)
    if isinstance(predictions, dict):
        predictions = predictions['location']
    
    # Calculate metrics
    metrics = calculate_competition_metrics(
        data['positions'][TRAJECTORY_MODEL_CONFIG['sequence_length']-1:],
        predictions
    )
    
    # Save results
    np.save(os.path.join(output_dir, 'trajectory_predictions.npy'), predictions)
    with open(os.path.join(output_dir, 'trajectory_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def validate_feature_selection_model(data: Dict[str, np.ndarray],
                                  output_dir: str) -> Dict[str, float]:
    """Validate feature selection model."""
    print("\nValidating Feature Selection Model...")
    
    # Create model
    model = GrantFreeAccessModel(**FEATURE_SELECTION_CONFIG)
    
    # Create dataset
    dataset = create_dataset(
        data['csi_features'],
        data['positions'],
        sequence_length=FEATURE_SELECTION_CONFIG['sequence_length']
    )
    
    # Make predictions
    predictions = model.predict(dataset)
    if isinstance(predictions, dict):
        predictions = predictions['location']
    
    # Calculate metrics
    metrics = calculate_competition_metrics(
        data['positions'][FEATURE_SELECTION_CONFIG['sequence_length']-1:],
        predictions
    )
    
    # Save results
    np.save(os.path.join(output_dir, 'feature_selection_predictions.npy'), predictions)
    with open(os.path.join(output_dir, 'feature_selection_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def main():
    """Run validation for all models."""
    # Load competition data
    data = load_competition_data(PATHS['data_dir'])
    
    # Create output directory
    os.makedirs(PATHS['output_dir'], exist_ok=True)
    
    # Validate all models
    results = {
        'vanilla': validate_vanilla_model(data, PATHS['output_dir']),
        'trajectory': validate_trajectory_model(data, PATHS['output_dir']),
        'feature_selection': validate_feature_selection_model(data, PATHS['output_dir'])
    }
    
    # Save combined results
    results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(os.path.join(PATHS['output_dir'], 'validation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nValidation Results Summary:")
    for model_type, metrics in results.items():
        if model_type != 'timestamp':
            print(f"\n{model_type.upper()} MODEL:")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"R90: {metrics['r90']:.4f}")
            print(f"Combined Score: {metrics['combined_metric']:.4f}")

if __name__ == "__main__":
    main() 