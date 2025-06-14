import os
import tensorflow as tf
import numpy as np
from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import GrantFreeAccessModel
from src.utils.competition_metrics import calculate_competition_metrics
from data_processing.generate_data import generate_synthetic_data
from src.utils.metrics import CombinedMetric, combined_loss
import json
from src.data_loader import CSIDataLoader

def validate_model(model_type: str, data_dir: str, output_dir: str):
    """
    Validate model against competition dataset.
    
    Args:
        model_type: Type of model to validate ('vanilla', 'trajectory', 'feature_selection')
        data_dir: Directory containing competition data
        output_dir: Directory to save validation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from TFRecord files
    print("Loading competition data...")
    data_loader = CSIDataLoader(os.path.join(data_dir, 'test_tfrecord', '*.tfrecords'))
    test_dataset, _ = data_loader.load_dataset(is_training=False)
    
    # Load appropriate model with custom objects
    print("\nLoading model...")
    if model_type == 'vanilla':
        model = VanillaLocalizationModel()
        model.model = tf.keras.models.load_model(
            os.path.join('models', 'vanilla_model.h5'),
            custom_objects={
                'VanillaLocalizationModel': VanillaLocalizationModel,
                'CombinedMetric': CombinedMetric
            },
            compile=False
        )
    elif model_type == 'trajectory':
        model = TrajectoryAwareLocalizationModel(sequence_length=5)
        model.model = tf.keras.models.load_model(
            os.path.join('models', 'trajectory_model.h5'),
            custom_objects={
                'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
                'CombinedMetric': CombinedMetric,
                'combined_loss': combined_loss
            },
            compile=False
        )
    else:  # feature_selection
        model = GrantFreeAccessModel()
        model.model = tf.keras.models.load_model(
            os.path.join('models', 'feature_selection_model.h5'),
            custom_objects={
                'GrantFreeAccessModel': GrantFreeAccessModel,
                'CombinedMetric': CombinedMetric
            },
            compile=False
        )
        
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    positions = []
    
    for batch in test_dataset:
        if model_type == 'trajectory':
            # For trajectory model, reshape input to sequence format
            csi = batch['csi']
            csi = tf.reshape(csi, [-1, 5, 32, 1024, 2])  # Reshape to sequence format
            pred = model.predict(csi)
        else:
            pred = model.predict(batch['csi'])
        predictions.append(pred)
        positions.append(batch['position'].numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    positions = np.concatenate(positions, axis=0)
    
    # Calculate metrics
    metrics = calculate_competition_metrics(positions, predictions)
    
    print("\nValidation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R90: {metrics['r90']:.4f}")
    print(f"Combined Metric: {metrics['combined_metric']:.4f}")
    
    # Save results
    np.save(os.path.join(output_dir, f'{model_type}_predictions.npy'), predictions)
    np.save(os.path.join(output_dir, f'{model_type}_positions.npy'), positions)
    with open(os.path.join(output_dir, f'{model_type}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

if __name__ == "__main__":
    # Configuration
    model_types = ['vanilla', 'trajectory', 'feature_selection']  # Updated model type
    data_dir = 'data/competition'
    output_dir = 'outputs/validation'
    
    # Validate all models
    for model_type in model_types:
        print(f"\nValidating {model_type} model...")
        model_output_dir = os.path.join(output_dir, model_type)
        metrics = validate_model(model_type, data_dir, model_output_dir) 