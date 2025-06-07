import os
import tensorflow as tf
import numpy as np
from src.isit2025.models.vanilla_localization import VanillaLocalizationModel
from src.isit2025.models.trajectory_model import TrajectoryAwareModel
from src.isit2025.models.feature_selection import GrantFreeAccessModel
from src.isit2025.utils.competition_metrics import calculate_competition_metrics
from src.isit2025.data_processing.generate_data import generate_synthetic_data

def validate_model(model_type: str, data_dir: str, output_dir: str):
    """
    Validate model against competition dataset.
    
    Args:
        model_type: Type of model to validate ('vanilla', 'trajectory', 'feature')
        data_dir: Directory containing competition data
        output_dir: Directory to save validation results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load or generate competition data
    if os.path.exists(os.path.join(data_dir, 'csi_features.npy')):
        print("Loading competition data...")
        csi_features = np.load(os.path.join(data_dir, 'csi_features.npy'))
        positions = np.load(os.path.join(data_dir, 'positions.npy'))
    else:
        print("Generating synthetic competition data...")
        csi_features, positions = generate_synthetic_data(
            n_samples=1000,
            n_arrays=4,  # 4 remote antenna arrays
            n_elements=8,  # 8 elements per array
            n_frequencies=16  # 16 frequency bands
        )
        
        # Save generated data
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'csi_features.npy'), csi_features)
        np.save(os.path.join(data_dir, 'positions.npy'), positions)
    
    # Verify data dimensions
    print("\nData Dimensions:")
    print(f"CSI Features: {csi_features.shape}")
    print(f"Positions: {positions.shape}")
    
    assert csi_features.shape[1] == 4, "Must have 4 antenna arrays"
    assert csi_features.shape[2] == 8, "Must have 8 elements per array"
    assert csi_features.shape[3] == 16, "Must have 16 frequency bands"
    assert csi_features.shape[4] == 2, "Must have real and imaginary components"
    
    # Load appropriate model
    print("\nLoading model...")
    if model_type == 'vanilla':
        model = VanillaLocalizationModel()
        model_path = os.path.join(output_dir, 'vanilla_model')
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((csi_features, positions))
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
    elif model_type == 'trajectory':
        model = TrajectoryAwareModel()
        model_path = os.path.join(output_dir, 'trajectory_model')
        
        # Create sequences for trajectory model
        n_sequences = len(csi_features) - 10  # sequence length = 10
        feature_sequences = np.zeros((n_sequences, 10, *csi_features.shape[1:]))
        position_sequences = np.zeros((n_sequences, 10, 2))
        
        for i in range(n_sequences):
            feature_sequences[i] = csi_features[i:i + 10]
            position_sequences[i] = positions[i:i + 10]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'csi_features': feature_sequences,
            'prev_positions': position_sequences[:, :-1]  # All but last position
        })
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Update positions for metrics calculation
        positions = positions[10-1:]  # Use targets from end of sequences
        
    else:  # feature selection
        model = GrantFreeAccessModel()
        model_path = os.path.join(output_dir, 'feature_selection_model')
        
        # Create sequences for feature selection model
        n_sequences = len(csi_features) - 10  # sequence length = 10
        feature_sequences = np.zeros((n_sequences, 10, *csi_features.shape[1:]))
        position_sequences = np.zeros((n_sequences, 10, 2))
        
        for i in range(n_sequences):
            feature_sequences[i] = csi_features[i:i + 10]
            position_sequences[i] = positions[i:i + 10]
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices({
            'csi_features': feature_sequences,
            'prev_positions': position_sequences[:, :-1]  # All but last position
        })
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Update positions for metrics calculation
        positions = positions[10-1:]  # Use targets from end of sequences
    
    # Load weights if available
    if os.path.exists(model_path):
        print("Loading saved weights...")
        model.load_weights(model_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions = model.predict(dataset)
    
    if isinstance(predictions, dict):
        predictions = predictions['location']
    
    if model_type == 'trajectory':
        # For trajectory model, we only want the last prediction of each sequence
        predictions = predictions.reshape(-1, 2)  # Flatten predictions
        positions = positions.reshape(-1, 2)  # Flatten ground truth
        
        # Ensure same number of samples
        min_len = min(len(predictions), len(positions))
        predictions = predictions[:min_len]
        positions = positions[:min_len]
    
    # Calculate metrics
    metrics = calculate_competition_metrics(positions, predictions)
    
    print("\nValidation Results:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R90: {metrics['r90']:.4f}")
    print(f"Combined Score: {metrics['combined_metric']:.4f}")
    
    # Save metrics
    np.save(os.path.join(output_dir, 'validation_predictions.npy'), predictions)
    with open(os.path.join(output_dir, 'validation_metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")
    
    return metrics

if __name__ == "__main__":
    # Configuration
    model_types = ['vanilla', 'trajectory', 'feature']
    data_dir = 'data/competition'
    output_dir = 'outputs/validation'
    
    # Validate all models
    for model_type in model_types:
        print(f"\nValidating {model_type} model...")
        model_output_dir = os.path.join(output_dir, model_type)
        metrics = validate_model(model_type, data_dir, model_output_dir) 