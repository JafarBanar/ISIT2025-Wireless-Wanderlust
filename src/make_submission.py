import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
import glob
import argparse


# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.data_loader import CSIDataLoader
from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import GrantFreeAccessModel, FeatureSelectionModel
from src.utils.competition_metrics import calculate_competition_metrics

def prepare_submission(data_dir: str = 'data/csi_data', output_dir: str = 'submissions'):
    """Prepare competition submission using ensemble of models."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    print("Loading test data...")

    # Get list of TFRecord files
    tfrecord_files = glob.glob(os.path.join(data_dir, "*.tfrecords"))
    if not tfrecord_files:
        raise ValueError(f"No TFRecord files found in {data_dir}")
    print(f"Found {len(tfrecord_files)} TFRecord files")
    
    # Create dataset from all files
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    data_loader = CSIDataLoader(tfrecord_files[0])  # Use first file for parsing function
    test_dataset = dataset.map(data_loader._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # Explicitly set batch size to 32
    
    # Load models
    print("Loading models...")
    input_shape = (32, 1024, 2)  # CSI data shape
    num_classes = 2  # x, y coordinates
    
    # Instantiate feature extractor for ensemble model
    feature_extractor = FeatureSelectionModel(input_shape=input_shape, num_classes=num_classes)
    # We'll use only up to the first dense layer (512-dim)
    def extract_features(x):
        x = feature_extractor.conv1(x)
        x = feature_extractor.bn1(x, training=False)
        x = feature_extractor.pool1(x)
        x = feature_extractor.dropout1(x, training=False)
        x = feature_extractor.conv2(x)
        x = feature_extractor.bn2(x, training=False)
        x = feature_extractor.pool2(x)
        x = feature_extractor.dropout2(x, training=False)
        x = feature_extractor.conv3(x)
        x = feature_extractor.bn3(x, training=False)
        x = feature_extractor.pool3(x)
        x = feature_extractor.dropout3(x, training=False)
        x = feature_extractor.flatten(x)
        x = feature_extractor.dense1(x)
        x = feature_extractor.dense_bn1(x, training=False)
        x = feature_extractor.dense_dropout1(x, training=False)
        return x
    
    models = {
        'basic': VanillaLocalizationModel(input_shape=input_shape, num_classes= ( num_classes )),
        'ensemble': TrajectoryAwareLocalizationModel(sequence_length= ( 5 )),  # Uses default l2_reg and dropout_rate
        'trajectory': TrajectoryAwareLocalizationModel(sequence_length= ( 5 ))  # Uses default l2_reg and dropout_rate
    }
    
    # Load model weights
    model_paths = {
        'basic': 'models/basic_localization.h5',
        'ensemble': 'models/ensemble_model.h5',
        'trajectory': 'models/trajectory_model.h5'
    }
    
    to_remove = []
    for name, model in models.items():
        if os.path.exists(model_paths[name]):
            print(f"Loading {name} model weights...")
            model.model = tf.keras.models.load_model(
                model_paths[name],
                custom_objects={
                    'VanillaLocalizationModel': VanillaLocalizationModel,
                    'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
                    'GrantFreeAccessModel': GrantFreeAccessModel
                },
                compile=False
            )
        else:
            print(f"Warning: {name} model weights not found at {model_paths[name]}")
            to_remove.append(name)
    for name in to_remove:
        models.pop(name)
    
    if not models:
        raise ValueError("No model weights found. Please train models first.")
    
    # Make predictions
    print("Making predictions...")
    all_predictions = []
    
    # Create a buffer for sequence data
    sequence_buffer = []
    sequence_length = 5  # Match the model's sequence_length parameter
    
    for batch in test_dataset:
        csi = batch['csi']
        
        # Update sequence buffer for both trajectory and ensemble models
        sequence_buffer.append(csi)
        if len(sequence_buffer) > sequence_length:
            sequence_buffer.pop(0)
        
        batch_predictions = []
        for name, model in models.items():
            if name in ['trajectory', 'ensemble']:
                # Both trajectory and ensemble models need sequences
                if len(sequence_buffer) < sequence_length:
                    # Pad with zeros if we don't have enough history
                    padding = [tf.zeros_like(csi) for _ in range(sequence_length - len(sequence_buffer))]
                    sequence_data = tf.stack(sequence_buffer + padding, axis=1)
                else:
                    sequence_data = tf.stack(sequence_buffer[-sequence_length:], axis=1)
                
                if name == 'ensemble':
                    # For ensemble model, we need to reshape for CSI processing
                    batch_size = tf.shape(sequence_data)[0]
                    # Reshape to (batch_size * sequence_length, 32, 1024, 2)
                    flattened_data = tf.reshape(sequence_data, [-1, 32, 1024, 2])
                    print(f"\n{name.capitalize()} model input shape (flattened): {flattened_data.shape}")
                    
                    # Get predictions for each timestep
                    timestep_preds = model.predict(flattened_data)
                    
                    # Reshape back to (batch_size, sequence_length, 2)
                    pred = tf.reshape(timestep_preds, [batch_size, sequence_length, 2])
                    
                    # Take mean across sequence dimension for final prediction
                    pred = tf.reduce_mean(pred, axis=1)
                else:
                    # For trajectory model, keep sequence format
                    print(f"\n{name.capitalize()} model input shape: {sequence_data.shape}")
                    pred = model.predict(sequence_data)
                
                batch_predictions.append(pred)
            else:
                # Basic model processes single frame
                print(f"Calling {name} model with input shape: {csi.shape}")
                pred = model.predict(csi)
                batch_predictions.append(pred)
        
        if batch_predictions:
            # Debug print shapes
            print("\nPrediction shapes before alignment:")
            for i, pred in enumerate(batch_predictions):
                print(f"Model {i} prediction shape: {pred.shape}")
            
            # Verify all predictions have the same batch size
            batch_sizes = [pred.shape[0] for pred in batch_predictions]
            if len(set(batch_sizes)) > 1:
                raise ValueError(f"Inconsistent batch sizes across models: {batch_sizes}")
            
            # Align prediction shapes
            aligned_predictions = []
            for i, pred in enumerate(batch_predictions):
                # Convert to numpy if it's a tensor
                if isinstance(pred, tf.Tensor):
                    pred = pred.numpy()
                
                # Handle different prediction shapes
                if pred.ndim == 3 and pred.shape[1] == 1:
                    pred = np.squeeze(pred, axis=1)
                elif pred.ndim > 2:
                    raise ValueError(f"Unexpected prediction shape from model {i}: {pred.shape}")
                
                # Verify final shape is (batch_size, 2)
                if pred.shape[-1] != 2:
                    raise ValueError(f"Model {i} prediction has wrong number of coordinates: {pred.shape}")
                
                aligned_predictions.append(pred)
            
            # Debug print aligned shapes
            print("\nPrediction shapes after alignment:")
            for i, pred in enumerate(aligned_predictions):
                print(f"Model {i} aligned prediction shape: {pred.shape}")
            
            ensemble_pred = np.mean(aligned_predictions, axis=0)
            all_predictions.append(ensemble_pred)
    
    # Combine predictions
    predictions = np.concatenate(all_predictions, axis=0)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame(predictions, columns=['x', 'y'])
    
    # Save submission
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to {submission_path}")
    
    # Print submission info
    print("\nSubmission Info:")
    print(f"Number of predictions: {len(submission_df)}")
    print(f"Shape: {submission_df.shape}")
    print("\nFirst few predictions:")
    print(submission_df.head())
    
    return submission_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to TFRecord directory")
    parser.add_argument("--output_dir", type=str, default="submissions", help="Where to save the submission CSV")
    args = parser.parse_args()

    submission_path = prepare_submission(data_dir=args.data_dir, output_dir=args.output_dir)
    print(f"\nTo submit to competition, use:")
    print(f"kaggle competitions submit -c isit-2025-wireless-wanderlust -f {submission_path} -m \"Your submission message\"")
