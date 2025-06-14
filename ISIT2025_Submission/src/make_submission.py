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

    # Use only the specified test set file for submission
    test_tfrecord = os.path.join(data_dir, "dichasus-cf02.tfrecords")
    if not os.path.exists(test_tfrecord):
        raise ValueError(f"Test TFRecord file not found: {test_tfrecord}")
    print(f"Using test set: {test_tfrecord}")

    # Create dataset from the test file and use all its samples
    dataset = tf.data.TFRecordDataset([test_tfrecord])
    data_loader = CSIDataLoader(test_tfrecord)  # Use the test file for parsing function
    test_dataset = dataset.map(data_loader._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)  # No .take(40), use all samples
    
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
        'basic': VanillaLocalizationModel(input_shape=input_shape, num_classes=num_classes)
    }
    
    # Load model weights
    model_paths = {
        'basic': 'models/basic_localization.h5'
    }
    
    to_remove = []
    for name, model in models.items():
        if os.path.exists(model_paths[name]):
            print(f"Loading {name} model weights...")
            try:
                model.model = tf.keras.models.load_model(
                    model_paths[name],
                    custom_objects={
                        'VanillaLocalizationModel': VanillaLocalizationModel,
                        'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
                        'GrantFreeAccessModel': GrantFreeAccessModel
                    },
                    compile=False
                )
            except Exception as e:
                print(f"Skipping {name} model due to incompatible weights or architecture: {e}")
                to_remove.append(name)
        else:
            print(f"Warning: {name} model weights not found at {model_paths[name]}")
            to_remove.append(name)
    for name in to_remove:
        models.pop(name)
    
    # Remove 'ensemble' from models if present, to avoid prediction errors
    if 'ensemble' in models:
        print('Skipping ensemble model due to incompatible input shape.')
        models.pop('ensemble')
    
    if not models:
        raise ValueError("No model weights found. Please train models first.")
    
    # Make predictions
    print("Making predictions...")
    all_predictions = []
    
    # Create a buffer for sequence data
    sequence_buffer = []
    sequence_length = 10  # Match the model's sequence_length parameter
    
    # Only process the basic model in the prediction loop
    for batch in test_dataset:
        csi = batch['csi']
        batch_predictions = []
        for name, model in models.items():
            print(f"Calling {name} model with input shape: {csi.shape}")
            pred = model.predict(csi)
            batch_predictions.append(pred)
        if batch_predictions:
            # Debug print shapes
            print("\nPrediction shapes before alignment:")
            for i, pred in enumerate(batch_predictions):
                print(f"Model {i} prediction shape: {pred.shape}")
            # Align prediction shapes
            aligned_predictions = []
            for i, pred in enumerate(batch_predictions):
                if pred.ndim > 2:
                    pred = np.squeeze(pred)
                elif pred.ndim == 3 and pred.shape[1] == 1:
                    pred = np.squeeze(pred, axis=1)
                elif pred.ndim != 2:
                    raise ValueError(f"Unexpected prediction shape from model {i}: {pred.shape}")
                if pred.shape[-1] != 2:
                    raise ValueError(f"Model {i} prediction has wrong number of coordinates: {pred.shape}")
                aligned_predictions.append(pred)
            print("\nPrediction shapes after alignment:")
            for i, pred in enumerate(aligned_predictions):
                print(f"Model {i} aligned prediction shape: {pred.shape}")
            ensemble_pred = np.mean(aligned_predictions, axis=0)
            all_predictions.append(ensemble_pred)
    
    # Combine predictions
    predictions = np.concatenate(all_predictions, axis=0)

    # Create submission DataFrame in required format
    submission_df = pd.DataFrame({
        'ID': np.arange(1, len(predictions) + 1),  # Use 1-based indexing for all predictions
        'x': predictions[:, 0],
        'y': predictions[:, 1]
    })
    
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
