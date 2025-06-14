import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import logging
from datetime import datetime
import json
import glob
import random

from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import FeatureSelectionModel
from src.utils.data_loader import CompetitionDataLoader
from src.utils.metrics import calculate_combined_score
from src.config.competition_config import (
    TRAINING_CONFIG,
    VANILLA_MODEL_CONFIG,
    TRAJECTORY_MODEL_CONFIG,
    FEATURE_SELECTION_CONFIG,
    PATHS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'min_lr': 1e-6
}

# Model configurations
VANILLA_MODEL_CONFIG = {
    'dropout_rate': 0.3,
    'l2_reg': 0.01
}

TRAJECTORY_MODEL_CONFIG = {
    'sequence_length': 5,
    'dropout_rate': 0.3,
    'l2_reg': 0.01
}

FEATURE_SELECTION_CONFIG = {
    'dropout_rate': 0.3,
    'l2_reg': 0.01,
    'n_features_to_select': 32
}

# Data paths
DATA_DIR = 'data/csi_data'  # Updated data directory

def get_tfrecord_splits(data_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    tfrecord_files = glob.glob(os.path.join(data_dir, '*.tfrecords'))
    random.seed(seed)
    random.shuffle(tfrecord_files)
    n_total = len(tfrecord_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_files = tfrecord_files[:n_train]
    val_files = tfrecord_files[n_train:n_train+n_val]
    test_files = tfrecord_files[n_train+n_val:]
    return train_files, val_files, test_files

def setup_training(model_type):
    """Setup training environment and callbacks."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"outputs/training/{model_type}_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            min_lr=TRAINING_CONFIG['min_lr']
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(run_dir / "best_model.h5"),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(run_dir / "logs"),
            histogram_freq=1
        )
    ]
    
    return run_dir, callbacks

def train_vanilla_model(data, callbacks):
    """Train vanilla localization model."""
    logger.info("Training vanilla model...")
    features, labels = data['train']
    val_data = data['val']
    
    model = VanillaLocalizationModel(
        input_shape=features.shape[1:],
        num_classes=2,
        **VANILLA_MODEL_CONFIG
    )
    
    history = model.train(
        train_data=features,
        train_labels=labels,
        validation_data=val_data,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks
    )
    
    return model, history

def train_trajectory_model(data, callbacks):
    """Train trajectory-aware localization model."""
    logger.info("Training trajectory model...")
    features, labels = data['train']
    val_data = data['val']
    
    model = TrajectoryAwareLocalizationModel(
        input_shape=features.shape[1:],
        num_classes=2,
        **TRAJECTORY_MODEL_CONFIG
    )
    
    history = model.train(
        train_data=features,
        train_labels=labels,
        validation_data=val_data,
        epochs=TRAINING_CONFIG['epochs'],
        batch_size=TRAINING_CONFIG['batch_size'],
        callbacks=callbacks
    )
    
    return model, history

def train_feature_selection_model(data, callbacks):
    """Train feature selection model."""
    logger.info("Training feature selection model...")
    features, labels = data['train']
    val_data = data['val']
    
    model = FeatureSelectionModel(**FEATURE_SELECTION_CONFIG)
    # Use a smaller batch size (e.g. 8) and a smaller shuffle buffer (e.g. 100) to reduce RAM usage.
    train_dataset = data['train'].shuffle(buffer_size=100, seed=42).batch(8)
    val_dataset = data['val'].batch(8)
    history = model.fit(train_dataset, epochs=FEATURE_SELECTION_CONFIG['epochs'], callbacks=callbacks, validation_data=val_dataset)
    return model, history

def evaluate_model(model, test_data, model_type):
    """Evaluate model on test data."""
    logger.info(f"Evaluating {model_type} model...")
    test_features, test_labels = test_data
    test_loss, test_mae = model.evaluate(test_features, test_labels)
    predictions = model.predict(test_features)
    combined_score = calculate_combined_score(test_labels, predictions)
    
    metrics = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'combined_score': float(combined_score)
    }
    
    logger.info(f"{model_type} model metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    return metrics

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load and preprocess data
    logger.info("Loading data...")
    train_files, val_files, test_files = get_tfrecord_splits(DATA_DIR)
    
    data_loader = CompetitionDataLoader(DATA_DIR, batch_size=8)
    data_loader.train_pattern = train_files
    data_loader.val_pattern = val_files
    data_loader.test_pattern = test_files
    train_dataset, val_dataset, test_dataset = data_loader.prepare_data()
    
    # Convert tf.data.Dataset to numpy arrays for training
    def dataset_to_numpy(dataset):
        features, labels = [], []
        for x, y in dataset:
            features.append(x.numpy())
            labels.append(y.numpy())
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)
    
    train_features, train_labels = dataset_to_numpy(train_dataset)
    val_features, val_labels = dataset_to_numpy(val_dataset)
    test_features, test_labels = dataset_to_numpy(test_dataset)
    
    data = {
        'train': (train_features, train_labels),
        'val': (val_features, val_labels),
        'test': (test_features, test_labels)
    }
    
    # Train and save all models
    model_types = ['vanilla', 'trajectory', 'feature_selection']
    trained_models = {}
    
    for model_type in model_types:
        logger.info(f"\nProcessing {model_type} model...")
        
        # Setup training
        run_dir, callbacks = setup_training(model_type)
        
        # Train model
        if model_type == 'vanilla':
            model, history = train_vanilla_model(data, callbacks)
        elif model_type == 'trajectory':
            model, history = train_trajectory_model(data, callbacks)
        else:  # feature_selection
            model, history = train_feature_selection_model(data, callbacks)
        
        # Evaluate model
        metrics = evaluate_model(model, data['test'], model_type)
        
        # Save model
        model_path = models_dir / f"{model_type}_model.h5"
        model.save_model(str(model_path))
        logger.info(f"Saved {model_type} model to {model_path}")
        
        # Save metrics
        metrics_path = run_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        trained_models[model_type] = {
            'model': model,
            'metrics': metrics,
            'path': model_path
        }
    
    # Verify all models are saved
    logger.info("\nVerifying saved models...")
    for model_type in model_types:
        model_path = models_dir / f"{model_type}_model.h5"
        if model_path.exists():
            logger.info(f"✓ {model_type} model saved successfully")
        else:
            logger.error(f"✗ {model_type} model not found at {model_path}")
    
    logger.info("\nModel regeneration complete!")

if __name__ == "__main__":
    main() 