import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from typing import Tuple
from models.feature_selection import GrantFreeAccessModel
from data_processing.trajectory_data import (
    TrajectoryDataGenerator,
    TrajectoryTFRecordHandler
)
from utils.competition_metrics import R90Metric, CombinedCompetitionMetric

def plot_channel_state(occupancy_mask, interference, save_path):
    """Plot channel state visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot channel occupancy
    im1 = ax1.imshow(occupancy_mask[0].numpy(), aspect='auto', cmap='YlOrRd')
    ax1.set_title('Channel Occupancy')
    ax1.set_xlabel('Frequency')
    ax1.set_ylabel('Antenna Array')
    plt.colorbar(im1, ax=ax1)
    
    # Plot interference
    im2 = ax2.imshow(interference[0, ..., 0].numpy(), aspect='auto', cmap='YlOrRd')
    ax2.set_title('Interference Estimation')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Antenna Array')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def load_competition_data(data_dir):
    """Load competition dataset."""
    # Load CSI features and positions
    csi_features = np.load(os.path.join(data_dir, 'csi_features.npy'))
    positions = np.load(os.path.join(data_dir, 'positions.npy'))
    
    # Generate random priority levels for training
    n_samples = len(positions)
    priorities = np.random.randint(0, 4, size=n_samples)  # 4 priority levels
    
    return csi_features, positions, priorities

def calculate_competition_metrics(actual, predicted):
    """Calculate competition metrics."""
    mae = np.mean(np.abs(actual - predicted))
    r90 = np.sum(np.abs(actual - predicted) <= np.percentile(np.abs(actual - predicted), 90)) / len(actual)
    combined_metric = 0.7 * mae + 0.3 * r90
    return {
        'mae': mae,
        'r90': r90,
        'combined_metric': combined_metric
    }

def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    output_dir: str,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the model with the given datasets."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get callbacks
    callbacks = model.get_callbacks(output_dir)
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    print("\nSaving training history...")
    np.save(os.path.join(output_dir, 'training_history.npy'), history.history)
    
    # Generate training plots
    print("\nGenerating training plots...")
    plot_training_history(history, output_dir)
    
    # Evaluate model
    print("\nEvaluating model...")
    val_predictions = model.predict(val_dataset)
    val_positions = np.concatenate([y['positions'].numpy() for x, y in val_dataset], axis=0)
    val_priorities = np.concatenate([y['occupancy_mask'].numpy() for x, y in val_dataset], axis=0)
    
    # Calculate metrics
    val_mae = np.mean(np.abs(val_predictions['positions'] - val_positions))
    val_r90 = np.percentile(np.abs(val_predictions['positions'] - val_positions), 90)
    val_combined_score = (val_mae + val_r90) / 2
    
    print(f"\nValidation Metrics:")
    print(f"MAE: {val_mae:.4f}")
    print(f"R90: {val_r90:.4f}")
    print(f"Combined Score: {val_combined_score:.4f}")
    
    # Calculate priority classification metrics
    val_priority_pred = np.argmax(val_predictions['occupancy_mask'], axis=-1)
    val_priority_true = np.argmax(val_priorities, axis=-1)
    val_priority_accuracy = np.mean(val_priority_pred == val_priority_true)
    print(f"Priority Classification Accuracy: {val_priority_accuracy:.4f}")
    
    # Generate evaluation plots
    plot_predictions(val_positions, val_predictions['positions'], output_dir)
    plot_error_distribution(val_positions, val_predictions['positions'], output_dir)
    
    # Plot feature importance
    plot_feature_importance(model, output_dir)
    
    print("\nTraining complete!")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the grant-free access model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing competition data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save model and results')
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'n_arrays': 4,
        'n_elements': 8,
        'n_freq': 16,
        'seq_len': 10
    }
    
    # Training configuration
    train_config = {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'learning_rate': 1e-4
    }
    
    # Load data
    print("Loading competition data...")
    csi_features, positions, priorities = load_competition_data(args.data_dir)
    
    # Create data generator
    print("Setting up data pipeline...")
    data_generator = TrajectoryDataGenerator(
        csi_features=csi_features,
        positions=positions,
        sequence_length=model_config['seq_len'],
        batch_size=train_config['batch_size'],
        validation_split=train_config['validation_split']
    )
    
    # Get datasets
    train_dataset = data_generator.get_train_dataset()
    val_dataset = data_generator.get_val_dataset()
    
    # Create model
    print("Creating model...")
    model = GrantFreeAccessModel(
        n_arrays=model_config['n_arrays'],
        n_elements=model_config['n_elements'],
        n_freq=model_config['n_freq'],
        seq_len=model_config['seq_len']
    )
    
    # Compile model
    print("Compiling model...")
    model.compile()
    
    # Train model
    model, history = train_model(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        epochs=train_config['epochs'],
        batch_size=train_config['batch_size'],
        learning_rate=train_config['learning_rate']
    ) 