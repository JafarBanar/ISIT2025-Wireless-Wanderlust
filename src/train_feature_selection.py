import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
from src.isit2025.models.feature_selection import GrantFreeAccessModel
from src.isit2025.data_processing.trajectory_data import (
    TrajectoryDataGenerator,
    TrajectoryTFRecordHandler
)
from src.isit2025.utils.visualization import (
    plot_training_history,
    plot_predictions,
    plot_error_distribution,
    plot_feature_importance
)

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

def train_model(model_config, train_config, data_dir, output_dir):
    """Train the grant-free random access model with channel sensing."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading competition data...")
    csi_features, positions, priorities = load_competition_data(data_dir)
    
    # Create data generator
    print("Setting up data pipeline...")
    data_generator = TrajectoryDataGenerator(
        csi_features=csi_features,
        positions=positions,
        sequence_length=model_config['sequence_length'],
        batch_size=train_config['batch_size'],
        validation_split=train_config['validation_split']
    )
    
    # Create model
    print("Creating model...")
    model = GrantFreeAccessModel(
        n_features_to_select=model_config['n_features_to_select'],
        n_priority_levels=model_config['n_priority_levels'],
        feature_dim=model_config['feature_dim'],
        dropout_rate=model_config['dropout_rate'],
        l2_reg=model_config['l2_reg']
    )
    
    # Compile model with competition metrics
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'location': tf.keras.losses.MeanSquaredError(),
            'priority': tf.keras.losses.SparseCategoricalCrossentropy(),
            'channel_occupancy': tf.keras.losses.BinaryCrossentropy()
        },
        loss_weights={
            'location': 1.0,
            'priority': 0.3,
            'channel_occupancy': 0.2
        },
        metrics={
            'location': [
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                R90Metric(name='r90'),
                CombinedCompetitionMetric(name='combined_score')
            ],
            'priority': ['accuracy'],
            'channel_occupancy': ['accuracy']
        }
    )
    
    # Get callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_location_combined_score',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_location_combined_score',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model'),
            monitor='val_location_combined_score',
            save_best_only=True,
            save_format='tf',
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        data_generator.get_train_dataset(),
        validation_data=data_generator.get_val_dataset(),
        epochs=train_config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training plots
    print("Generating training plots...")
    plot_training_history(history, output_dir)
    
    # Evaluate on validation set
    print("Evaluating model...")
    val_dataset = data_generator.get_val_dataset()
    val_predictions = model.predict(val_dataset)
    
    # Get actual positions from validation set
    val_positions = np.concatenate([y['location'].numpy() for x, y in val_dataset], axis=0)
    
    # Calculate competition metrics
    metrics = calculate_competition_metrics(
        val_positions,
        val_predictions['location']
    )
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate evaluation plots
    plot_predictions(val_positions, val_predictions['location'], output_dir)
    plot_error_distribution(val_positions, val_predictions['location'], output_dir)
    
    # Plot feature importance
    feature_importance = np.mean(val_predictions['selection_mask'], axis=0)
    plot_feature_importance(
        feature_importance,
        save_path=os.path.join(output_dir, 'feature_importance.png')
    )
    
    # Plot channel state
    plot_channel_state(
        val_predictions['channel_occupancy'],
        val_predictions['interference'],
        os.path.join(output_dir, 'channel_state.png')
    )
    
    # Save TFRecord for competition submission
    print("Saving TFRecord for submission...")
    tfrecord_handler = TrajectoryTFRecordHandler(
        sequence_length=model_config['sequence_length']
    )
    tfrecord_handler.write_tfrecord(
        csi_features,
        positions,
        os.path.join(output_dir, 'submission.tfrecord')
    )
    
    print("Training complete!")
    print("\nValidation Metrics:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R90: {metrics['r90']:.4f}")
    print(f"Combined Score: {metrics['combined_metric']:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Model configuration
    model_config = {
        'sequence_length': 10,
        'n_features_to_select': 32,
        'n_priority_levels': 4,
        'feature_dim': 128,
        'dropout_rate': 0.3,
        'l2_reg': 0.01,
        'energy_threshold': 0.1,  # Added for channel sensing
        'interference_window': 5   # Added for channel sensing
    }
    
    # Training configuration
    train_config = {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2
    }
    
    # Paths
    data_dir = 'data/competition'
    output_dir = 'outputs/feature_selection'
    
    # Train model
    model, history = train_model(
        model_config=model_config,
        train_config=train_config,
        data_dir=data_dir,
        output_dir=output_dir
    ) 