import os
import sys
import numpy as np
import tensorflow as tf

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.models.trajectory_model import TrajectoryAwareModel
from src.data_processing.trajectory_data import TrajectoryDataGenerator
from src.utils.visualization import (
    plot_training_history,
    plot_predictions,
    plot_trajectory,
    plot_error_distribution
)
from src.utils.competition_metrics import (
    calculate_competition_metrics,
    R90Metric,
    CombinedCompetitionMetric
)
import json

def train_model(model_config, train_config, data_dir, output_dir):
    """Train the trajectory-aware localization model."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading competition data...")
    csi_features = np.load(os.path.join(data_dir, 'csi_features.npy'))
    positions = np.load(os.path.join(data_dir, 'positions.npy'))
    
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
    model = TrajectoryAwareModel(
        lstm_units=model_config['lstm_units'],
        attention_units=model_config['attention_units'],
        dropout_rate=model_config['dropout_rate']
    )
    
    # Compile model
    print("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['learning_rate']),
        loss={'location': 'mse'},
        metrics={
            'location': [
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                R90Metric(name='r90'),
                CombinedCompetitionMetric(name='combined_score')
            ]
        }
    )
    
    # Get callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_location_mae',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_location_mae',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model'),
            monitor='val_location_mae',
            save_best_only=True,
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
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate evaluation plots
    plot_predictions(val_positions, val_predictions['location'], output_dir)
    plot_error_distribution(val_positions, val_predictions['location'], output_dir)
    plot_trajectory(val_positions, val_predictions['location'], output_dir)
    
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
        'lstm_units': 256,
        'attention_units': 128,
        'dropout_rate': 0.4
    }
    
    # Training configuration
    train_config = {
        'batch_size': 32,
        'epochs': 200,
        'learning_rate': 1e-3,
        'validation_split': 0.2
    }
    
    # Paths
    data_dir = 'data/competition'
    output_dir = 'outputs/trajectory_model'
    
    # Train model
    model, history = train_model(
        model_config=model_config,
        train_config=train_config,
        data_dir=data_dir,
        output_dir=output_dir
    ) 