import argparse
import os
import tensorflow as tf
from datetime import datetime
from pathlib import Path

from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import FeatureSelectionModel
from utils.data_processing import (
    load_csi_data,
    preprocess_csi_data,
    create_sequence_data,
    create_data_generator,
    split_data
)
from src.utils.metrics import calculate_combined_score

def parse_args():
    parser = argparse.ArgumentParser(description='Train ISIT 2025 localization models')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vanilla', 'trajectory', 'feature_selection'],
                      help='Type of model to train')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--sequence_length', type=int, default=5,
                      help='Sequence length for trajectory model')
    parser.add_argument('--output_dir', type=str, default='results/isit2025',
                      help='Directory to save results')
    return parser.parse_args()

def setup_training(model_type, output_dir):
    """Setup training environment and create output directories."""
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / model_type / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    log_dir = run_dir / 'logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    
    # Setup model checkpoint
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_dir / 'best_model.keras',
        monitor='val_loss',
        save_best_only=True
    )
    
    return run_dir, [tensorboard_callback, checkpoint_callback]

def train_vanilla_model(data, args, callbacks):
    """Train vanilla localization model."""
    features, labels = data['train']
    val_data = data['val']
    
    model = VanillaLocalizationModel(
        input_shape=features.shape[1:],
        num_classes=2
    )
    
    history = model.train(
        train_data=features,
        train_labels=labels,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return model, history

def train_trajectory_model(data, args, callbacks):
    """Train trajectory-aware localization model."""
    features, labels = data['train']
    val_data = data['val']
    
    # Create sequence data
    seq_features, seq_labels = create_sequence_data(
        features, labels, args.sequence_length
    )
    val_seq_features, val_seq_labels = create_sequence_data(
        val_data[0], val_data[1], args.sequence_length
    )
    
    model = TrajectoryAwareLocalizationModel(
        input_shape=features.shape[1:],
        sequence_length=args.sequence_length
    )
    
    history = model.train(
        train_data=seq_features,
        train_labels=seq_labels,
        validation_data=(val_seq_features, val_seq_labels),
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return model, history

def train_feature_selection_model(data, args, callbacks):
    """Train feature selection model with grant-free random access."""
    features, labels = data['train']
    val_data = data['val']
    
    model = FeatureSelectionModel(
        input_shape=features.shape[1:],
        num_features=features.shape[-1]
    )
    
    history = model.train(
        train_data=features,
        train_labels=labels,
        validation_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return model, history

def main():
    args = parse_args()
    
    # Load and preprocess data
    data_path = Path(args.data_dir) / 'csi_data.npz'
    features, labels = load_csi_data(str(data_path))
    features, labels = preprocess_csi_data(features, labels)
    data = split_data(features, labels)
    
    # Setup training environment
    run_dir, callbacks = setup_training(args.model_type, args.output_dir)
    
    # Train model
    if args.model_type == 'vanilla':
        model, history = train_vanilla_model(data, args, callbacks)
    elif args.model_type == 'trajectory':
        model, history = train_trajectory_model(data, args, callbacks)
    else:  # feature_selection
        model, history = train_feature_selection_model(data, args, callbacks)
    
    # Evaluate on test set
    test_features, test_labels = data['test']
    test_loss, test_mae = model.evaluate(test_features, test_labels)
    
    # Calculate combined score
    predictions = model.predict(test_features)
    combined_score = calculate_combined_score(test_labels, predictions)
    
    # Save results
    results = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'combined_score': float(combined_score)
    }
    
    with open(run_dir / 'results.txt', 'w') as f:
        for metric, value in results.items():
            f.write(f'{metric}: {value}\n')
    
    print(f"Training completed. Results saved to {run_dir}")

if __name__ == '__main__':
    main() 