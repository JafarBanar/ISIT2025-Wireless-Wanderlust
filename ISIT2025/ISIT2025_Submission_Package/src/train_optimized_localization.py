import os
import time
import numpy as np
import tensorflow as tf
from src.models.optimized_localization import OptimizedLocalizationModel
from src.utils.data_loader import load_csi_data
from src.utils.metrics import calculate_metrics
from src.utils.logging_utils import get_logger
from datetime import datetime

logger = get_logger(__name__)

def create_data_augmentation_layer():
    """Create a data augmentation pipeline for CSI data"""
    return tf.keras.Sequential([
        # Spatial augmentations
        tf.keras.layers.RandomRotation(factor=0.02, fill_mode='nearest'),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.05, 0.05),
            width_factor=(-0.05, 0.05),
            fill_mode='nearest'
        ),
        # Intensity augmentations
        tf.keras.layers.GaussianNoise(0.01),
        tf.keras.layers.RandomBrightness(factor=0.1),
    ])

def train_optimized_model(data_dir, batch_size=32, epochs=100):
    """
    Train the optimized localization model.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
    """
    # Load and preprocess data
    logger.info("Loading and preprocessing data")
    train_dataset, val_dataset, test_dataset = load_csi_data(
        data_dir=data_dir,
        batch_size=batch_size
    )
    
    # Create data augmentation layer
    augmentation_layer = create_data_augmentation_layer()
    
    # Initialize model
    logger.info("Initializing optimized model")
    model = OptimizedLocalizationModel()
    
    # Apply augmentation to the training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)
    
    # Ensure validation dataset is properly batched and prefetched
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Train model
    start_time = time.time()
    logger.info("Starting model training")
    history = model.train(
        train_dataset,
        val_dataset,
        epochs=epochs,
        batch_size=batch_size
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    logger.info("Evaluating model")
    test_metrics = model.evaluate(test_dataset)
    
    # Calculate additional metrics
    logger.info("Calculating additional metrics")
    y_true = []
    y_pred = []
    
    for x, y in test_dataset:
        y_true.extend(y.numpy())
        y_pred.extend(model.predict(x))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = calculate_metrics(y_true, y_pred)
    metrics.update({
        'test_loss': test_metrics['loss'],
        'test_mae': test_metrics['mae'],
        'training_time': training_time,
        'history': history.history
    })
    
    # Save results
    os.makedirs('results/optimized_localization', exist_ok=True)
    np.save('results/optimized_localization/metrics.npy', metrics)
    model.save('results/optimized_localization/model.keras')
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logger.info(f"Test MAE: {metrics['test_mae']:.4f}")
    
    return metrics

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train optimized localization model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs/fit', exist_ok=True)
    
    train_optimized_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    ) 