import os
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from src.models.basic_localization import BasicLocalizationModel
from src.utils.data_loader import load_csi_data
import numpy as np
import time
from pathlib import Path
import argparse
from src.utils.metrics import calculate_mae, calculate_rmse, calculate_r2
import matplotlib.pyplot as plt

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ])

def train_basic_model(data_dir, batch_size=32, epochs=100):
    # Load and preprocess data
    train_dataset, val_dataset, test_dataset = load_csi_data(
        data_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=batch_size
    )
    
    # Create data augmentation
    data_augmentation = create_data_augmentation()
    
    # Apply augmentation to training data
    train_dataset = train_dataset.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimize dataset performance
    train_dataset = train_dataset.cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(tf.data.AUTOTUNE)
    
    # Get input shape from a batch
    for batch in train_dataset.take(1):
        input_shape = batch[0].shape[1:]
        break
    
    # Create and compile model
    model = BasicLocalizationModel(
        model_input_shape=input_shape,
        num_classes=2,
        l2_reg=0.01,
        dropout_rate=0.3
    )
    
    # Get training callbacks
    callbacks = model.get_training_callbacks()
    
    # Train model with timing
    start_time = time.time()
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    # Evaluate model
    test_loss, test_mae = model.evaluate(test_dataset)
    
    # Get predictions
    predictions = model.predict(test_dataset)
    true_labels = np.concatenate([y for x, y in test_dataset], axis=0)
    
    # Calculate metrics
    mae = calculate_mae(true_labels, predictions)
    rmse = calculate_rmse(true_labels, predictions)
    r2 = calculate_r2(true_labels, predictions)
    
    # Save metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_mae': float(test_mae),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'training_time': float(training_time),
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
    }
    
    # Create results directory if it doesn't exist
    results_dir = Path('results/basic_localization')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    np.save(results_dir / 'metrics.npy', metrics)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_history.png')
    plt.close()
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Train basic localization model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing CSI data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    
    args = parser.parse_args()
    
    metrics = train_basic_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print("\nTraining Results:")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Training Time: {metrics['training_time']:.2f} seconds")

if __name__ == '__main__':
    main() 