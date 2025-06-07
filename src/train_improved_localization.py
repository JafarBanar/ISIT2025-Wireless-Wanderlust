import tensorflow as tf
import numpy as np
from pathlib import Path
import time
from src.utils.data_loader import load_csi_data
from src.models.improved_localization import create_improved_localization_model, get_training_callbacks
import argparse

def train_improved_model(data_dir, batch_size=32, epochs=100):
    """
    Train the improved localization model with:
    - Consistent data pipeline
    - Proper batching
    - Optimized performance
    """
    # Create results directory
    results_dir = Path("results/improved_localization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    print("Loading data...")
    train_ds, val_ds, test_ds = load_csi_data(
        data_dir=data_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=batch_size
    )
    
    # Optimize dataset performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)
    
    # Get input shape from one batch
    for X_batch, _ in train_ds.take(1):
        input_shape = X_batch.shape[1:]
    
    # Create and compile model
    print("Creating model...")
    model = create_improved_localization_model(
        input_shape=input_shape,
        num_classes=2,
        l2_reg=1e-4,
        dropout_rate=0.3
    )
    
    # Get callbacks
    callbacks = get_training_callbacks()
    
    # Train model
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = model.evaluate(test_ds, return_dict=True)
    
    # Save results
    metrics = {
        'test_loss': test_results['loss'],
        'test_mae': test_results['mae'],
        'history': history.history,
        'training_time': training_time
    }
    np.save(results_dir / 'metrics.npy', metrics)
    
    # Save final model
    model.save(results_dir / 'final_model.keras')
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Test Loss: {test_results['loss']:.4f}")
    print(f"Test MAE: {test_results['mae']:.4f}")
    
    return model, metrics

def main():
    parser = argparse.ArgumentParser(description='Train improved CSI localization model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing TFRecord files')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    args = parser.parse_args()
    
    train_improved_model(args.data_dir, args.batch_size, args.epochs)

if __name__ == "__main__":
    main() 