import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from src.utils.data_loader import load_csi_data
from src.models.basic_localization import create_basic_localization_model
import argparse
from tqdm import tqdm
import time

def load_training_history():
    """Load training history from saved metrics."""
    metrics_path = Path("results/basic_localization/metrics.npy")
    if metrics_path.exists():
        try:
            metrics = np.load(metrics_path, allow_pickle=True).item()
            # Check if we have the new format with test metrics
            if 'test_loss' in metrics and 'test_mae' in metrics:
                print("Found test metrics:")
                print(f"Test Loss: {metrics['test_loss']:.4f}")
                print(f"Test MAE: {metrics['test_mae']:.4f}")
                if 'history' in metrics:
                    print("\nFound training history in metrics file.")
                    return metrics['history']
            return None
        except Exception as e:
            print(f"Error loading training history: {e}")
            return None
    else:
        print("No training history file found at:", metrics_path)
        return None

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/basic_localization/training_history.png')
    plt.close()

def evaluate_model_on_test_set(data_dir):
    """Evaluate model on test set and generate detailed metrics."""
    # Load test data
    print(f"Loading data from {data_dir}...")
    start_time = time.time()
    _, _, test_ds = load_csi_data(data_dir)
    load_time = time.time() - start_time
    print(f"Data loading completed in {load_time:.2f} seconds")
    
    # Load best model
    model_path = Path("results/basic_localization/best_model.keras")
    if not model_path.exists():
        print("Error: Best model file not found at:", model_path)
        return None
    
    try:
        model = tf.keras.models.load_model(str(model_path))
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    start_time = time.time()
    test_results = model.evaluate(test_ds, return_dict=True)
    eval_time = time.time() - start_time
    print(f"Model evaluation completed in {eval_time:.2f} seconds")
    
    # Generate predictions and get true labels
    predictions = []
    true_labels = []
    
    print("\nGenerating predictions and collecting true labels...")
    start_time = time.time()
    
    # Count total batches
    total_batches = sum(1 for _ in test_ds)
    test_ds = test_ds.repeat(1)  # Reset the dataset
    
    # Create progress bar
    pbar = tqdm(total=total_batches, desc="Processing batches", unit="batch")
    
    for x, y in test_ds:
        pred = model.predict(x, verbose=0)
        predictions.extend(pred)
        true_labels.extend(y.numpy())
        pbar.update(1)
    
    pbar.close()
    
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    pred_time = time.time() - start_time
    print(f"Prediction generation completed in {pred_time:.2f} seconds")
    
    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - true_labels))
    mse = np.mean((predictions - true_labels) ** 2)
    rmse = np.sqrt(mse)
    
    # Save metrics
    metrics = {
        'test_loss': test_results['loss'],
        'test_mae': test_results['mae'],
        'test_mse': mse,
        'test_rmse': rmse,
        'processing_times': {
            'data_loading': load_time,
            'model_evaluation': eval_time,
            'prediction_generation': pred_time,
            'total_time': load_time + eval_time + pred_time
        }
    }
    
    np.save('results/basic_localization/test_metrics.npy', metrics)
    
    return metrics, predictions, true_labels

def analyze_error_distribution(predictions, true_labels):
    """Analyze error distribution and generate visualizations."""
    print("\nGenerating error analysis visualizations...")
    start_time = time.time()
    
    errors = predictions - true_labels
    
    plt.figure(figsize=(12, 4))
    
    # Plot error distribution
    plt.subplot(1, 2, 1)
    sns.histplot(errors.flatten(), bins=50)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Count')
    
    # Plot error vs prediction
    plt.subplot(1, 2, 2)
    plt.scatter(predictions, errors, alpha=0.5)
    plt.title('Error vs Prediction')
    plt.xlabel('Prediction')
    plt.ylabel('Error')
    
    plt.tight_layout()
    plt.savefig('results/basic_localization/error_analysis.png')
    plt.close()
    
    viz_time = time.time() - start_time
    print(f"Visualization generation completed in {viz_time:.2f} seconds")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze CSI localization model results')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing TFRecord files')
    args = parser.parse_args()
    
    total_start_time = time.time()
    
    # Create results directory if it doesn't exist
    Path("results/basic_localization").mkdir(parents=True, exist_ok=True)
    
    # Load and plot training history
    history = load_training_history()
    if history:
        print("Plotting training history...")
        plot_training_history(history)
    else:
        print("Skipping training history plotting...")
    
    # Evaluate model and generate detailed metrics
    print("\nStarting model evaluation...")
    result = evaluate_model_on_test_set(args.data_dir)
    if result:
        metrics, predictions, true_labels = result
        print("\nTest Set Metrics:")
        for metric, value in metrics.items():
            if metric != 'processing_times':
                print(f"{metric}: {value:.4f}")
        
        print("\nProcessing Times:")
        for stage, time_taken in metrics['processing_times'].items():
            print(f"{stage}: {time_taken:.2f} seconds")
        
        # Generate error analysis
        analyze_error_distribution(predictions, true_labels)
    else:
        print("Skipping model evaluation...")
    
    total_time = time.time() - total_start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main() 