import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from src.utils.data_loader import load_csi_data
import argparse
import os

def load_metrics(model_type):
    """Load metrics for a specific model type."""
    metrics_path = Path(f"results/{model_type}/metrics.npy")
    if metrics_path.exists():
        try:
            metrics = np.load(metrics_path, allow_pickle=True).item()
            print(f"\nLoaded metrics for {model_type}:")
            print(f"Available keys: {list(metrics.keys())}")
            return metrics
        except Exception as e:
            print(f"Error loading metrics for {model_type}: {str(e)}")
            return None
    print(f"No metrics file found for {model_type} at {metrics_path}")
    return None

def plot_comparison(basic_metrics, improved_metrics):
    """Plot comparison between basic and improved models."""
    plt.figure(figsize=(15, 10))
    
    # Plot test metrics comparison
    plt.subplot(2, 2, 1)
    metrics = ['test_loss', 'test_mae']
    basic_values = [basic_metrics[m] for m in metrics]
    improved_values = [improved_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, basic_values, width, label='Basic Model')
    plt.bar(x + width/2, improved_values, width, label='Improved Model')
    plt.title('Test Metrics Comparison')
    plt.xticks(x, metrics)
    plt.ylabel('Value')
    plt.legend()
    
    # Plot training time comparison if available
    plt.subplot(2, 2, 2)
    if 'training_time' in improved_metrics:
        times = [improved_metrics['training_time']]
        labels = ['Improved Model']
        if 'training_time' in basic_metrics:
            times.insert(0, basic_metrics['training_time'])
            labels.insert(0, 'Basic Model')
        plt.bar(labels, times)
        plt.title('Training Time Comparison')
        plt.ylabel('Time (seconds)')
    else:
        plt.text(0.5, 0.5, 'Training time not available', 
                horizontalalignment='center', verticalalignment='center')
        plt.title('Training Time Comparison')
    
    # Plot history if available
    if 'history' in basic_metrics and 'history' in improved_metrics:
        history = basic_metrics['history']
        improved_history = improved_metrics['history']
        
        # Plot loss history
        plt.subplot(2, 2, 3)
        if 'loss' in history and 'val_loss' in history:
            plt.plot(history['loss'], label='Basic Training Loss')
            plt.plot(history['val_loss'], label='Basic Validation Loss')
        if 'loss' in improved_history and 'val_loss' in improved_history:
            plt.plot(improved_history['loss'], label='Improved Training Loss')
            plt.plot(improved_history['val_loss'], label='Improved Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot MAE history
        plt.subplot(2, 2, 4)
        if 'mae' in history and 'val_mae' in history:
            plt.plot(history['mae'], label='Basic Training MAE')
            plt.plot(history['val_mae'], label='Basic Validation MAE')
        if 'mae' in improved_history and 'val_mae' in improved_history:
            plt.plot(improved_history['mae'], label='Improved Training MAE')
            plt.plot(improved_history['val_mae'], label='Improved Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()

def evaluate_models(data_dir):
    """Evaluate both models on the test set."""
    try:
        # Load test data
        _, _, test_ds = load_csi_data(data_dir)
        test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Load models
        basic_model_path = 'results/basic_localization/best_model.keras'
        improved_model_path = 'results/improved_localization/best_model.keras'
        
        if os.path.exists(basic_model_path):
            basic_model = tf.keras.models.load_model(basic_model_path)
            basic_results = basic_model.evaluate(test_ds, return_dict=True)
            print("\nBasic Model Results:")
            print(f"Test Loss: {basic_results['loss']:.4f}")
            print(f"Test MAE: {basic_results['mae']:.4f}")
        else:
            print(f"\nBasic model not found at {basic_model_path}")
            basic_results = None
        
        if os.path.exists(improved_model_path):
            improved_model = tf.keras.models.load_model(improved_model_path)
            improved_results = improved_model.evaluate(test_ds, return_dict=True)
            print("\nImproved Model Results:")
            print(f"Test Loss: {improved_results['loss']:.4f}")
            print(f"Test MAE: {improved_results['mae']:.4f}")
        else:
            print(f"\nImproved model not found at {improved_model_path}")
            improved_results = None
        
        return basic_results, improved_results
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Compare basic and improved models')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing TFRecord files')
    args = parser.parse_args()
    
    # Load metrics
    basic_metrics = load_metrics('basic_localization')
    improved_metrics = load_metrics('improved_localization')
    
    if basic_metrics and improved_metrics:
        # Plot comparison
        plot_comparison(basic_metrics, improved_metrics)
        print("\nModel comparison plots saved as results/model_comparison.png")
        
        # Try to evaluate models, but skip if not possible
        try:
            basic_results, improved_results = evaluate_models(args.data_dir)
            if basic_results and improved_results:
                # Calculate improvement percentages
                loss_improvement = ((basic_results['loss'] - improved_results['loss']) / basic_results['loss']) * 100
                mae_improvement = ((basic_results['mae'] - improved_results['mae']) / basic_results['mae']) * 100
                print("\nImprovement Analysis (from model evaluation):")
                print(f"Loss Improvement: {loss_improvement:.2f}%")
                print(f"MAE Improvement: {mae_improvement:.2f}%")
                if 'training_time' in improved_metrics and 'training_time' in basic_metrics:
                    print(f"Training Time Difference: {improved_metrics['training_time'] - basic_metrics['training_time']:.2f} seconds")
            else:
                print("\nModel evaluation skipped: Could not load one or both models. Showing only metrics from training.")
        except Exception as e:
            print(f"\nModel evaluation skipped due to error: {e}\nShowing only metrics from training.")
    else:
        print("\nError: Could not load metrics for both models")
        if not basic_metrics:
            print("Basic model metrics are missing")
        if not improved_metrics:
            print("Improved model metrics are missing")

if __name__ == "__main__":
    main() 