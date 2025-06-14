import os
import argparse
import tensorflow as tf
import numpy as np
from models.feature_selection import GrantFreeAccessModel
from data_processing.trajectory_data import TrajectoryDataGenerator
from train_feature_selection import load_competition_data
from utils.visualization import plot_predictions, plot_error_distribution, plot_feature_importance

def evaluate_model(model_path: str, data_dir: str, output_dir: str):
    """Evaluate a trained model and generate plots."""
    # Load data
    print("Loading competition data...")
    csi_features, positions, priorities = load_competition_data(data_dir)
    
    # Create data generator
    print("Setting up data pipeline...")
    data_generator = TrajectoryDataGenerator(
        csi_features=csi_features,
        positions=positions,
        sequence_length=10,  # Using default sequence length
        batch_size=32,
        validation_split=0.2
    )
    
    # Get validation dataset
    val_dataset = data_generator.get_val_dataset()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
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
    
    # Generate plots
    print("\nGenerating plots...")
    plot_predictions(val_positions, val_predictions['positions'], output_dir)
    plot_error_distribution(val_positions, val_predictions['positions'], output_dir)
    # feature_scores = val_predictions['selection_mask']
    # plot_feature_importance(feature_scores, output_dir)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate a trained grant-free access model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing competition data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    evaluate_model(args.model_path, args.data_dir, args.output_dir) 