import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import tensorflow as tf
import sys
import logging

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.data_loader import CSIDataLoader
from validate_competition import validate_model
from prepare_documentation import generate_competition_documentation
from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import GrantFreeAccessModel as FeatureSelectionModel
from src.utils.metrics import combined_loss, CombinedMetric
from src.utils.competition_metrics import R90Metric, calculate_competition_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ISIT 2025 competition submission')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vanilla', 'trajectory', 'feature_selection', 'all'],
                      help='Type of model to use (use "all" for preparing submissions for all models)')
    parser.add_argument('--model_path', type=str,
                      help='Path to the trained model file (required for single model submission, ignored for "all" with --ensemble)')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test data (should contain test_tfrecord/*.tfrecords)')
    parser.add_argument('--output_dir', type=str, default='submissions',
                      help='Directory to save submission files (default: submissions)')
    parser.add_argument('--ensemble', action='store_true',
                      help='Prepare an ensemble submission using models from models/ensemble/ (requires --model_type all)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_type != 'all' and not args.model_path:
        parser.error('--model_path is required when --model_type is not "all"')
    
    return args

def load_model(model_type, model_path, input_shape=(32, 1024, 2), num_classes=2):
    """Load a trained model based on the model type."""
    try:
        if model_type == 'vanilla':
            model = VanillaLocalizationModel(input_shape, num_classes)
            model.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'VanillaLocalizationModel': VanillaLocalizationModel,
                    'CombinedMetric': CombinedMetric,
                    'combined_loss': combined_loss,
                    'R90Metric': R90Metric
                },
                compile=False
            )
        elif model_type == 'trajectory':
            model = TrajectoryAwareLocalizationModel(sequence_length=5)
            model.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
                    'CombinedMetric': CombinedMetric,
                    'combined_loss': combined_loss,
                    'R90Metric': R90Metric
                },
                compile=False
            )
        elif model_type == 'feature_selection':
            model = FeatureSelectionModel(input_shape, num_classes)
            model.model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'FeatureSelectionModel': FeatureSelectionModel,
                    'CombinedMetric': CombinedMetric,
                    'combined_loss': combined_loss,
                    'R90Metric': R90Metric
                },
                compile=False
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def prepare_submission(model, data_loader, output_path, model_type):
    """
    Prepare submission predictions for a given model.
    
    Args:
        model: Trained model instance
        data_loader: CSIDataLoader instance
        output_path: Path to save submission file
        model_type: Type of model ('vanilla', 'trajectory', 'feature_selection')
    """
    predictions = []
    positions = []

    # Get only the labeled dataset
    labeled_dataset, _ = data_loader.load_dataset(is_training=False)
    for batch in labeled_dataset:
        if model_type == 'trajectory':
            # For trajectory model, reshape input to sequence format
            csi = batch['csi']
            # Reshape to (batch_size, sequence_length, height, width, channels)
            csi = tf.reshape(csi, [-1, 5, 32, 1024, 2])
            flattened_data = tf.reduce_mean(csi, axis=-1, keepdims=True)
            print(flattened_data.shape)  # (batch, 32, 1024, 1)

            # Apply global average pooling to get (batch, 512)
            pooled_data = tf.keras.layers.GlobalAveragePooling2D()(flattened_data)
            print(pooled_data.shape)  # (batch, 512)

            # Now pass pooled_data to model.predict
            batch_pred = model.predict(pooled_data)
        else:
            batch_pred = model.predict(batch['csi'])
        predictions.append(batch_pred)
        positions.append(batch['position'].numpy())

    predictions = np.concatenate(predictions, axis=0)
    positions = np.concatenate(positions, axis=0)

    # Save predictions and ground truth
    output_dir = os.path.dirname(output_path)
    model_name = os.path.splitext(os.path.basename(output_path))[0]
    np.save(os.path.join(output_dir, f'{model_name}_predictions.npy'), predictions)
    np.save(os.path.join(output_dir, f'{model_name}_positions.npy'), positions)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'pred_x': predictions[:, 0],
        'pred_y': predictions[:, 1],
        'model': model_type
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")
    return predictions, positions

def prepare_submission_files(data_dir: str, output_dir: str, input_shape=(32, 1024, 2), num_classes=2):
    """
    Prepare competition submission files.
    
    Args:
        data_dir: Directory containing competition data
        output_dir: Directory to save submission files
        input_shape: Input shape for models (default: (32, 1024, 2))
        num_classes: Number of output classes (default: 2 for x,y coordinates)
    """
    # Create output directory
    submission_dir = os.path.join(output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)
    
    # Initialize data loader
    data_loader = CSIDataLoader("string/*.tfrecords")

    
    # Load test dataset
    test_dataset, _ = data_loader.load_dataset(is_training=False)
    
    # Validate all models
    model_types = ['vanilla', 'trajectory', 'feature_selection']
    all_predictions = {}
    all_metrics = {}
    
    for model_type in model_types:
        print(f"\nValidating {model_type} model...")
        model_output_dir = os.path.join(output_dir, 'validation', model_type)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Load model with correct custom objects
        model = load_model(model_type, os.path.join('models', f'{model_type}_model.h5'), input_shape, num_classes)
        
        # Make predictions
        predictions = []
        positions = []
        for batch in test_dataset:
            if model_type == 'trajectory':
                # For trajectory model, reshape input to sequence format
                csi = batch['csi']
                # Reshape to (batch_size, sequence_length, height, width, channels)
                csi = tf.reshape(csi, [-1, 5, 32, 1024, 2])
                flattened_data = tf.reduce_mean(csi, axis=-1, keepdims=True)
                print(flattened_data.shape)  # (batch, 32, 1024, 1)

                # Apply global average pooling to get (batch, 512)
                pooled_data = tf.keras.layers.GlobalAveragePooling2D()(flattened_data)
                print(pooled_data.shape)  # (batch, 512)

                # Now pass pooled_data to model.predict
                pred = model.predict(pooled_data)
            else:
                pred = model.predict(batch['csi'])
            predictions.append(pred)
            positions.append(batch['position'].numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        positions = np.concatenate(positions, axis=0)
        all_predictions[model_type] = predictions
        
        # Save predictions and ground truth
        np.save(os.path.join(model_output_dir, f'{model_type}_predictions.npy'), predictions)
        np.save(os.path.join(model_output_dir, f'{model_type}_positions.npy'), positions)
        
        # Calculate metrics
        metrics = validate_model(model_type, data_dir, model_output_dir)
        all_metrics[model_type] = metrics
    
    # Create predictions CSV
    print("\nPreparing predictions file...")
    predictions_df = pd.DataFrame()
    for model_type in model_types:
        predictions = all_predictions[model_type]
        df = pd.DataFrame({
            'x': predictions[:, 0],
            'y': predictions[:, 1],
            'model': model_type
        })
        predictions_df = pd.concat([predictions_df, df], ignore_index=True)
    
    predictions_df.to_csv(os.path.join(submission_dir, 'predictions.csv'), index=False)
    
    # Create metrics summary
    print("Preparing metrics summary...")
    metrics_summary = {
        'submission_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': {}
    }
    
    for model_type in model_types:
        metrics_summary['models'][model_type] = all_metrics[model_type]
    
    with open(os.path.join(submission_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    # Generate documentation
    print("Generating documentation...")
    docs_dir = os.path.join(submission_dir, 'docs')
    generate_competition_documentation(
        results_dir=os.path.join(output_dir, 'validation'),
        output_dir=docs_dir
    )
    
    # Create README
    print("Creating submission README...")
    readme_content = [
        "# ISIT 2025 Wireless Wanderlust Competition Submission",
        "\n## Team Information",
        "- Team Name: [Your Team Name]",
        "- Team Members:",
        "  * [Member 1] - IEEE Student Member",
        "  * [Member 2] - IEEE Student Member",
        "  * [Member 3] - IEEE Student Member",
        "\n## Submission Contents",
        "1. `predictions.csv` - Model predictions",
        "2. `metrics_summary.json` - Performance metrics",
        "3. `docs/` - Technical documentation",
        "\n## Model Performance Summary"
    ]
    
    for model_type in model_types:
        metrics = all_metrics[model_type]
        readme_content.extend([
            f"\n### {model_type.title()} Model",
            f"- MAE: {metrics['mae']:.4f}",
            f"- R90: {metrics['r90']:.4f}",
            f"- Combined Metric: {metrics['combined_metric']:.4f}"
        ])
    
    with open(os.path.join(submission_dir, 'README.md'), 'w') as f:
        f.write('\n'.join(readme_content))
    
    print(f"\nSubmission files prepared in {submission_dir}")

def prepare_all_submissions(data_dir, output_dir, model_paths=None, input_shape=(32, 1024, 2), num_classes=2):
    """Prepare submission files for all model types (vanilla, trajectory, feature_selection)."""
    if model_paths is None:
        model_paths = {
            "vanilla": "models/vanilla_model.h5",
            "trajectory": "models/trajectory_model.h5",
            "feature_selection": "models/feature_selection_model.h5"
        }
    model_types = ["vanilla", "trajectory", "feature_selection"]
    for model_type in model_types:
        model_path = model_paths.get(model_type, f"models/{model_type}_model.h5")
        print(f"\nPreparing submission for model type: {model_type} (using model at {model_path})")
        data_loader = CSIDataLoader(os.path.join(data_dir, "test_tfrecord", "*.tfrecords"))
        model = load_model(model_type, model_path, input_shape=input_shape, num_classes=num_classes)
        output_path = os.path.join(output_dir, f"submission_{model_type}.csv")
        predictions, positions = prepare_submission(model, data_loader, output_path, model_type)
        print(f"Submission for {model_type} saved to {output_path}.")
        # Save model (using save_model if available)
        try:
            if hasattr(model, "save_model"):
                model.save_model(os.path.join(output_dir, f"{model_type}_model.h5"))
            elif hasattr(model, "model") and hasattr(model.model, "save"):
                model.model.save(os.path.join(output_dir, f"{model_type}_model.h5"))
            else:
                print(f"Warning: Model for {model_type} does not have a save method.")
        except Exception as e:
            print(f"Error saving model for {model_type}: {e}")

def prepare_ensemble_submission(data_dir, output_dir, ensemble_model_paths, input_shape=(32, 1024, 2), num_classes=2):
    """Prepare an ensemble submission using custom model paths (one per model type)."""
    if not ensemble_model_paths:
        raise ValueError("Ensemble model paths must be provided.")
    print("Preparing ensemble submission using custom model paths.")
    prepare_all_submissions(data_dir, output_dir, model_paths=ensemble_model_paths, input_shape=input_shape, num_classes=num_classes)

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using data directory: {args.data_dir}")

    # Create validation directory
    validation_dir = output_dir / 'validation'
    validation_dir.mkdir(exist_ok=True)
    
    if args.model_type == "all":
        if args.ensemble:
            # For ensemble submission, use custom model paths
            ensemble_model_paths = {
                "vanilla": "models/ensemble/vanilla_ensemble.h5",
                "trajectory": "models/ensemble/trajectory_ensemble.h5",
                "feature_selection": "models/ensemble/feature_selection_ensemble.h5"
            }
            prepare_ensemble_submission(args.data_dir, str(output_dir), ensemble_model_paths)
        else:
            # Prepare submissions for all models
            prepare_all_submissions(args.data_dir, str(output_dir))
            
        # Prepare final submission files with validation
        prepare_submission_files(args.data_dir, str(output_dir))
    else:
        # Single model submission
        print(f"\nPreparing submission for {args.model_type} model...")
    
        # Create model-specific validation directory
        model_validation_dir = validation_dir / args.model_type
        model_validation_dir.mkdir(exist_ok=True)
        
        # Load and validate model
        model = load_model(args.model_type, args.model_path)
        metrics = validate_model(args.model_type, args.data_dir, str(model_validation_dir))
        
        # Print validation metrics
        print("\nValidation Metrics:")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R90: {metrics['r90']:.4f}")
        print(f"Combined Metric: {metrics['combined_metric']:.4f}")

        # Prepare submission
        data_loader = CSIDataLoader(os.path.join(args.data_dir, "test_tfrecord", "*.tfrecords"))
        output_path = output_dir / f"submission_{args.model_type}.csv"
        predictions, positions = prepare_submission(model, data_loader, str(output_path), args.model_type)
        
        # Save metrics
        metrics_path = model_validation_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nSubmission file prepared and saved to {output_path}")
        print(f"Validation metrics saved to {metrics_path}")

if __name__ == "__main__":
    main() 