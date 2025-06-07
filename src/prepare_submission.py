import argparse
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.isit2025.validate_competition import validate_model
from src.isit2025.prepare_documentation import generate_competition_documentation

from models.vanilla_localization import VanillaLocalizationModel
from models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from models.feature_selection_model import FeatureSelectionModel
from utils.data_processing import load_csi_data, preprocess_csi_data

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ISIT 2025 competition submission')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['vanilla', 'trajectory', 'feature_selection'],
                      help='Type of model to use')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='submissions',
                      help='Directory to save submission file')
    return parser.parse_args()

def load_model(model_type, model_path):
    """Load the trained model."""
    if model_type == 'vanilla':
        model = VanillaLocalizationModel.load_model(model_path)
    elif model_type == 'trajectory':
        model = TrajectoryAwareLocalizationModel.load_model(model_path)
    else:  # feature_selection
        model = FeatureSelectionModel.load_model(model_path)
    return model

def prepare_submission(model, test_data, output_path):
    """Prepare submission file with predictions."""
    # Get predictions
    if isinstance(model, FeatureSelectionModel):
        location_pred, _ = model.predict(test_data)
    else:
        location_pred = model.predict(test_data)
    
    # Create submission format
    submission = {
        'predictions': location_pred.tolist()
    }
    
    # Save submission
    with open(output_path, 'w') as f:
        json.dump(submission, f)

def prepare_submission_files(data_dir: str, output_dir: str):
    """
    Prepare competition submission files.
    
    Args:
        data_dir: Directory containing competition data
        output_dir: Directory to save submission files
    """
    # Create output directory
    submission_dir = os.path.join(output_dir, 'submission')
    os.makedirs(submission_dir, exist_ok=True)
    
    # Validate all models
    model_types = ['vanilla', 'trajectory', 'feature']
    all_predictions = {}
    all_metrics = {}
    
    for model_type in model_types:
        print(f"\nValidating {model_type} model...")
        model_output_dir = os.path.join(output_dir, 'validation', model_type)
        metrics = validate_model(model_type, data_dir, model_output_dir)
        
        # Load predictions
        predictions = np.load(os.path.join(model_output_dir, 'validation_predictions.npy'))
        all_predictions[model_type] = predictions
        all_metrics[model_type] = metrics
    
    # Create predictions CSV
    print("\nPreparing predictions file...")
    predictions_df = pd.DataFrame()
    for model_type in model_types:
        predictions = all_predictions[model_type]
        df = pd.DataFrame(predictions, columns=['x', 'y'])
        df['model'] = model_type
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
            f"- Combined Score: {metrics['combined_metric']:.4f}"
        ])
    
    with open(os.path.join(submission_dir, 'README.md'), 'w') as f:
        f.write('\n'.join(readme_content))
    
    print(f"\nSubmission files prepared in {submission_dir}")

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    data_path = Path(args.data_dir) / 'test_data.npz'
    test_features, _ = load_csi_data(str(data_path))
    test_features, _ = preprocess_csi_data(test_features, None)
    
    # Load model
    model = load_model(args.model_type, args.model_path)
    
    # Prepare submission
    output_path = output_dir / f'submission_{args.model_type}.json'
    prepare_submission(model, test_features, output_path)
    
    print(f"Submission file prepared and saved to {output_path}")

if __name__ == '__main__':
    main() 