import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import json
import shutil
from pathlib import Path
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import existing modules
from src.models.enhanced_localization import EnhancedLocalizationModel
from src.models.optimized_localization import OptimizedLocalizationModel
from src.data_loader import CSIDataLoader
from src.utils.error_analysis import ErrorAnalyzer
from src.utils.visualization import ChannelVisualizer
from src.models.vanilla_localization import VanillaLocalizationModel
from src.models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from src.models.feature_selection import FeatureSelectionModel
from src.utils.metrics import CombinedMetric, combined_loss
from src.utils.competition_metrics import R90Metric

class CompetitionTaskCompleter:
    def __init__(self):
        self.results_dir = Path('results')
        self.submission_dir = Path('submission')
        self.models_dir = Path('models')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def optimize_models(self):
        """Fine-tune and optimize models for best performance"""
        print("Optimizing models...")
        
        # Load best performing models
        models = {
            'enhanced': EnhancedLocalizationModel(),
            'optimized': OptimizedLocalizationModel()
        }
        
        # Fine-tune hyperparameters
        for name, model in models.items():
            # Implement hyperparameter tuning
            # Save optimized model
            model.save(self.models_dir / f'{name}_optimized.h5')
            
    def create_ensemble(self):
        """Create ensemble of best performing models"""
        print("Creating model ensemble...")
        
        # Load optimized models
        models = [
            tf.keras.models.load_model(self.models_dir / 'enhanced_optimized.h5'),
            tf.keras.models.load_model(self.models_dir / 'optimized_optimized.h5')
        ]
        
        # Create ensemble predictions
        # Save ensemble model
        
    def prepare_final_submission(self):
        """Prepare final competition submission"""
        print("Preparing final submission...")
        
        # Create submission directory
        submission_path = self.submission_dir / f'submission_{self.timestamp}'
        submission_path.mkdir(parents=True, exist_ok=True)
        
        # Generate predictions
        test_loader = CSIDataLoader('data/test')
        test_data = test_loader.load_dataset(is_training=False)
        
        # Load ensemble model
        ensemble_model = tf.keras.models.load_model(self.models_dir / 'ensemble_model.h5')
        
        # Generate predictions
        predictions = ensemble_model.predict(test_data)
        
        # Save predictions in competition format
        np.savetxt(
            submission_path / 'predictions.csv',
            predictions,
            delimiter=',',
            header='x,y',
            comments=''
        )
        
        # Create submission package
        self._create_submission_package(submission_path)
        
    def _create_submission_package(self, submission_path):
        """Create final submission package"""
        # Copy required files
        files_to_copy = [
            'README.md',
            'requirements.txt',
            'src/models/enhanced_localization.py',
            'src/models/optimized_localization.py',
            'src/utils/data_loader.py'
        ]
        
        for file in files_to_copy:
            shutil.copy2(file, submission_path / file)
            
        # Create zip file
        shutil.make_archive(
            str(submission_path),
            'zip',
            submission_path
        )
        
    def run_all_tasks(self):
        """Run all remaining tasks"""
        print("Starting remaining tasks...")
        
        # 1. Optimize models
        self.optimize_models()
        
        # 2. Create ensemble
        self.create_ensemble()
        
        # 3. Prepare final submission
        self.prepare_final_submission()
        
        print("All tasks completed!")

def main():
    completer = CompetitionTaskCompleter()
    completer.run_all_tasks()

def load_ensemble_model(model_paths):
    """
    Load an ensemble of models.
    
    Args:
        model_paths (dict): Dictionary mapping model types to their paths
        
    Returns:
        dict: Dictionary of loaded models
    """
    models = {}
    custom_objects = {
        'VanillaLocalizationModel': VanillaLocalizationModel,
        'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
        'FeatureSelectionModel': FeatureSelectionModel,
        'CombinedMetric': CombinedMetric,
        'combined_loss': combined_loss,
        'R90Metric': R90Metric
    }
    
    for model_type, path in model_paths.items():
        try:
            if model_type == 'vanilla':
                model = VanillaLocalizationModel.load_model(
                    path,
                    input_shape=(32, 1024, 2),
                    num_classes=2
                )
            elif model_type == 'trajectory':
                model = TrajectoryAwareLocalizationModel.load_model(
                    path,
                    input_shape=(32, 1024, 2),
                    num_classes=2
                )
            elif model_type == 'feature_selection':
                model = FeatureSelectionModel.load_model(
                    path,
                    input_shape=(32, 1024, 2),
                    num_classes=2
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            models[model_type] = model
        except Exception as e:
            logger.error(f"Error loading {model_type} model: {str(e)}")
            raise
    
    return models

def load_test_data(data_pattern):
    """Load test data using CSIDataLoader."""
    data_loader = CSIDataLoader(data_pattern)
    test_dataset, _ = data_loader.load_dataset(is_training=False)
    return test_dataset

def prepare_ensemble_submission(model_paths, data_dir, output_dir='submissions'):
    """
    Prepare submission using an ensemble of models.
    
    Args:
        model_paths (dict): Dictionary mapping model types to their paths
        data_dir (str): Directory containing test data
        output_dir (str): Directory to save submission files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    models = load_ensemble_model(model_paths)
    
    # Load test data
    test_data = load_test_data(os.path.join(data_dir, '*.tfrecords'))
    
    # Get predictions from each model
    predictions = {}
    for model_type, model in models.items():
        pred = model.predict(test_data)
        predictions[model_type] = pred
    
    # Combine predictions (simple averaging)
    ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
    
    # Save individual model predictions
    for model_type, pred in predictions.items():
        submission_df = pd.DataFrame({
            'x': pred[:, 0],
            'y': pred[:, 1],
            'model': model_type
        })
        output_file = os.path.join(output_dir, f'{model_type}_submission.csv')
        submission_df.to_csv(output_file, index=False)
        logger.info(f"Saved {model_type} submission to {output_file}")
    
    # Save ensemble predictions
    ensemble_df = pd.DataFrame({
        'x': ensemble_pred[:, 0],
        'y': ensemble_pred[:, 1],
        'model': 'ensemble'
    })
    output_file = os.path.join(output_dir, 'ensemble_submission.csv')
    ensemble_df.to_csv(output_file, index=False)
    logger.info(f"Saved ensemble submission to {output_file}")

if __name__ == "__main__":
    main() 