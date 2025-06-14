import os
import numpy as np
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime
from src.models.basic_localization import BasicLocalizationModel
from src.models.trajectory_model import TrajectoryAwareModel
from src.utils.competition_metrics import calculate_competition_metrics
from src.utils.model_utils import ModelUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    def __init__(self, model_paths: dict, output_dir: str = "optimized_models"):
        """
        Initialize model optimizer.
        
        Args:
            model_paths (dict): Dictionary of model paths for different tasks
            output_dir (str): Directory to save optimized models
        """
        self.model_paths = model_paths
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_utils = ModelUtils(str(self.output_dir))
        
        # Load models
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all models for optimization."""
        try:
            for task, path in self.model_paths.items():
                self.models[task] = tf.keras.models.load_model(path)
                logger.info(f"Loaded {task} model from {path}")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def optimize_hyperparameters(self, task: str, param_grid: dict, train_data: tuple, val_data: tuple):
        """
        Optimize hyperparameters for a specific task.
        
        Args:
            task (str): Task name ('basic', 'trajectory', or 'feature_selection')
            param_grid (dict): Grid of hyperparameters to try
            train_data (tuple): Training data (X_train, y_train)
            val_data (tuple): Validation data (X_val, y_val)
        """
        try:
            best_score = float('inf')
            best_params = None
            best_model = None
            
            # Grid search
            for params in self._generate_param_combinations(param_grid):
                logger.info(f"Trying parameters: {params}")
                
                # Create and train model with current parameters
                if task == 'basic':
                    model = BasicLocalizationModel(**params)
                elif task == 'trajectory':
                    model = TrajectoryAwareModel(**params)
                else:
                    raise ValueError(f"Unknown task: {task}")
                
                # Train model
                history = model.fit(
                    train_data[0], train_data[1],
                    validation_data=val_data,
                    epochs=50,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            patience=5,
                            restore_best_weights=True
                        )
                    ]
                )
                
                # Evaluate on validation set
                score = model.evaluate(val_data[0], val_data[1])[0]
                
                if score < best_score:
                    best_score = score
                    best_params = params
                    best_model = model
                    logger.info(f"New best score: {best_score}")
            
            # Save best model
            if best_model is not None:
                save_path = self.output_dir / f"{task}_optimized.h5"
                best_model.save(save_path)
                logger.info(f"Saved optimized model to {save_path}")
                
                # Save best parameters
                with open(self.output_dir / f"{task}_best_params.json", 'w') as f:
                    json.dump(best_params, f, indent=2)
                
                return best_model, best_params
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            raise
    
    def _generate_param_combinations(self, param_grid: dict) -> list:
        """Generate all combinations of parameters from grid."""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in itertools.product(*values):
            yield dict(zip(keys, instance))
    
    def quantize_model(self, task: str, representative_data: np.ndarray):
        """
        Quantize model for faster inference.
        
        Args:
            task (str): Task name
            representative_data (np.ndarray): Representative data for quantization
        """
        try:
            model = self.models[task]
            
            # Convert to TFLite model
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Set optimization flags
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
            # Set representative dataset
            def representative_dataset():
                for data in representative_data:
                    yield [data]
            converter.representative_dataset = representative_dataset
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save quantized model
            save_path = self.output_dir / f"{task}_quantized.tflite"
            with open(save_path, 'wb') as f:
                f.write(tflite_model)
            logger.info(f"Saved quantized model to {save_path}")
            
        except Exception as e:
            logger.error(f"Error quantizing model: {str(e)}")
            raise
    
    def create_ensemble(self, task: str, model_paths: list, weights: list = None):
        """
        Create ensemble of models for a specific task.
        
        Args:
            task (str): Task name
            model_paths (list): List of model paths to ensemble
            weights (list): Optional weights for each model
        """
        try:
            # Load models
            models = [tf.keras.models.load_model(path) for path in model_paths]
            
            # Create ensemble predictions function
            def ensemble_predict(x):
                predictions = []
                for model in models:
                    pred = model.predict(x)
                    predictions.append(pred)
                predictions = np.stack(predictions, axis=0)
                
                if weights is not None:
                    weights_norm = np.array(weights) / sum(weights)
                    return np.average(predictions, axis=0, weights=weights_norm)
                else:
                    return np.mean(predictions, axis=0)
            
            # Save ensemble configuration
            ensemble_config = {
                'model_paths': model_paths,
                'weights': weights,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.output_dir / f"{task}_ensemble_config.json", 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            
            logger.info(f"Created ensemble for {task} with {len(models)} models")
            return ensemble_predict
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            raise

def main():
    """Main function to optimize models."""
    # Define model paths
    model_paths = {
        'basic': 'models/best_model.h5',
        'trajectory': 'models/trajectory_model.h5',
        'feature_selection': 'models/feature_selection_model.h5'
    }
    
    # Initialize optimizer
    optimizer = ModelOptimizer(model_paths)
    
    # Define hyperparameter grids
    basic_param_grid = {
        'learning_rate': [0.001, 0.0001],
        'dropout_rate': [0.2, 0.3, 0.4],
        'l2_reg': [0.01, 0.001]
    }
    
    trajectory_param_grid = {
        'lstm_units': [64, 128],
        'attention_units': [32, 64],
        'dropout_rate': [0.2, 0.3]
    }
    
    # Load data (implement based on your dataset)
    # train_data = ...
    # val_data = ...
    
    # Optimize models
    # optimizer.optimize_hyperparameters('basic', basic_param_grid, train_data, val_data)
    # optimizer.optimize_hyperparameters('trajectory', trajectory_param_grid, train_data, val_data)
    
    # Quantize models
    # representative_data = ...
    # optimizer.quantize_model('basic', representative_data)
    # optimizer.quantize_model('trajectory', representative_data)
    
    # Create ensembles
    # model_paths = ['models/model1.h5', 'models/model2.h5']
    # optimizer.create_ensemble('basic', model_paths)

if __name__ == "__main__":
    main() 