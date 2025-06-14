import tensorflow as tf
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import json

class EnsembleModel:
    def __init__(self, model_paths: List[str], weights: List[float] = None):
        """Initialize ensemble model with multiple models and their weights"""
        self.models = [tf.keras.models.load_model(path) for path in model_paths]
        self.weights = weights if weights is not None else [1.0] * len(model_paths)
        self.weights = np.array(self.weights) / sum(self.weights)  # Normalize weights
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate ensemble predictions"""
        predictions = []
        
        # Get predictions from each model
        for model in self.models:
            pred = model.predict(x)
            predictions.append(pred)
            
        # Stack predictions
        predictions = np.stack(predictions, axis=0)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        return weighted_pred
        
    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate ensemble model"""
        predictions = self.predict(x)
        
        # Calculate metrics
        mse = tf.keras.losses.mean_squared_error(y, predictions)
        mae = tf.keras.losses.mean_absolute_error(y, predictions)
        
        # Calculate R90
        errors = np.abs(predictions - y)
        r90 = np.percentile(errors, 90)
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'r90': float(r90)
        }
        
    def save(self, path: str):
        """Save ensemble configuration"""
        config = {
            'model_paths': [str(Path(p).absolute()) for p in self.model_paths],
            'weights': self.weights.tolist()
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)
            
    @classmethod
    def load(cls, path: str) -> 'EnsembleModel':
        """Load ensemble configuration"""
        with open(path, 'r') as f:
            config = json.load(f)
            
        return cls(config['model_paths'], config['weights'])

def create_ensemble_from_best_models():
    """Create ensemble from best performing models"""
    # Load best models
    model_paths = [
        'results/hyperparam_tuning/best_model.h5',
        'models/enhanced_optimized.h5',
        'models/optimized_optimized.h5'
    ]
    
    # Create ensemble
    ensemble = EnsembleModel(model_paths)
    
    # Save ensemble configuration
    ensemble.save('models/ensemble_config.json')
    
    return ensemble

def main():
    # Create ensemble
    ensemble = create_ensemble_from_best_models()
    
    # Load test data
    from src.utils.data_loader import CSIDataLoader
    data_loader = CSIDataLoader('data/test')
    test_data, _ = data_loader.load_dataset(is_training=False)
    
    # Evaluate ensemble
    metrics = ensemble.evaluate(test_data[0], test_data[1])
    print("\nEnsemble Model Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 