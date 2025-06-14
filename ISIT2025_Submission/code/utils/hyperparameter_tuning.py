import tensorflow as tf
import numpy as np
from sklearn.model_selection import ParameterGrid
from typing import Dict, List, Any
import json
from pathlib import Path

class HyperparameterTuner:
    def __init__(self, model_class, param_grid: Dict[str, List[Any]]):
        self.model_class = model_class
        self.param_grid = param_grid
        self.best_params = None
        self.best_score = float('inf')
        self.results_dir = Path('results/hyperparam_tuning')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def tune(self, train_data, val_data, epochs=10):
        """Perform hyperparameter tuning"""
        print("Starting hyperparameter tuning...")
        
        # Create parameter combinations
        param_combinations = list(ParameterGrid(self.param_grid))
        results = []
        
        for i, params in enumerate(param_combinations):
            print(f"\nTrying combination {i+1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            # Create and train model
            model = self.model_class(**params)
            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=5,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Evaluate model
            val_loss = min(history.history['val_loss'])
            results.append({
                'params': params,
                'val_loss': val_loss
            })
            
            # Update best parameters
            if val_loss < self.best_score:
                self.best_score = val_loss
                self.best_params = params
                
                # Save best model
                model.save(self.results_dir / 'best_model.h5')
                
        # Save results
        self._save_results(results)
        
        return self.best_params, self.best_score
        
    def _save_results(self, results: List[Dict]):
        """Save tuning results"""
        # Save results to JSON
        with open(self.results_dir / 'tuning_results.json', 'w') as f:
            json.dump({
                'results': results,
                'best_params': self.best_params,
                'best_score': self.best_score
            }, f, indent=4)
            
        # Save best parameters
        with open(self.results_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)

def get_default_param_grid():
    """Get default parameter grid for tuning"""
    return {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'dropout_rate': [0.1, 0.2, 0.3],
        'l2_reg': [1e-4, 1e-3, 1e-2],
        'num_filters': [32, 64, 128],
        'num_attention_heads': [4, 8, 16]
    }

def main():
    # Example usage
    from src.models.enhanced_localization import EnhancedLocalizationModel
    
    # Load data
    from src.utils.data_loader import CSIDataLoader
    data_loader = CSIDataLoader('data/train')
    train_data, _ = data_loader.load_dataset(is_training=True)
    
    # Create tuner
    tuner = HyperparameterTuner(
        EnhancedLocalizationModel,
        get_default_param_grid()
    )
    
    # Perform tuning
    best_params, best_score = tuner.tune(train_data, train_data)
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score}")

if __name__ == "__main__":
    main() 