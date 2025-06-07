import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
import json
import pickle
from datetime import datetime

class ModelUtils:
    """Utility class for model-related operations."""
    
    def __init__(self, model_dir: str = 'models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: tf.keras.Model, model_name: str,
                  save_format: str = 'tf', include_optimizer: bool = True):
        """Save a model to disk."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = self.model_dir / f'{model_name}_{timestamp}'
            
            if save_format == 'tf':
                model.save(model_path, save_format='tf', include_optimizer=include_optimizer)
            elif save_format == 'h5':
                model.save(f'{model_path}.h5', save_format='h5', include_optimizer=include_optimizer)
            else:
                raise ValueError(f"Unsupported save format: {save_format}")
            
            logging.info(f"Model saved to {model_path}")
            return model_path
        
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: Union[str, Path]) -> tf.keras.Model:
        """Load a model from disk."""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Model loaded from {model_path}")
            return model
        
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise
    
    def save_model_config(self, model: tf.keras.Model, config_path: Union[str, Path]):
        """Save model configuration to a JSON file."""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            config = model.get_config()
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            
            logging.info(f"Model configuration saved to {config_path}")
        
        except Exception as e:
            logging.error(f"Error saving model configuration: {str(e)}")
            raise
    
    def load_model_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load model configuration from a JSON file."""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logging.info(f"Model configuration loaded from {config_path}")
            return config
        
        except Exception as e:
            logging.error(f"Error loading model configuration: {str(e)}")
            raise
    
    def save_training_history(self, history: Dict[str, List[float]],
                            history_path: Union[str, Path]):
        """Save training history to a JSON file."""
        try:
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
            
            logging.info(f"Training history saved to {history_path}")
        
        except Exception as e:
            logging.error(f"Error saving training history: {str(e)}")
            raise
    
    def load_training_history(self, history_path: Union[str, Path]) -> Dict[str, List[float]]:
        """Load training history from a JSON file."""
        try:
            history_path = Path(history_path)
            
            if not history_path.exists():
                raise FileNotFoundError(f"History file not found: {history_path}")
            
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            logging.info(f"Training history loaded from {history_path}")
            return history
        
        except Exception as e:
            logging.error(f"Error loading training history: {str(e)}")
            raise
    
    def save_weights(self, model: tf.keras.Model, weights_path: Union[str, Path]):
        """Save model weights to disk."""
        try:
            weights_path = Path(weights_path)
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            
            model.save_weights(weights_path)
            logging.info(f"Model weights saved to {weights_path}")
        
        except Exception as e:
            logging.error(f"Error saving model weights: {str(e)}")
            raise
    
    def load_weights(self, model: tf.keras.Model, weights_path: Union[str, Path]):
        """Load model weights from disk."""
        try:
            weights_path = Path(weights_path)
            
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
            model.load_weights(weights_path)
            logging.info(f"Model weights loaded from {weights_path}")
        
        except Exception as e:
            logging.error(f"Error loading model weights: {str(e)}")
            raise
    
    def save_optimizer_state(self, optimizer: tf.keras.optimizers.Optimizer,
                           state_path: Union[str, Path]):
        """Save optimizer state to disk."""
        try:
            state_path = Path(state_path)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_path, 'wb') as f:
                pickle.dump(optimizer.get_weights(), f)
            
            logging.info(f"Optimizer state saved to {state_path}")
        
        except Exception as e:
            logging.error(f"Error saving optimizer state: {str(e)}")
            raise
    
    def load_optimizer_state(self, optimizer: tf.keras.optimizers.Optimizer,
                           state_path: Union[str, Path]):
        """Load optimizer state from disk."""
        try:
            state_path = Path(state_path)
            
            if not state_path.exists():
                raise FileNotFoundError(f"Optimizer state file not found: {state_path}")
            
            with open(state_path, 'rb') as f:
                optimizer.set_weights(pickle.load(f))
            
            logging.info(f"Optimizer state loaded from {state_path}")
        
        except Exception as e:
            logging.error(f"Error loading optimizer state: {str(e)}")
            raise
    
    def get_model_summary(self, model: tf.keras.Model) -> str:
        """Get a string representation of the model summary."""
        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        return '\n'.join(string_list)
    
    def count_model_parameters(self, model: tf.keras.Model) -> Dict[str, int]:
        """Count the number of trainable and non-trainable parameters."""
        trainable_params = 0
        non_trainable_params = 0
        
        for layer in model.layers:
            trainable_params += layer.count_params() if layer.trainable else 0
            non_trainable_params += layer.count_params() if not layer.trainable else 0
        
        return {
            'trainable_params': trainable_params,
            'non_trainable_params': non_trainable_params,
            'total_params': trainable_params + non_trainable_params
        }
    
    def get_layer_outputs(self, model: tf.keras.Model, input_data: np.ndarray,
                         layer_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get outputs of specified layers for given input data."""
        if layer_names is None:
            layer_names = [layer.name for layer in model.layers]
        
        outputs = {}
        for layer_name in layer_names:
            try:
                layer_model = tf.keras.Model(inputs=model.input,
                                          outputs=model.get_layer(layer_name).output)
                outputs[layer_name] = layer_model.predict(input_data)
            except Exception as e:
                logging.warning(f"Could not get output for layer {layer_name}: {str(e)}")
        
        return outputs
    
    def freeze_layers(self, model: tf.keras.Model, layer_names: List[str]):
        """Freeze specified layers in the model."""
        for layer in model.layers:
            if layer.name in layer_names:
                layer.trainable = False
            else:
                layer.trainable = True
        
        model.compile(optimizer=model.optimizer,
                     loss=model.loss,
                     metrics=model.metrics)
    
    def unfreeze_layers(self, model: tf.keras.Model, layer_names: List[str]):
        """Unfreeze specified layers in the model."""
        for layer in model.layers:
            if layer.name in layer_names:
                layer.trainable = True
        
        model.compile(optimizer=model.optimizer,
                     loss=model.loss,
                     metrics=model.metrics)
    
    def create_model_checkpoint(self, filepath: Union[str, Path],
                              monitor: str = 'val_loss',
                              mode: str = 'min',
                              save_best_only: bool = True,
                              save_weights_only: bool = False) -> tf.keras.callbacks.ModelCheckpoint:
        """Create a model checkpoint callback."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(filepath),
            monitor=monitor,
            mode=mode,
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            verbose=1
        )
    
    def create_early_stopping(self, monitor: str = 'val_loss',
                            patience: int = 10,
                            mode: str = 'min') -> tf.keras.callbacks.EarlyStopping:
        """Create an early stopping callback."""
        return tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience,
            mode=mode,
            verbose=1
        )
    
    def create_reduce_lr(self, monitor: str = 'val_loss',
                        factor: float = 0.1,
                        patience: int = 5,
                        mode: str = 'min') -> tf.keras.callbacks.ReduceLROnPlateau:
        """Create a learning rate reduction callback."""
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            mode=mode,
            verbose=1
        ) 