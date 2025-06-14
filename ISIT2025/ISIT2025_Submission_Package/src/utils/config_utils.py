import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import logging
from datetime import datetime

class ConfigUtils:
    """Utility class for managing configuration files."""
    
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path],
                   config_type: str = 'json') -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            if config_type == 'json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            elif config_type == 'yaml':
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported configuration type: {config_type}")
            
            logging.info(f"Configuration loaded from {config_path}")
            return config
        
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise
    
    def save_config(self, config: Dict[str, Any],
                   config_path: Union[str, Path],
                   config_type: str = 'json'):
        """Save configuration to file."""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            if config_type == 'json':
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
            elif config_type == 'yaml':
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported configuration type: {config_type}")
            
            logging.info(f"Configuration saved to {config_path}")
        
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            raise
    
    def create_default_config(self, config_type: str = 'json') -> Dict[str, Any]:
        """Create a default configuration dictionary."""
        config = {
            'model': {
                'name': 'default_model',
                'input_shape': [None, 32],
                'output_shape': [None, 2],
                'layers': [
                    {'type': 'dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'dense', 'units': 32, 'activation': 'relu'},
                    {'type': 'dense', 'units': 2, 'activation': 'linear'}
                ]
            },
            'training': {
                'batch_size': 32,
                'epochs': 100,
                'learning_rate': 0.001,
                'optimizer': 'adam',
                'loss': 'mse',
                'metrics': ['mae', 'mse'],
                'validation_split': 0.2,
                'early_stopping': {
                    'monitor': 'val_loss',
                    'patience': 10,
                    'mode': 'min'
                },
                'reduce_lr': {
                    'monitor': 'val_loss',
                    'factor': 0.1,
                    'patience': 5,
                    'mode': 'min'
                }
            },
            'data': {
                'train_path': 'data/train',
                'val_path': 'data/val',
                'test_path': 'data/test',
                'feature_columns': ['feature1', 'feature2'],
                'target_columns': ['target1', 'target2'],
                'preprocessing': {
                    'scale_features': True,
                    'scale_targets': False
                }
            },
            'logging': {
                'level': 'INFO',
                'log_dir': 'logs',
                'save_frequency': 1
            },
            'output': {
                'model_dir': 'models',
                'results_dir': 'results',
                'save_format': 'tf'
            }
        }
        
        return config
    
    def update_config(self, config: Dict[str, Any],
                     updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update configuration with new values."""
        def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in u.items():
                if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                    d[k] = deep_update(d[k], v)
                else:
                    d[k] = v
            return d
        
        return deep_update(config, updates)
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        required_sections = ['model', 'training', 'data', 'logging', 'output']
        
        # Check required sections
        for section in required_sections:
            if section not in config:
                logging.error(f"Missing required section: {section}")
                return False
        
        # Validate model configuration
        model_config = config['model']
        if not all(k in model_config for k in ['name', 'input_shape', 'output_shape', 'layers']):
            logging.error("Invalid model configuration")
            return False
        
        # Validate training configuration
        training_config = config['training']
        if not all(k in training_config for k in ['batch_size', 'epochs', 'learning_rate']):
            logging.error("Invalid training configuration")
            return False
        
        # Validate data configuration
        data_config = config['data']
        if not all(k in data_config for k in ['train_path', 'val_path', 'test_path']):
            logging.error("Invalid data configuration")
            return False
        
        # Validate logging configuration
        logging_config = config['logging']
        if not all(k in logging_config for k in ['level', 'log_dir']):
            logging.error("Invalid logging configuration")
            return False
        
        # Validate output configuration
        output_config = config['output']
        if not all(k in output_config for k in ['model_dir', 'results_dir']):
            logging.error("Invalid output configuration")
            return False
        
        return True
    
    def create_config_from_template(self, template_name: str,
                                  output_path: Union[str, Path],
                                  config_type: str = 'json',
                                  **kwargs) -> Dict[str, Any]:
        """Create a configuration file from a template."""
        try:
            # Load template
            template_path = self.config_dir / f'templates/{template_name}.{config_type}'
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_path}")
            
            config = self.load_config(template_path, config_type)
            
            # Update with provided values
            if kwargs:
                config = self.update_config(config, kwargs)
            
            # Validate configuration
            if not self.validate_config(config):
                raise ValueError("Invalid configuration")
            
            # Save configuration
            self.save_config(config, output_path, config_type)
            
            return config
        
        except Exception as e:
            logging.error(f"Error creating configuration from template: {str(e)}")
            raise
    
    def backup_config(self, config_path: Union[str, Path]):
        """Create a backup of the configuration file."""
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = config_path.parent / f'{config_path.stem}_backup_{timestamp}{config_path.suffix}'
            
            with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            
            logging.info(f"Configuration backup created at {backup_path}")
        
        except Exception as e:
            logging.error(f"Error creating configuration backup: {str(e)}")
            raise
    
    def merge_configs(self, configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple configurations."""
        merged_config = {}
        
        for config in configs:
            merged_config = self.update_config(merged_config, config)
        
        return merged_config
    
    def get_config_diff(self, config1: Dict[str, Any],
                       config2: Dict[str, Any]) -> Dict[str, Any]:
        """Get the difference between two configurations."""
        def deep_diff(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]:
            diff = {}
            
            for k, v in d2.items():
                if k not in d1:
                    diff[k] = v
                elif isinstance(v, dict) and isinstance(d1[k], dict):
                    nested_diff = deep_diff(d1[k], v)
                    if nested_diff:
                        diff[k] = nested_diff
                elif v != d1[k]:
                    diff[k] = v
            
            return diff
        
        return deep_diff(config1, config2) 