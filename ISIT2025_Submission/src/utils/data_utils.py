import numpy as np
import torch
import pandas as pd
from typing import Tuple, Dict, Optional, Union, List
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def load_csi_data(data_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load CSI data from the dataset.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Tuple containing:
        - CSI measurements (complex-valued tensor)
        - Ground truth positions (if available)
    """
    # TODO: Implement actual data loading
    raise NotImplementedError("Data loading not implemented yet")

def preprocess_csi(csi_data: np.ndarray) -> torch.Tensor:
    """
    Preprocess CSI data for model input.
    
    Args:
        csi_data: Raw CSI measurements
        
    Returns:
        Preprocessed tensor ready for model input
    """
    # Convert complex data to real representation
    # Shape: (4, 8, 16) complex -> (4, 8, 16, 2) real
    real_data = np.stack([csi_data.real, csi_data.imag], axis=-1)
    return torch.from_numpy(real_data).float()

def calculate_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    Args:
        predictions: Predicted positions
        targets: Ground truth positions
        
    Returns:
        MAE value
    """
    return np.mean(np.abs(predictions - targets))

def calculate_r90(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    Calculate R90 metric.
    
    Args:
        predictions: Predicted positions
        targets: Ground truth positions
        
    Returns:
        R90 value
    """
    # Calculate distances between predictions and targets
    distances = np.sqrt(np.sum((predictions - targets) ** 2, axis=1))
    # Find radius that contains 90% of predictions
    return np.percentile(distances, 90)

def split_data(csi_data: np.ndarray, 
               positions: Optional[np.ndarray] = None,
               labeled_ratio: float = 0.25) -> Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]]:
    """
    Split data into labeled and unlabeled sets.
    
    Args:
        csi_data: CSI measurements
        positions: Ground truth positions (optional)
        labeled_ratio: Ratio of labeled data
        
    Returns:
        Dictionary containing train/val/test splits
    """
    n_samples = len(csi_data)
    n_labeled = int(n_samples * labeled_ratio)
    
    # Random shuffle
    indices = np.random.permutation(n_samples)
    
    # Split indices
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    
    # Create splits
    splits = {
        'labeled': (csi_data[labeled_indices], positions[labeled_indices] if positions is not None else None),
        'unlabeled': (csi_data[unlabeled_indices], None)
    }
    
    return splits 

class DataUtils:
    """Utility class for data manipulation and preprocessing."""
    
    def __init__(self):
        self.scalers = {}
    
    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """Load data from various file formats."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if file_path.suffix == '.csv':
                data = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix == '.xlsx':
                data = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix == '.json':
                data = pd.read_json(file_path, **kwargs)
            elif file_path.suffix == '.parquet':
                data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logging.info(f"Successfully loaded data from {file_path}")
            return data
        
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def save_data(self, data: Union[pd.DataFrame, np.ndarray],
                 file_path: Union[str, Path], **kwargs):
        """Save data to various file formats."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            
            if file_path.suffix == '.csv':
                data.to_csv(file_path, **kwargs)
            elif file_path.suffix == '.xlsx':
                data.to_excel(file_path, **kwargs)
            elif file_path.suffix == '.json':
                data.to_json(file_path, **kwargs)
            elif file_path.suffix == '.parquet':
                data.to_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logging.info(f"Successfully saved data to {file_path}")
        
        except Exception as e:
            logging.error(f"Error saving data to {file_path}: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame,
                       feature_columns: List[str],
                       target_columns: Optional[List[str]] = None,
                       scale_features: bool = True,
                       scale_targets: bool = False,
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Dict[str, Dict[str, np.ndarray]]:
        """Preprocess data for model training."""
        # Split features and targets
        X = data[feature_columns].values
        y = data[target_columns].values if target_columns else None
        
        # Scale features if requested
        if scale_features:
            self.scalers['features'] = StandardScaler()
            X = self.scalers['features'].fit_transform(X)
        
        # Scale targets if requested
        if scale_targets and y is not None:
            self.scalers['targets'] = StandardScaler()
            y = self.scalers['targets'].fit_transform(y)
        
        # Split into train, validation, and test sets
        if y is not None:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
            )
            
            return {
                'train': {'X': X_train, 'y': y_train},
                'val': {'X': X_val, 'y': y_val},
                'test': {'X': X_test, 'y': y_test}
            }
        else:
            X_train_val, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            
            val_ratio = val_size / (1 - test_size)
            X_train, X_val = train_test_split(
                X_train_val, test_size=val_ratio, random_state=random_state
            )
            
            return {
                'train': {'X': X_train},
                'val': {'X': X_val},
                'test': {'X': X_test}
            }
    
    def inverse_transform(self, data: np.ndarray, data_type: str = 'features') -> np.ndarray:
        """Convert scaled data back to original scale."""
        if data_type not in self.scalers:
            raise ValueError(f"No scaler found for {data_type}")
        
        return self.scalers[data_type].inverse_transform(data)
    
    def normalize_data(self, data: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Normalize data to a specified range."""
        scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit_transform(data)
    
    def create_sequences(self, data: np.ndarray, sequence_length: int,
                        stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""
        sequences = []
        targets = []
        
        for i in range(0, len(data) - sequence_length, stride):
            sequences.append(data[i:i + sequence_length])
            targets.append(data[i + sequence_length])
        
        return np.array(sequences), np.array(targets)
    
    def create_windows(self, data: np.ndarray, window_size: int,
                      stride: int = 1) -> np.ndarray:
        """Create sliding windows from data."""
        windows = []
        
        for i in range(0, len(data) - window_size + 1, stride):
            windows.append(data[i:i + window_size])
        
        return np.array(windows)
    
    def add_noise(self, data: np.ndarray, noise_type: str = 'gaussian',
                 noise_level: float = 0.1) -> np.ndarray:
        """Add noise to data."""
        if noise_type == 'gaussian':
            noise = np.random.normal(0, noise_level, data.shape)
        elif noise_type == 'uniform':
            noise = np.random.uniform(-noise_level, noise_level, data.shape)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")
        
        return data + noise
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using z-score method."""
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return data[z_scores < threshold]
    
    def interpolate_missing(self, data: np.ndarray, method: str = 'linear') -> np.ndarray:
        """Interpolate missing values in data."""
        if isinstance(data, pd.DataFrame):
            return data.interpolate(method=method).values
        else:
            return pd.DataFrame(data).interpolate(method=method).values
    
    def resample_data(self, data: np.ndarray, target_length: int,
                     method: str = 'linear') -> np.ndarray:
        """Resample data to target length."""
        if len(data) == target_length:
            return data
        
        indices = np.linspace(0, len(data) - 1, target_length)
        return np.interp(indices, np.arange(len(data)), data)
    
    def augment_data(self, data: np.ndarray, augmentation_type: str = 'rotation',
                    **kwargs) -> np.ndarray:
        """Augment data using various methods."""
        if augmentation_type == 'rotation':
            angle = kwargs.get('angle', 15)
            return np.rot90(data, k=angle // 90)
        elif augmentation_type == 'flip':
            axis = kwargs.get('axis', 0)
            return np.flip(data, axis=axis)
        elif augmentation_type == 'scale':
            scale_factor = kwargs.get('scale_factor', 1.2)
            return data * scale_factor
        else:
            raise ValueError(f"Unsupported augmentation type: {augmentation_type}")
    
    def balance_classes(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'undersample') -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes in the dataset."""
        from collections import Counter
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import RandomOverSampler
        
        class_counts = Counter(y)
        min_class = min(class_counts.values())
        
        if method == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy='all')
        elif method == 'oversample':
            sampler = RandomOverSampler(sampling_strategy='all')
        else:
            raise ValueError(f"Unsupported balancing method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def get_data_info(self, data: Union[pd.DataFrame, np.ndarray]) -> Dict[str, Any]:
        """Get information about the data."""
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        info = {
            'shape': data.shape,
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'numeric_stats': data.describe().to_dict() if data.select_dtypes(include=[np.number]).columns.any() else None,
            'categorical_stats': data.describe(include=['object']).to_dict() if data.select_dtypes(include=['object']).columns.any() else None
        }
        
        return info 