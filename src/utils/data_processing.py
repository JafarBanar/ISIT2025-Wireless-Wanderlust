import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any

def load_csi_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CSI data from file.
    
    Args:
        file_path: Path to the CSI data file
        
    Returns:
        Tuple of (features, labels)
    """
    data = np.load(file_path)
    features = data['csi_features']
    labels = data['locations']
    return features, labels

def preprocess_csi_data(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess CSI data for model input.
    
    Args:
        features: Raw CSI features
        labels: Location labels
        
    Returns:
        Tuple of (processed_features, processed_labels)
    """
    # Normalize features
    features = features.astype(np.float32)
    features = (features - np.mean(features)) / (np.std(features) + 1e-8)
    
    # Normalize labels
    labels = labels.astype(np.float32)
    labels = (labels - np.mean(labels)) / (np.std(labels) + 1e-8)
    
    return features, labels

def create_sequence_data(features: np.ndarray, labels: np.ndarray, 
                        sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequence data for trajectory-aware model.
    
    Args:
        features: CSI features
        labels: Location labels
        sequence_length: Length of each sequence
        
    Returns:
        Tuple of (sequence_features, sequence_labels)
    """
    n_samples = len(features) - sequence_length + 1
    sequence_features = np.zeros((n_samples, sequence_length, *features.shape[1:]))
    sequence_labels = np.zeros((n_samples, 2))
    
    for i in range(n_samples):
        sequence_features[i] = features[i:i+sequence_length]
        sequence_labels[i] = labels[i+sequence_length-1]
    
    return sequence_features, sequence_labels

def create_data_generator(features: np.ndarray, labels: np.ndarray, 
                         batch_size: int) -> tf.data.Dataset:
    """
    Create TensorFlow dataset for training.
    
    Args:
        features: CSI features
        labels: Location labels
        batch_size: Batch size for training
        
    Returns:
        TensorFlow dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def split_data(features: np.ndarray, labels: np.ndarray, 
               train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Any]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        features: CSI features
        labels: Location labels
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        
    Returns:
        Dictionary containing split datasets
    """
    n_samples = len(features)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    return {
        'train': (features[train_indices], labels[train_indices]),
        'val': (features[val_indices], labels[val_indices]),
        'test': (features[test_indices], labels[test_indices])
    }

def prepare_submission_data(model, test_data: np.ndarray) -> np.ndarray:
    """
    Prepare data for competition submission.
    
    Args:
        model: Trained model
        test_data: Test data
        
    Returns:
        Predictions for submission
    """
    predictions = model.predict(test_data)
    return predictions 