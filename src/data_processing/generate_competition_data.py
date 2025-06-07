"""Generate competition-specific data for validation."""

import os
import numpy as np
from typing import Tuple, Dict
from src.isit2025.config.competition_config import DATASET_CONFIG

def generate_csi_features(n_samples: int,
                         n_arrays: int,
                         n_elements: int,
                         n_frequencies: int) -> np.ndarray:
    """
    Generate synthetic CSI features for competition validation.
    
    Args:
        n_samples: Number of samples
        n_arrays: Number of antenna arrays
        n_elements: Number of elements per array
        n_frequencies: Number of frequency bands
        
    Returns:
        CSI features array of shape (n_samples, n_arrays, n_elements, n_frequencies, 2)
    """
    # Generate random complex CSI values
    csi_real = np.random.normal(0, 1, (n_samples, n_arrays, n_elements, n_frequencies))
    csi_imag = np.random.normal(0, 1, (n_samples, n_arrays, n_elements, n_frequencies))
    
    # Add path loss effects
    distances = np.random.uniform(1, 10, (n_samples, n_arrays))
    path_loss = 1 / np.sqrt(distances)
    path_loss = path_loss[..., np.newaxis, np.newaxis]  # Add dimensions for elements and frequencies
    
    csi_real *= path_loss
    csi_imag *= path_loss
    
    # Stack real and imaginary parts
    csi_features = np.stack([csi_real, csi_imag], axis=-1)
    
    return csi_features

def generate_positions(n_samples: int,
                      room_size: float = 10.0) -> np.ndarray:
    """
    Generate random positions within a room.
    
    Args:
        n_samples: Number of samples
        room_size: Size of the room in meters
        
    Returns:
        Position array of shape (n_samples, 2)
    """
    return np.random.uniform(0, room_size, (n_samples, 2))

def generate_competition_data(output_dir: str) -> Dict[str, np.ndarray]:
    """
    Generate and save competition data.
    
    Args:
        output_dir: Directory to save the data
        
    Returns:
        Dictionary containing the generated data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate CSI features
    print("Generating CSI features...")
    csi_features = generate_csi_features(
        n_samples=DATASET_CONFIG['n_samples'],
        n_arrays=DATASET_CONFIG['n_arrays'],
        n_elements=DATASET_CONFIG['n_elements'],
        n_frequencies=DATASET_CONFIG['n_frequencies']
    )
    
    # Generate positions
    print("Generating positions...")
    positions = generate_positions(DATASET_CONFIG['n_samples'])
    
    # Save data
    print("Saving data...")
    np.save(os.path.join(output_dir, 'csi_features.npy'), csi_features)
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    
    # Save dataset info
    dataset_info = {
        'n_samples': DATASET_CONFIG['n_samples'],
        'n_arrays': DATASET_CONFIG['n_arrays'],
        'n_elements': DATASET_CONFIG['n_elements'],
        'n_frequencies': DATASET_CONFIG['n_frequencies'],
        'csi_features_shape': csi_features.shape,
        'positions_shape': positions.shape
    }
    
    np.save(os.path.join(output_dir, 'dataset_info.npy'), dataset_info)
    
    return {
        'csi_features': csi_features,
        'positions': positions,
        'dataset_info': dataset_info
    }

if __name__ == "__main__":
    from src.isit2025.config.competition_config import PATHS
    generate_competition_data(PATHS['data_dir']) 