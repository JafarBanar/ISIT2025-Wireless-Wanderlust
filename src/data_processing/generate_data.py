import numpy as np
import os
from pathlib import Path

def generate_synthetic_data(n_samples: int = 1000,
                          n_arrays: int = 4,
                          n_elements: int = 8,
                          n_frequencies: int = 16,
                          sequence_length: int = 10):
    """
    Generate synthetic CSI data and positions for testing.
    
    Args:
        n_samples: Number of samples
        n_arrays: Number of antenna arrays (4 in competition)
        n_elements: Number of elements per array (8 in competition)
        n_frequencies: Number of frequency bands (16 in competition)
        sequence_length: Length of trajectory sequences
    """
    # Generate random CSI features
    csi_features = np.random.normal(
        size=(n_samples, n_arrays, n_elements, n_frequencies, 2)  # 2 for real/imag
    ).astype(np.float32)
    
    # Generate random positions (x,y coordinates)
    positions = np.zeros((n_samples, 2), dtype=np.float32)
    
    # Create trajectories with smooth motion
    for i in range(0, n_samples, sequence_length):
        # Random starting point
        start_x = np.random.uniform(-10, 10)
        start_y = np.random.uniform(-10, 10)
        
        # Random velocity components
        vx = np.random.uniform(-1, 1)
        vy = np.random.uniform(-1, 1)
        
        # Generate trajectory
        for j in range(min(sequence_length, n_samples - i)):
            # Add some noise to velocity
            vx += np.random.normal(0, 0.1)
            vy += np.random.normal(0, 0.1)
            
            # Update position
            positions[i + j, 0] = start_x + vx * j
            positions[i + j, 1] = start_y + vy * j
    
    return csi_features, positions

def save_competition_data(output_dir: str,
                         n_samples: int = 1000,
                         n_arrays: int = 4,
                         n_elements: int = 8,
                         n_frequencies: int = 16,
                         sequence_length: int = 10):
    """
    Generate and save synthetic competition data.
    
    Args:
        output_dir: Directory to save data
        n_samples: Number of samples
        n_arrays: Number of antenna arrays
        n_elements: Number of elements per array
        n_frequencies: Number of frequency bands
        sequence_length: Length of trajectory sequences
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate data
    print("Generating synthetic data...")
    csi_features, positions = generate_synthetic_data(
        n_samples=n_samples,
        n_arrays=n_arrays,
        n_elements=n_elements,
        n_frequencies=n_frequencies,
        sequence_length=sequence_length
    )
    
    # Save data
    print("Saving data...")
    np.save(os.path.join(output_dir, 'csi_features.npy'), csi_features)
    np.save(os.path.join(output_dir, 'positions.npy'), positions)
    
    print(f"Data saved to {output_dir}")
    print(f"CSI features shape: {csi_features.shape}")
    print(f"Positions shape: {positions.shape}")

if __name__ == "__main__":
    # Generate data for testing
    save_competition_data(
        output_dir='data/competition',
        n_samples=1000,
        sequence_length=10
    ) 