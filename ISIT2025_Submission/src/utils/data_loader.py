import os
import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, List, Any, Optional, Generator
import h5py
from pathlib import Path
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def parse_tfrecord_function(example_proto):
    """Parse TFRecord data"""
    feature_description = {
        'csi_data': tf.io.FixedLenFeature([], tf.string),
        'position': tf.io.FixedLenFeature([], tf.string),
        'timestamp': tf.io.FixedLenFeature([], tf.int64)
    }
    
    parsed_features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode the serialized tensors
    csi_data = tf.io.parse_tensor(parsed_features['csi_data'], out_type=tf.float32)
    position = tf.io.parse_tensor(parsed_features['position'], out_type=tf.float32)
    position = position[..., :2]  # Only use x and y
    
    return csi_data, position

def create_dataset(file_pattern: str, batch_size: int, shuffle: bool = False):
    """Create dataset from TFRecord files"""
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Parse TFRecord data
    dataset = dataset.map(parse_tfrecord_function, 
                         num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def prepare_data(data_dir: str = 'data/competition', batch_size: int = 32):
    """Prepare training, validation and test datasets"""
    # Update file patterns for TFRecord files
    train_pattern = os.path.join(data_dir, 'train_tfrecord', '*.tfrecord')
    val_pattern = os.path.join(data_dir, 'val_tfrecord', '*.tfrecord')
    test_pattern = os.path.join(data_dir, 'test_tfrecord', '*.tfrecord')
    
    # Create datasets
    train_dataset = create_dataset(train_pattern, batch_size, shuffle=True)
    val_dataset = create_dataset(val_pattern, batch_size)
    test_dataset = create_dataset(test_pattern, batch_size)
    
    return train_dataset, val_dataset, test_dataset

def load_and_preprocess_data(data_dir='data/competition', batch_size=32):
    """Load and preprocess data for the channel-aware model."""
    data_dir = Path(data_dir)
    
    # Load data
    train_csi = np.load(data_dir / 'train_csi.npy')
    train_pos = np.load(data_dir / 'train_pos.npy')
    val_csi = np.load(data_dir / 'val_csi.npy')
    val_pos = np.load(data_dir / 'val_pos.npy')
    test_csi = np.load(data_dir / 'test_csi.npy')
    test_pos = np.load(data_dir / 'test_pos.npy')
    
    # Log data shapes
    logging.info(f"Training data shapes: CSI {train_csi.shape}, Positions {train_pos.shape}")
    logging.info(f"Validation data shapes: CSI {val_csi.shape}, Positions {val_pos.shape}")
    logging.info(f"Test data shapes: CSI {test_csi.shape}, Positions {test_pos.shape}")
    
    # Normalize CSI data
    def normalize_csi(csi_data):
        # Normalize each channel independently
        mean = np.mean(csi_data, axis=(1, 2), keepdims=True)
        std = np.std(csi_data, axis=(1, 2), keepdims=True)
        return (csi_data - mean) / (std + 1e-8)
    
    train_csi = normalize_csi(train_csi)
    val_csi = normalize_csi(val_csi)
    test_csi = normalize_csi(test_csi)
    
    # Normalize position data
    def normalize_positions(pos_data):
        mean = np.mean(pos_data, axis=0)
        std = np.std(pos_data, axis=0)
        return (pos_data - mean) / (std + 1e-8), mean, std
    
    train_pos, pos_mean, pos_std = normalize_positions(train_pos)
    val_pos = (val_pos - pos_mean) / (pos_std + 1e-8)
    test_pos = (test_pos - pos_mean) / (pos_std + 1e-8)
    
    # Save normalization parameters
    np.save(data_dir / 'pos_mean.npy', pos_mean)
    np.save(data_dir / 'pos_std.npy', pos_std)
    
    return (train_csi, train_pos), (val_csi, val_pos), (test_csi, test_pos)

class CompetitionDataLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, sequence_length: int = 5):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.feature_shape = (32, 1024, 2)  # 32 subcarriers, 1024 time samples, 2 for complex
        
    def _parse_tfrecord(self, example_proto):
        """Parse TFRecord data format"""
        feature_description = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'pos-tachy': tf.io.FixedLenFeature([], tf.string),
            'gt-interp-age-tachy': tf.io.FixedLenFeature([], tf.float32),
            'cfo': tf.io.FixedLenFeature([], tf.string),
            'snr': tf.io.FixedLenFeature([], tf.string),
            'time': tf.io.FixedLenFeature([], tf.float32)
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Parse CSI data and ensure correct shape
        csi_data = tf.io.parse_tensor(parsed_features['csi'], out_type=tf.float32)
        # Reshape to (32, 1024, 2) - remove any extra dimensions
        csi_data = tf.reshape(csi_data, [32, 1024, 2])
        
        # Parse position data with float64 and convert to float32
        position = tf.io.parse_tensor(parsed_features['pos-tachy'], out_type=tf.float64)
        position = tf.cast(position, tf.float32)
        position = position[..., :2]  # Take only x and y coordinates
        
        return csi_data, position
    
    def _create_sequence_dataset(self, dataset, sequence_length: int):
        """Create sequences for trajectory prediction"""
        def make_sequence(data, label):
            return tf.data.Dataset.from_tensors((data, label))
            
        dataset = dataset.window(sequence_length, shift=1, drop_remainder=True)
        dataset = dataset.flat_map(lambda x, y: tf.data.Dataset.zip((
            x.batch(sequence_length),
            y.skip(sequence_length-1).batch(1))))
        return dataset
    
    def load_dataset(self, split: str):
        """Load dataset for a specific split. If subfolders (train_tfrecord, val_tfrecord, test_tfrecord) do not exist, use all .tfrecords files in data_dir and split internally."""
        subfolder = split + "_tfrecord"
        subfolder_path = os.path.join(self.data_dir, subfolder)
        if os.path.isdir(subfolder_path):
            file_pattern = os.path.join(subfolder_path, "*.tfrecords")
            dataset = tf.data.Dataset.list_files(file_pattern)
        else:
            # Fallback: use all .tfrecords files in data_dir and split internally
            all_files = tf.io.gfile.glob(os.path.join(self.data_dir, "*.tfrecords"))
            if not all_files:
                raise FileNotFoundError("No .tfrecords files found in " + self.data_dir)
            # Shuffle and split (e.g., 80% train, 10% val, 10% test)
            np.random.shuffle(all_files)
            n = len(all_files)
            if split == "train":
                files = all_files[: int(0.8 * n)]
            elif split == "val":
                files = all_files[int(0.8 * n) : int(0.9 * n)]
            else:  # test
                files = all_files[int(0.9 * n) :]
            dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if split == "train":
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize CSI data"""
        # Compute magnitude and phase
        magnitude = np.abs(data)
        phase = np.angle(data)
        
        # Normalize magnitude
        magnitude = (magnitude - np.mean(magnitude)) / np.std(magnitude)
        
        # Normalize phase to [-π, π]
        phase = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        
        # Combine normalized components
        return np.stack([magnitude, phase], axis=-1)
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare all datasets"""
        train_dataset = self.load_dataset("train")
        val_dataset = self.load_dataset("val")
        test_dataset = self.load_dataset("test")
        
        # Calculate steps per epoch
        train_steps = tf.data.experimental.cardinality(train_dataset).numpy()
        val_steps = tf.data.experimental.cardinality(val_dataset).numpy()
        test_steps = tf.data.experimental.cardinality(test_dataset).numpy()
        
        print("\nDataset sizes:")
        print(f"Training steps per epoch: {train_steps}")
        print(f"Validation steps: {val_steps}")
        print(f"Test steps: {test_steps}")
        
        return train_dataset, val_dataset, test_dataset
    
    @staticmethod
    def get_dataset_info(dataset: tf.data.Dataset) -> Dict[str, int]:
        """Get information about the dataset"""
        total_samples = 0
        batch_size = None
        
        # Get the batch size from the dataset
        for batch_x, batch_y in dataset.take(1):
            batch_size = batch_x.shape[0]
            break
        
        # Count total batches
        for _ in dataset:
            total_samples += 1
            
        return {
            'total_samples': total_samples * batch_size if batch_size else total_samples,
            'batch_size': batch_size,
            'steps_per_epoch': total_samples
        }
    
    def save_preprocessed_data(self, output_dir: str):
        """Save preprocessed data to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            dataset = self.load_dataset(split)
            
            # Create H5 file for this split
            with h5py.File(os.path.join(output_dir, f'{split}.h5'), 'w') as f:
                # Initialize datasets with unknown size
                data_list = []
                label_list = []
                
                for data, labels in dataset:
                    data_list.append(data.numpy())
                    label_list.append(labels.numpy())
                
                # Convert to arrays and save
                data_array = np.concatenate(data_list, axis=0)
                label_array = np.concatenate(label_list, axis=0)
                
                f.create_dataset('data', data=data_array)
                f.create_dataset('labels', data=label_array)

class SequenceDataLoader:
    """Data loader for sequence-based (trajectory-aware) training."""
    
    def __init__(self, data_dir: str, sequence_length: int = 10, batch_size: int = 32):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        
    def _parse_sequence_example(self, example_proto):
        """Parse a sequence example from TFRecord."""
        feature_description = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'position': tf.io.FixedLenFeature([], tf.string),
            'timestamp': tf.io.FixedLenFeature([], tf.float32),
            'cfo': tf.io.FixedLenFeature([], tf.float32),
            'snr': tf.io.FixedLenFeature([], tf.float32),
            'gt_interp_age': tf.io.FixedLenFeature([], tf.float32)
        }
        
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode CSI and position
        csi = tf.io.decode_raw(parsed['csi'], tf.float32)
        csi = tf.reshape(csi, [4, 8, 16, 2])
        
        position = tf.io.decode_raw(parsed['position'], tf.float32)
        position = tf.reshape(position, [2])
        
        return {
            'csi': csi,
            'position': position,
            'timestamp': parsed['timestamp'],
            'cfo': parsed['cfo'],
            'snr': parsed['snr'],
            'gt_interp_age': parsed['gt_interp_age']
        }
    
    def _create_sequences(self, dataset):
        """Create sequences from individual samples."""
        def create_sequence_windows(features):
            # Create sliding windows of sequence_length
            csi_seq = tf.signal.frame(
                features['csi'],
                frame_length=self.sequence_length,
                frame_step=1,
                pad_end=True
            )
            pos_seq = tf.signal.frame(
                features['position'],
                frame_length=self.sequence_length,
                frame_step=1,
                pad_end=True
            )
            return {
                'csi_sequence': csi_seq,
                'position_sequence': pos_seq,
                'timestamp': features['timestamp'],
                'cfo': features['cfo'],
                'snr': features['snr'],
                'gt_interp_age': features['gt_interp_age']
            }
        
        return dataset.map(create_sequence_windows)
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Prepare sequence-based datasets for training, validation, and testing."""
        # List all TFRecord files
        tfrecord_files = tf.io.gfile.glob(os.path.join(self.data_dir, '*.tfrecord'))
        
        # Create dataset from TFRecord files
        raw_dataset = tf.data.TFRecordDataset(tfrecord_files)
        parsed_dataset = raw_dataset.map(self._parse_sequence_example)
        
        # Create sequences
        sequence_dataset = self._create_sequences(parsed_dataset)
        
        # Split into train/val/test (80/10/10)
        total_size = tf.data.experimental.cardinality(sequence_dataset).numpy()
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        
        # Shuffle and split
        shuffled_dataset = sequence_dataset.shuffle(total_size, seed=42)
        
        train_ds = shuffled_dataset.take(train_size)
        val_ds = shuffled_dataset.skip(train_size).take(val_size)
        test_ds = shuffled_dataset.skip(train_size + val_size)
        
        # Batch and prefetch
        train_ds = train_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds, test_ds 

class DataLoader:
    """Load and preprocess data for model training and evaluation."""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.target_columns = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        try:
            data = pd.read_csv(self.data_dir / file_path)
            logging.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame,
                       feature_columns: List[str],
                       target_columns: List[str],
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Preprocess data for model training."""
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        
        # Split features and targets
        X = data[feature_columns].values
        y = data[target_columns].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
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
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Convert scaled predictions back to original scale."""
        return self.scaler.inverse_transform(predictions)
    
    def save_processed_data(self, data: Dict[str, Dict[str, np.ndarray]], output_dir: str):
        """Save processed data to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in data.items():
            np.save(output_path / f'X_{split_name}.npy', split_data['X'])
            np.save(output_path / f'y_{split_name}.npy', split_data['y'])
        
        # Save scaler parameters
        scaler_params = {
            'mean_': self.scaler.mean_,
            'scale_': self.scaler.scale_,
            'var_': self.scaler.var_,
            'n_samples_seen_': self.scaler.n_samples_seen_
        }
        np.save(output_path / 'scaler_params.npy', scaler_params)
        
        # Save column information
        column_info = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        np.save(output_path / 'column_info.npy', column_info)
        
        logging.info(f"Processed data saved to {output_path}")
    
    def load_processed_data(self, input_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Load processed data from files."""
        input_path = Path(input_dir)
        
        # Load data splits
        data = {}
        for split_name in ['train', 'val', 'test']:
            data[split_name] = {
                'X': np.load(input_path / f'X_{split_name}.npy'),
                'y': np.load(input_path / f'y_{split_name}.npy')
            }
        
        # Load scaler parameters
        scaler_params = np.load(input_path / 'scaler_params.npy', allow_pickle=True).item()
        self.scaler.mean_ = scaler_params['mean_']
        self.scaler.scale_ = scaler_params['scale_']
        self.scaler.var_ = scaler_params['var_']
        self.scaler.n_samples_seen_ = scaler_params['n_samples_seen_']
        
        # Load column information
        column_info = np.load(input_path / 'column_info.npy', allow_pickle=True).item()
        self.feature_columns = column_info['feature_columns']
        self.target_columns = column_info['target_columns']
        
        logging.info(f"Processed data loaded from {input_path}")
        return data
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        if not self.feature_columns or not self.target_columns:
            raise ValueError("No data has been loaded yet")
        
        return {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'n_features': len(self.feature_columns),
            'n_targets': len(self.target_columns)
        }
    
    def create_data_generator(self, data: Dict[str, np.ndarray],
                            batch_size: int = 32,
                            shuffle: bool = True) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Create a data generator for batch processing."""
        n_samples = len(data['X'])
        indices = np.arange(n_samples)
        
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            yield data['X'][batch_indices], data['y'][batch_indices]

def load_csi_data(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15, batch_size: int = 32):
    """Load and prepare CSI data for training, validation, and testing.
    
    Args:
        data_dir: Directory containing the data
        train_ratio: Ratio of data to use for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        batch_size: Batch size for the datasets
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create data loader
    data_loader = CompetitionDataLoader(data_dir, batch_size=batch_size)
    
    # Load and prepare datasets
    train_dataset, val_dataset, test_dataset = data_loader.prepare_data()
    
    return train_dataset, val_dataset, test_dataset 