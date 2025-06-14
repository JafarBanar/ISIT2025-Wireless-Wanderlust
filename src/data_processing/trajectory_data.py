import tensorflow as tf
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path

class TrajectoryDataGenerator:
    """
    Data generator for trajectory-aware localization.
    Handles sequence creation and batching for temporal data.
    """
    
    def __init__(self,
                 csi_features: np.ndarray,  # (N, n_arrays=4, n_elements=8, n_freq=16, n_complex=2)
                 positions: np.ndarray,      # (N, 2)
                 sequence_length: int = 10,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 validation_split: float = 0.2):
        """
        Initialize the trajectory data generator.
        
        Args:
            csi_features: CSI features array (N, 4, 8, 16, 2)
            positions: Position array (N, 2)
            sequence_length: Length of trajectory sequences
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            validation_split: Fraction of data to use for validation
        """
        if len(csi_features) != len(positions):
            raise ValueError("Number of features and positions must match")
        
        # Verify competition data format
        if csi_features.shape[1:] != (4, 8, 16, 2):
            raise ValueError(
                f"Expected CSI features shape (N, 4, 8, 16, 2), got {csi_features.shape}"
            )
        
        self.csi_features = csi_features
        self.positions = positions
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create sequences
        self.feature_sequences, self.position_sequences = self._create_sequences()
        
        # Split into train and validation
        n_val = int(len(self.feature_sequences) * validation_split)
        if shuffle:
            indices = np.random.permutation(len(self.feature_sequences))
            self.feature_sequences = self.feature_sequences[indices]
            self.position_sequences = self.position_sequences[indices]
        
        self.train_features = self.feature_sequences[n_val:]
        self.train_positions = self.position_sequences[n_val:]
        self.val_features = self.feature_sequences[:n_val]
        self.val_positions = self.position_sequences[:n_val]
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for trajectory prediction."""
        n_sequences = len(self.csi_features) - self.sequence_length
        
        # Get feature shape excluding the first dimension
        feature_shape = self.csi_features.shape[1:]
        
        feature_sequences = np.zeros((
            n_sequences,
            self.sequence_length,
            *feature_shape  # Preserve all feature dimensions
        ))
        
        position_sequences = np.zeros((
            n_sequences,
            self.sequence_length,
            2
        ))
        
        for i in range(n_sequences):
            feature_sequences[i] = self.csi_features[i:i + self.sequence_length]
            position_sequences[i] = self.positions[i:i + self.sequence_length]
        
        return feature_sequences, position_sequences
    
    def get_train_dataset(self) -> tf.data.Dataset:
        """Get training dataset."""
        return self._create_dataset(
            self.train_features,
            self.train_positions,
            shuffle=self.shuffle
        )
    
    def get_val_dataset(self) -> tf.data.Dataset:
        """Get validation dataset."""
        return self._create_dataset(
            self.val_features,
            self.val_positions,
            shuffle=False
        )
    
    def _create_dataset(self,
                       features: np.ndarray,
                       positions: np.ndarray,
                       shuffle: bool) -> tf.data.Dataset:
        """Create a TensorFlow dataset."""
        # Create input sequences (features and previous positions)
        x = {
            'csi_features': features,
            'prev_positions': positions[:, :-1]  # All but last position
        }
        
        # Create targets dictionary
        y = {
            'positions': positions.astype(np.float32),  # (batch, seq_len, 2)
            'occupancy_mask': np.ones((len(features), self.sequence_length, 4, 8, 1), dtype=np.float32),
            'interference': np.zeros((len(features), self.sequence_length, 1), dtype=np.float32)  # Placeholder
        }
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        # Drop remainder to ensure consistent batch size
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset

class TrajectoryTFRecordHandler:
    """
    Handler for reading and writing trajectory data in TFRecord format.
    """
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
    
    def _bytes_feature(self, value):
        """Convert value to bytes feature."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    def _float_feature(self, value):
        """Convert value to float feature."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    def serialize_sequence(self,
                         features: np.ndarray,
                         positions: np.ndarray) -> tf.train.Example:
        """
        Serialize a single trajectory sequence.
        
        Args:
            features: CSI features for sequence (sequence_length, feature_dim)
            positions: Positions for sequence (sequence_length, 2)
        """
        feature_dict = {
            'features': self._bytes_feature(tf.io.serialize_tensor(features)),
            'positions': self._bytes_feature(tf.io.serialize_tensor(positions))
        }
        
        return tf.train.Example(
            features=tf.train.Features(feature=feature_dict)
        )
    
    def write_tfrecord(self,
                      features: np.ndarray,
                      positions: np.ndarray,
                      output_path: str):
        """
        Write trajectory data to TFRecord file.
        
        Args:
            features: CSI features array (N, feature_dim)
            positions: Position array (N, 2)
            output_path: Path to save TFRecord file
        """
        # Create sequences
        n_sequences = len(features) - self.sequence_length
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for i in range(n_sequences):
                feature_seq = features[i:i + self.sequence_length]
                position_seq = positions[i:i + self.sequence_length]
                
                example = self.serialize_sequence(feature_seq, position_seq)
                writer.write(example.SerializeToString())
    
    def parse_tfrecord(self, example_proto):
        """Parse TFRecord example."""
        feature_description = {
            'features': tf.io.FixedLenFeature([], tf.string),
            'positions': tf.io.FixedLenFeature([], tf.string)
        }
        
        example = tf.io.parse_single_example(example_proto, feature_description)
        
        # Parse features and positions
        features = tf.io.parse_tensor(example['features'], out_type=tf.float32)
        positions = tf.io.parse_tensor(example['positions'], out_type=tf.float32)
        
        # Create model inputs
        x = {
            'csi_features': features,
            'prev_positions': positions[:-1]  # All but last position
        }
        
        # Target is the last position
        y = positions[-1]
        
        return x, y
    
    def load_dataset(self,
                    tfrecord_path: str,
                    batch_size: int = 32,
                    shuffle: bool = True) -> tf.data.Dataset:
        """
        Load trajectory dataset from TFRecord file.
        
        Args:
            tfrecord_path: Path to TFRecord file
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        dataset = dataset.map(
            self.parse_tfrecord,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset 