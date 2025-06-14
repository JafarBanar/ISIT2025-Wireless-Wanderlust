import tensorflow as tf
import numpy as np
from data_loader import CSIDataLoader

class TrajectoryDataLoader:
    """Data loader for trajectory-aware localization."""
    
    def __init__(self, data_path, sequence_length=10, batch_size=32):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.base_loader = CSIDataLoader(data_path, batch_size=1)  # Load single samples
        
    def _create_sequences(self, dataset):
        """Create sequences of CSI measurements."""
        sequences = []
        current_sequence = []
        
        for sample in dataset:
            current_sequence.append(sample)
            
            if len(current_sequence) == self.sequence_length:
                sequences.append(current_sequence)
                current_sequence = current_sequence[1:]  # Sliding window
        
        # Pad the last sequence if needed
        if current_sequence:
            while len(current_sequence) < self.sequence_length:
                current_sequence.append(current_sequence[-1])  # Repeat last sample
            sequences.append(current_sequence)
        
        return sequences
    
    def _prepare_sequence_batch(self, sequence):
        """Prepare a batch of sequences for training."""
        # Stack CSI measurements
        csi = tf.stack([s['csi'] for s in sequence])
        
        # Get metadata from the last sample in sequence
        position = sequence[-1]['position']
        timestamp = sequence[-1]['timestamp']
        cfo = sequence[-1]['cfo']
        snr = sequence[-1]['snr']
        gt_interp_age = sequence[-1]['gt-interp-age-tachy']
        
        return {
            'csi': csi,
            'position': position,
            'timestamp': timestamp,
            'cfo': cfo,
            'snr': snr,
            'gt-interp-age-tachy': gt_interp_age
        }
    
    def load_dataset(self, is_training=True):
        """Load and prepare the dataset for training or evaluation."""
        # Load base dataset
        base_ds, _ = self.base_loader.load_dataset(is_training=is_training)
        
        # Create sequences
        sequences = self._create_sequences(base_ds)
        
        # Convert to TensorFlow dataset
        ds = tf.data.Dataset.from_tensor_slices(sequences)
        
        # Prepare batches
        ds = ds.map(
            lambda x: tf.py_function(
                self._prepare_sequence_batch,
                [x],
                [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle and batch
        if is_training:
            ds = ds.shuffle(1000)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds, None  # Return None for validation as we use the same dataset 