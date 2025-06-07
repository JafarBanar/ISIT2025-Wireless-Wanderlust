import tensorflow as tf
import numpy as np
import os

class CSIDataLoader:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        
    def _parse_tfrecord(self, example_proto):
        feature_description = {
            'pos-tachy': tf.io.FixedLenFeature([], tf.string),
            'gt-interp-age-tachy': tf.io.FixedLenFeature([], tf.float32),
            'cfo': tf.io.FixedLenFeature([], tf.string),
            'snr': tf.io.FixedLenFeature([], tf.string),
            'time': tf.io.FixedLenFeature([], tf.float32),
            'csi': tf.io.FixedLenFeature([], tf.string)
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Decode CSI data from protobuf
        csi_bytes = parsed_features['csi']
        # Use tf.strings.substr to get the data portion
        csi_data = tf.strings.substr(csi_bytes, 16, -1)  # Skip first 16 bytes
        csi = tf.io.decode_raw(csi_data, tf.float32)
        csi = tf.reshape(csi, [4, 8, 16, 2])  # Reshape to match the expected dimensions
        
        # Normalize CSI data
        csi = tf.cast(csi, tf.float32)
        csi = (csi - tf.reduce_mean(csi)) / (tf.math.reduce_std(csi) + 1e-8)
        
        # Decode position from pos-tachy protobuf
        pos_bytes = parsed_features['pos-tachy']
        # Use tf.strings.substr to get the data portion
        pos_data = tf.strings.substr(pos_bytes, 8, -1)  # Skip first 8 bytes
        position = tf.io.decode_raw(pos_data, tf.float32)
        position = tf.reshape(position, [2])  # Assuming it's [x, y]
        
        # Normalize position coordinates
        position = tf.cast(position, tf.float32)
        # Scale to [0, 1] range based on competition coordinate ranges
        position = (position - [-12, -14]) / [14, 12]  # x: [-12, 2], y: [-14, -2]
        
        # Decode CFO and SNR from protobuf
        cfo_bytes = parsed_features['cfo']
        snr_bytes = parsed_features['snr']
        # Use tf.strings.substr to get the data portion
        cfo_data = tf.strings.substr(cfo_bytes, 8, -1)  # Skip first 8 bytes
        snr_data = tf.strings.substr(snr_bytes, 8, -1)  # Skip first 8 bytes
        cfo = tf.io.decode_raw(cfo_data, tf.float32)
        snr = tf.io.decode_raw(snr_data, tf.float32)
        
        return {
            'csi': csi,
            'position': position,
            'timestamp': parsed_features['time'],
            'cfo': cfo,
            'snr': snr,
            'gt_interp_age': parsed_features['gt-interp-age-tachy']
        }
    
    def load_dataset(self, is_training=True):
        dataset = tf.data.TFRecordDataset(self.data_path)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(1000)
        
        # For now, treat all data as labeled since we don't have an explicit label
        labeled_dataset = dataset
        unlabeled_dataset = dataset.take(0)  # Empty dataset for unlabeled data
        
        # Batch both datasets
        labeled_dataset = labeled_dataset.batch(self.batch_size)
        unlabeled_dataset = unlabeled_dataset.batch(self.batch_size)
        
        # Prefetch for better performance
        labeled_dataset = labeled_dataset.prefetch(tf.data.AUTOTUNE)
        unlabeled_dataset = unlabeled_dataset.prefetch(tf.data.AUTOTUNE)
        
        return labeled_dataset, unlabeled_dataset

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