import tensorflow as tf
import numpy as np
import os

class CSIDataLoader:
    def __init__(self, data_path, batch_size=32):
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Load normalization parameters if available
        self.pos_mean = None
        self.pos_std = None
        data_dir = os.path.dirname(data_path)
        if os.path.exists(os.path.join(data_dir, 'pos_mean.npy')):
            self.pos_mean = np.load(os.path.join(data_dir, 'pos_mean.npy'))
            self.pos_std = np.load(os.path.join(data_dir, 'pos_std.npy'))
        
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
        
        # Parse CSI data (float32)
        csi = tf.io.parse_tensor(parsed_features['csi'], out_type=tf.float32)
        # Reshape to match model input (32, 1024, 2)
        csi = tf.reshape(csi, [32, 1024, 2])
        # Normalize CSI data
        csi = (csi - tf.reduce_mean(csi)) / (tf.math.reduce_std(csi) + 1e-8)
        
        # Parse position data (float64)
        position = tf.io.parse_tensor(parsed_features['pos-tachy'], out_type=tf.float64)
        position = tf.cast(position, tf.float32)  # Convert to float32
        position = position[:2]  # Take only x,y coordinates
        
        # Normalize position data if parameters are available
        if self.pos_mean is not None and self.pos_std is not None:
            position = (position - self.pos_mean) / (self.pos_std + 1e-8)
        else:
            # Use default normalization if parameters not available
            position = (position - [-12, -14]) / [14, 12]  # Normalize to [0,1] range
        
        # Parse CFO and SNR (float32)
        cfo = tf.io.parse_tensor(parsed_features['cfo'], out_type=tf.float32)
        snr = tf.io.parse_tensor(parsed_features['snr'], out_type=tf.float32)
        
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

def create_dataset(pattern, batch_size, shuffle=False):
    files = tf.io.gfile.glob(pattern)
    loader = CSIDataLoader(files, batch_size)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(loader._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(1000)
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