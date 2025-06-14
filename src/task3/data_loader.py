import tensorflow as tf
import os
import numpy as np

class CSIDataLoader:
    def __init__(self, data_dir, batch_size=32):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def load_dataset(self, is_training=True):
        csi_features = np.load(os.path.join(self.data_dir, "csi_features.npy")).astype(np.float32)
        positions = np.load(os.path.join(self.data_dir, "positions.npy")).astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices({"csi": csi_features, "position": positions})
        ds = ds.batch(self.batch_size, drop_remainder=is_training)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        if is_training:
            ds_train, ds_val = tf.keras.utils.split_dataset(ds, left_size=0.8, shuffle=True, seed=42)
            return ds_train, ds_val
        else:
            return ds, None 