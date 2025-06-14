import tensorflow as tf
import h5py
import numpy as np
import os
from glob import glob

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def convert_h5_to_tfrecord(input_file: str, output_file: str):
    """Convert H5 file to TFRecord format"""
    with h5py.File(input_file, 'r') as h5f:
        # Read CSI data and positions
        csi_data = h5f['csi_data'][:]
        positions = h5f['positions'][:]
        timestamps = h5f['timestamps'][:]
        
        with tf.io.TFRecordWriter(output_file) as writer:
            for i in range(len(positions)):
                # Create feature dictionary
                feature = {
                    'csi_data': _bytes_feature(tf.io.serialize_tensor(csi_data[i].astype(np.float32))),
                    'position': _bytes_feature(tf.io.serialize_tensor(positions[i].astype(np.float32))),
                    'timestamp': _int64_feature(int(timestamps[i]))
                }
                
                # Create Example
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                
                # Write to TFRecord file
                writer.write(example.SerializeToString())

def convert_directory(input_dir: str, output_dir: str):
    """Convert all H5 files in a directory to TFRecord format"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all H5 files
    h5_files = glob(os.path.join(input_dir, '*.h5'))
    
    for h5_file in h5_files:
        # Create output filename
        base_name = os.path.splitext(os.path.basename(h5_file))[0]
        output_file = os.path.join(output_dir, f'{base_name}.tfrecord')
        
        print(f"Converting {h5_file} to {output_file}")
        convert_h5_to_tfrecord(h5_file, output_file)

def main():
    # Convert training data
    convert_directory('data/competition/train', 'data/competition/train_tfrecord')
    
    # Convert validation data
    convert_directory('data/competition/val', 'data/competition/val_tfrecord')
    
    # Convert test data
    convert_directory('data/competition/test', 'data/competition/test_tfrecord')
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 