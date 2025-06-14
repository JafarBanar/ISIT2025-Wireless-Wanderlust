import tensorflow as tf
import os
import numpy as np

def inspect_tfrecord(file_path):
    dataset = tf.data.TFRecordDataset(file_path)
    for raw_record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print("Features in the TFRecord:")
        for key, feature in example.features.feature.items():
            kind = feature.WhichOneof('kind')
            print(f"Feature: {key}")
            print(f"Type: {kind}")
            if kind == 'bytes_list':
                print(f"Shape: {len(feature.bytes_list.value)}")
                if key in ['pos-tachy', 'csi']:
                    raw = feature.bytes_list.value[0]
                    print(f"Raw bytes length: {len(raw)}")
                    print(f"First 34 bytes as int: {list(raw[:34])}")
                    # Try to decode as float32
                    try:
                        arr = np.frombuffer(raw, dtype=np.float32)
                        print(f"Decoded as float32: shape {arr.shape}, values: {arr[:10]}")
                    except Exception as e:
                        print(f"Could not decode as float32: {e}")
                    # Try to decode as float64
                    try:
                        arr = np.frombuffer(raw, dtype=np.float64)
                        print(f"Decoded as float64: shape {arr.shape}, values: {arr[:5]}")
                    except Exception as e:
                        print(f"Could not decode as float64: {e}")
                    # Try to decode as int32
                    try:
                        arr = np.frombuffer(raw, dtype=np.int32)
                        print(f"Decoded as int32: shape {arr.shape}, values: {arr[:10]}")
                    except Exception as e:
                        print(f"Could not decode as int32: {e}")
            elif kind == 'float_list':
                print(f"Shape: {len(feature.float_list.value)}")
                print(f"Values: {feature.float_list.value}")
            elif kind == 'int64_list':
                print(f"Shape: {len(feature.int64_list.value)}")
                print(f"Values: {feature.int64_list.value}")
            print("---")

if __name__ == "__main__":
    data_dir = "data/csi_data"
    tfrecord_files = [f for f in os.listdir(data_dir) if f.endswith('.tfrecords')]
    
    for file in tfrecord_files:
        print(f"\nInspecting {file}:")
        inspect_tfrecord(os.path.join(data_dir, file)) 