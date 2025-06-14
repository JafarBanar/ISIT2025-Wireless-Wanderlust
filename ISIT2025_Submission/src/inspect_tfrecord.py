import tensorflow as tf
import numpy as np

def inspect_tfrecord(file_path):
    """Inspect the structure of a TFRecord file"""
    dataset = tf.data.TFRecordDataset(file_path)
    
    # Get the first record
    for record in dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        
        print("Features in TFRecord:")
        for key in example.features.feature.keys():
            feature = example.features.feature[key]
            
            # Determine feature type
            if feature.HasField('bytes_list'):
                print(f"{key}: bytes_list")
                # Try to parse tensor
                try:
                    tensor = tf.io.parse_tensor(feature.bytes_list.value[0], out_type=tf.float32)
                    print(f"  Shape: {tensor.shape}")
                except:
                    print("  Could not parse tensor")
            elif feature.HasField('float_list'):
                print(f"{key}: float_list")
                print(f"  Values: {feature.float_list.value}")
            elif feature.HasField('int64_list'):
                print(f"{key}: int64_list")
                print(f"  Values: {feature.int64_list.value}")

if __name__ == "__main__":
    inspect_tfrecord("data/csi_data/dichasus-cf02.tfrecords") 