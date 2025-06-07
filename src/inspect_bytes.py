import tensorflow as tf
import numpy as np

def inspect_bytes(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        print("\nDetailed inspection of bytes fields:")
        for key, feature in example.features.feature.items():
            if feature.WhichOneof('kind') == 'bytes_list':
                print(f"\n{key}:")
                raw_bytes = feature.bytes_list.value[0]
                print(f"Total length: {len(raw_bytes)} bytes")
                print("First 100 bytes (hex):")
                print(' '.join(f'{b:02x}' for b in raw_bytes[:100]))
                
                # Try to interpret as different types
                try:
                    # Try as float32
                    floats = np.frombuffer(raw_bytes, dtype=np.float32)
                    print(f"\nAs float32 (first 5 values):")
                    print(floats[:5])
                except:
                    print("Could not interpret as float32")
                
                try:
                    # Try as int32
                    ints = np.frombuffer(raw_bytes, dtype=np.int32)
                    print(f"\nAs int32 (first 5 values):")
                    print(ints[:5])
                except:
                    print("Could not interpret as int32")
                
                try:
                    # Try as uint8
                    bytes_array = np.frombuffer(raw_bytes, dtype=np.uint8)
                    print(f"\nAs uint8 (first 10 values):")
                    print(bytes_array[:10])
                except:
                    print("Could not interpret as uint8")
                
                # Print ASCII representation if possible
                try:
                    ascii_str = raw_bytes.decode('ascii', errors='replace')
                    print(f"\nASCII representation (first 100 chars):")
                    print(ascii_str[:100])
                except:
                    print("Could not decode as ASCII")
                
                print("-" * 80)
        break

if __name__ == "__main__":
    inspect_bytes('data/dichasus-cf02.tfrecords') 