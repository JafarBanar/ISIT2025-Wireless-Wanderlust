import tensorflow as tf
import os
import sys
from pathlib import Path

def scan_tfrecord(tfrecord_path, max_records=10):
    """
    Scan a TFRecord file and print the dtype of each field for each record.
    
    Args:
        tfrecord_path: Path to the TFRecord file
        max_records: Maximum number of records to scan (default: 10)
    """
    print(f"\nScanning TFRecord file: {tfrecord_path}")
    print("=" * 80)
    
    # Feature description for parsing
    feature_description = {
        'pos-tachy': tf.io.FixedLenFeature([], tf.string),
        'gt-interp-age-tachy': tf.io.FixedLenFeature([], tf.float32),
        'cfo': tf.io.FixedLenFeature([], tf.string),
        'snr': tf.io.FixedLenFeature([], tf.string),
        'time': tf.io.FixedLenFeature([], tf.float32),
        'csi': tf.io.FixedLenFeature([], tf.string)
    }
    
    # Create dataset
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    # Counter for records
    record_count = 0
    
    # Try parsing each record
    for raw_record in dataset:
        if record_count >= max_records:
            break
            
        try:
            # Parse the record
            example = tf.io.parse_single_example(raw_record, feature_description)
            
            print(f"\nRecord {record_count + 1}:")
            print("-" * 40)
            
            # Try parsing each field with both float32 and float64
            for field_name in ['csi', 'pos-tachy', 'cfo', 'snr']:
                try:
                    # Try float32 first
                    tensor = tf.io.parse_tensor(example[field_name], out_type=tf.float32)
                    print(f"{field_name}: dtype={tensor.dtype}, shape={tensor.shape}")
                except tf.errors.InvalidArgumentError as e:
                    if "Type mismatch" in str(e):
                        # If float32 fails, try float64
                        try:
                            tensor = tf.io.parse_tensor(example[field_name], out_type=tf.float64)
                            print(f"{field_name}: dtype={tensor.dtype}, shape={tensor.shape} (was float64)")
                        except tf.errors.InvalidArgumentError as e2:
                            print(f"{field_name}: Error parsing as float64: {str(e2)}")
                    else:
                        print(f"{field_name}: Error parsing as float32: {str(e)}")
            
            # Print other fields
            print(f"time: dtype={example['time'].dtype}")
            print(f"gt-interp-age-tachy: dtype={example['gt-interp-age-tachy'].dtype}")
            
            record_count += 1
            
        except Exception as e:
            print(f"Error parsing record {record_count + 1}: {str(e)}")
            record_count += 1
            continue
    
    print("\n" + "=" * 80)
    print(f"Scanned {record_count} records")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scan_tfrecord.py <tfrecord_path>")
        sys.exit(1)
    
    tfrecord_path = sys.argv[1]
    if not os.path.exists(tfrecord_path):
        print(f"Error: File {tfrecord_path} does not exist")
        sys.exit(1)
    
    scan_tfrecord(tfrecord_path)

if __name__ == '__main__':
    main() 