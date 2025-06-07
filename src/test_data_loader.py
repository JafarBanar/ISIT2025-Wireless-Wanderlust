import tensorflow as tf
from data_loader import CSIDataLoader
import numpy as np

def test_data_loader():
    # Initialize data loader with the first dataset
    data_loader = CSIDataLoader('data/dichasus-cf02.tfrecords')
    
    # Load both labeled and unlabeled datasets
    labeled_dataset, unlabeled_dataset = data_loader.load_dataset(is_training=True)
    
    # Get a batch from the labeled dataset
    labeled_batch = next(iter(labeled_dataset))
    
    # Print shapes and information
    print("\nLabeled Data Batch:")
    print("CSI shape:", labeled_batch['csi'].shape)
    print("Position shape:", labeled_batch['position'].shape)
    print("Timestamp shape:", labeled_batch['timestamp'].shape)
    print("CFO shape:", labeled_batch['cfo'].shape)
    print("SNR shape:", labeled_batch['snr'].shape)
    print("GT Interp Age shape:", labeled_batch['gt_interp_age'].shape)
    
    # Print some sample values
    print("\nSample CSI values (first element):")
    print(labeled_batch['csi'][0, 0, 0, 0, :])  # First sample, first antenna, first element, first frequency
    
    print("\nSample position values (first element):")
    print(labeled_batch['position'][0])  # First sample position
    
    print("\nSample CFO values (first element):")
    print(labeled_batch['cfo'][0][:5])  # First 5 values of CFO
    
    print("\nSample SNR values (first element):")
    print(labeled_batch['snr'][0][:5])  # First 5 values of SNR

if __name__ == "__main__":
    test_data_loader() 