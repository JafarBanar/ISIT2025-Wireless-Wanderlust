import numpy as np
import pandas as pd
from keras.models import load_model
from utils.data_loader import load_csi_data
import os

def load_new_data(data_dir, batch_size):
    """Load new CSI data for inference.
    
    Args:
        data_dir: Directory containing new CSI data (.tfrecords files)
        batch_size: Batch size for inference
        
    Returns:
        Dataset for inference
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Data directory '{data_dir}' not found. "
            f"Please place your new CSI data (.tfrecords files) in this directory."
        )
    
    # Check if directory is empty
    tfrecord_files = [f for f in os.listdir(data_dir) if f.endswith('.tfrecords')]
    if not tfrecord_files:
        raise FileNotFoundError(
            f"No .tfrecords files found in '{data_dir}'. "
            f"Please add your CSI data files to this directory."
        )
    
    # Load data
    ds = load_csi_data(
        data_dir=data_dir,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=batch_size
    )[0]
    return ds

def main():
    # Configuration
    new_data_dir = 'data/new_csi_data'
    batch_size = 8
    
    try:
        # Load new data
        print(f"Loading data from {new_data_dir}...")
        new_ds = load_new_data(new_data_dir, batch_size)
        
        # Load trained model
        print("Loading trained model...")
        model = load_model('results/basic_localization/best_model.keras', compile=False)
        
        # Run predictions
        print("Running predictions...")
        predictions = []
        for X_batch, _ in new_ds:
            preds = model.predict(X_batch)
            predictions.extend(preds)
        
        # Save predictions to CSV
        output_dir = 'results/batch_inference'
        os.makedirs(output_dir, exist_ok=True)
        pred_df = pd.DataFrame(predictions, columns=['x_pred', 'y_pred'])
        output_file = os.path.join(output_dir, 'predictions.csv')
        pred_df.to_csv(output_file, index=False)
        print(f'Predictions saved to {output_file}')
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("\nTo use this script:")
        print("1. Create directory 'data/new_csi_data'")
        print("2. Place your new CSI data (.tfrecords files) in this directory")
        print("3. Run this script again")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 