import tensorflow as tf
import numpy as np
import pandas as pd
from src.models.vanilla_localization import VanillaLocalizationModel
import os

def load_test_data(tfrecord_path):
    """Load test data from TFRecord file."""
    def parse_function(example_proto):
        feature_description = {
            'csi': tf.io.FixedLenFeature([], tf.string),
            'position': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        csi = tf.io.decode_raw(parsed_features['csi'], tf.float32)
        csi = tf.reshape(csi, (64, 64, 1))  # Reshape to match model input
        return csi

    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(1)
    return dataset

def generate_predictions():
    # Model parameters
    input_shape = (32, 1024, 2)  # CSI data shape
    num_classes = 2  # x, y coordinates
    
    # Load the model using the fixed classmethod
    model = VanillaLocalizationModel.load_model('models/vanilla_model.h5', input_shape=input_shape, num_classes=num_classes)
    
    # Load test data
    test_data = load_test_data('data/test_tfrecord/*.tfrecords')
    
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'x': predictions[:, 0],
        'y': predictions[:, 1],
        'model': 'vanilla'
    })
    
    # Save to CSV
    os.makedirs('submissions', exist_ok=True)
    submission_df.to_csv('submissions/submission.csv', index=False)
    print("Submission file generated at submissions/submission.csv")

if __name__ == "__main__":
    generate_predictions() 