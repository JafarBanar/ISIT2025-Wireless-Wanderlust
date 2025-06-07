import tensorflow as tf
import numpy as np
from data_loader import CSIDataLoader
from task1.model import SimpleLocalizationModel
import pandas as pd
import os

DATA_PATH = 'data/dichasus-cf02.tfrecords'  # Change to test set path as needed
CHECKPOINT_PATH = 'models/task1/best_model.h5'
BATCH_SIZE = 32
SUBMISSION_PATH = 'submission_task1.csv'

def main():
    loader = CSIDataLoader(DATA_PATH, batch_size=BATCH_SIZE)
    test_ds, _ = loader.load_dataset(is_training=False)

    model = SimpleLocalizationModel()
    model.build(input_shape=(None, 4, 8, 16, 2))
    model.load_weights(CHECKPOINT_PATH)

    predictions = []
    for batch in test_ds:
        csi = batch['csi']
        preds = model.predict(csi)
        predictions.append(preds)
    predictions = np.concatenate(predictions, axis=0)

    # Save as CSV (adjust columns as required by competition)
    df = pd.DataFrame(predictions, columns=['x', 'y'])
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

if __name__ == "__main__":
    main() 