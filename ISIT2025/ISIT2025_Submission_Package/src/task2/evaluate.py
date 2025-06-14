import tensorflow as tf
import numpy as np
from task2.data_loader import TrajectoryDataLoader
from task2.model import TrajectoryAwareModel
import os

DATA_PATH = 'data/dichasus-cf02.tfrecords'
CHECKPOINT_DIR = 'models/task2/'
BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

# R90 metric: radius of smallest circle containing 90% of errors
def r90_metric(y_true, y_pred):
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    return np.percentile(errors, 90)

def main():
    # Data
    loader = TrajectoryDataLoader(
        DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    val_ds, _ = loader.load_dataset(is_training=False)
    
    # Model
    model = TrajectoryAwareModel()
    model.load_weights(os.path.join(CHECKPOINT_DIR, 'best_model.h5'))
    
    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    
    for batch in val_ds:
        csi = batch['csi']
        pos = batch['position']
        preds = model(csi)
        
        y_true.append(pos.numpy())
        y_pred.append(preds)
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    r90 = r90_metric(y_true, y_pred)
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R90: {r90:.4f}")

if __name__ == "__main__":
    main() 