import tensorflow as tf
import numpy as np
from data_loader import CSIDataLoader
from task3.model import GrantFreeModel
import os

DATA_PATH = 'data/dichasus-cf02.tfrecords'
CHECKPOINT_DIR = 'models/task3/'
BATCH_SIZE = 32

# R90 metric: radius of smallest circle containing 90% of errors
def r90_metric(y_true, y_pred):
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    return np.percentile(errors, 90)

def main():
    # Data
    loader = CSIDataLoader(DATA_PATH, batch_size=BATCH_SIZE)
    val_ds, _ = loader.load_dataset(is_training=False)

    # Model
    model = GrantFreeModel()
    model.local_model.load_weights(os.path.join(CHECKPOINT_DIR, 'best_local_model.h5'))
    model.central_model.load_weights(os.path.join(CHECKPOINT_DIR, 'best_central_model.h5'))

    # Collect predictions and ground truth
    y_true = []
    y_pred = []
    transmission_rates = []
    
    for batch in val_ds:
        csi = batch['csi']
        pos = batch['position']
        
        # Get transmission decisions
        transmit_probs = model.local_model(csi)
        transmit_mask = tf.cast(transmit_probs > 0.5, tf.float32)
        transmission_rates.append(tf.reduce_mean(transmit_mask).numpy())
        
        # Get predictions
        transmitted_csi = csi * transmit_mask
        preds = model.central_model(transmitted_csi)
        
        y_true.append(pos.numpy())
        y_pred.append(preds)
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    avg_transmission_rate = np.mean(transmission_rates)

    # Compute metrics
    mae = np.mean(np.abs(y_true - y_pred))
    r90 = r90_metric(y_true, y_pred)
    
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation R90: {r90:.4f}")
    print(f"Average Transmission Rate: {avg_transmission_rate:.4f}")

if __name__ == "__main__":
    main() 