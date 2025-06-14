import tensorflow as tf
import numpy as np
from data_loader import CSIDataLoader
from task3.model import GrantFreeModel
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 32
DATA_PATH = 'data/competition'
CHECKPOINT_DIR = '../models/task3/'

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
    model.local_model.load_weights(os.path.join(CHECKPOINT_DIR, 'best_local_model.weights.h5'))
    model.central_model.load_weights(os.path.join(CHECKPOINT_DIR, 'best_central_model.weights.h5'))

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
        
        # Reshape transmit_mask so that it broadcasts (e.g. [32, 1, 1, 1, 1]) for multiplication with CSI (shape [32, 4, 8, 16, 2]).
        transmit_mask = tf.reshape(transmit_mask, (tf.shape(transmit_mask)[0], 1, 1, 1, 1))
        transmitted_csi = csi * transmit_mask
        
        # Get predictions
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

    # Plot error distribution (histogram) for Task 3 evaluation
    errors = np.linalg.norm(y_true - y_pred, axis=1)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Error (Euclidean Distance)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution (Task 3)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'error_distribution.png'))
    plt.close()

if __name__ == "__main__":
    main() 