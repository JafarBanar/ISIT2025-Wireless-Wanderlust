import tensorflow as tf
import os
from data_loader import CSIDataLoader
from task3.model import GrantFreeModel
from utils.plotting import plot_loss
import numpy as np

BATCH_SIZE = 32
EPOCHS = 20
DATA_PATH = 'data/dichasus-cf02.tfrecords'
CHECKPOINT_DIR = 'models/task3/'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class GrantFreeTrainer:
    def __init__(self, model, train_ds, val_ds=None):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.local_optimizer = tf.keras.optimizers.Adam(1e-3)
        self.central_optimizer = tf.keras.optimizers.Adam(1e-3)
        
    @tf.function
    def train_step(self, csi, position):
        with tf.GradientTape() as local_tape, tf.GradientTape() as central_tape:
            # Local model decides whether to transmit
            transmit_probs = self.model.local_model(csi, training=True)
            transmit_mask = tf.cast(transmit_probs > 0.5, tf.float32)
            
            # Only transmitted CSI reaches the central model
            transmitted_csi = csi * transmit_mask
            
            # Central model predicts position
            position_pred = self.model.central_model(transmitted_csi, training=True)
            
            # Calculate losses
            transmission_loss = tf.keras.losses.binary_crossentropy(
                transmit_mask, transmit_probs
            )
            position_loss = tf.keras.losses.mae(position, position_pred)
            
            # Combined loss
            total_loss = transmission_loss + position_loss
        
        # Update local model
        local_gradients = local_tape.gradient(
            total_loss, self.model.local_model.trainable_variables
        )
        self.local_optimizer.apply_gradients(
            zip(local_gradients, self.model.local_model.trainable_variables)
        )
        
        # Update central model
        central_gradients = central_tape.gradient(
            total_loss, self.model.central_model.trainable_variables
        )
        self.central_optimizer.apply_gradients(
            zip(central_gradients, self.model.central_model.trainable_variables)
        )
        
        return {
            'total_loss': total_loss,
            'transmission_loss': transmission_loss,
            'position_loss': position_loss
        }
    
    def train(self, epochs):
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            epoch_losses = []
            for batch in self.train_ds:
                csi = batch['csi']
                position = batch['position']
                losses = self.train_step(csi, position)
                epoch_losses.append(losses)
            
            # Calculate average training losses
            avg_losses = {
                k: np.mean([l[k] for l in epoch_losses])
                for k in epoch_losses[0].keys()
            }
            train_losses.append(avg_losses)
            
            # Validation
            if self.val_ds is not None:
                val_losses_epoch = []
                for batch in self.val_ds:
                    csi = batch['csi']
                    position = batch['position']
                    losses = self.train_step(csi, position)
                    val_losses_epoch.append(losses)
                
                avg_val_losses = {
                    k: np.mean([l[k] for l in val_losses_epoch])
                    for k in val_losses_epoch[0].keys()
                }
                val_losses.append(avg_val_losses)
                
                # Save best model
                if avg_val_losses['total_loss'] < best_val_loss:
                    best_val_loss = avg_val_losses['total_loss']
                    self.model.local_model.save_weights(
                        os.path.join(CHECKPOINT_DIR, 'best_local_model.h5')
                    )
                    self.model.central_model.save_weights(
                        os.path.join(CHECKPOINT_DIR, 'best_central_model.h5')
                    )
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train - Total Loss: {avg_losses['total_loss']:.4f}, "
                  f"Transmission Loss: {avg_losses['transmission_loss']:.4f}, "
                  f"Position Loss: {avg_losses['position_loss']:.4f}")
            if self.val_ds is not None:
                print(f"Val   - Total Loss: {avg_val_losses['total_loss']:.4f}, "
                      f"Transmission Loss: {avg_val_losses['transmission_loss']:.4f}, "
                      f"Position Loss: {avg_val_losses['position_loss']:.4f}")
        
        return train_losses, val_losses

def main():
    # Data
    loader = CSIDataLoader(DATA_PATH, batch_size=BATCH_SIZE)
    train_ds, val_ds = loader.load_dataset(is_training=True)
    
    # Model
    model = GrantFreeModel()
    
    # Training
    trainer = GrantFreeTrainer(model, train_ds, val_ds)
    train_losses, val_losses = trainer.train(EPOCHS)
    
    # Plot training curves
    plot_loss(
        [l['total_loss'] for l in train_losses],
        [l['total_loss'] for l in val_losses] if val_losses else None,
        save_path=os.path.join(CHECKPOINT_DIR, 'training_loss.png')
    )

if __name__ == "__main__":
    main() 