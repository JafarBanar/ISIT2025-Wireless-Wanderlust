import tensorflow as tf
import os
from task2.data_loader import TrajectoryDataLoader
from task2.model import TrajectoryAwareModel
from utils.plotting import plot_loss

BATCH_SIZE = 32
EPOCHS = 20
SEQUENCE_LENGTH = 10
DATA_PATH = 'data/dichasus-cf02.tfrecords'
CHECKPOINT_DIR = 'models/task2/'

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def main():
    # Data
    loader = TrajectoryDataLoader(
        DATA_PATH,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )
    train_ds, _ = loader.load_dataset(is_training=True)
    
    # Model
    model = TrajectoryAwareModel()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mae',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(CHECKPOINT_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Training
    history = model.fit(
        train_ds,
        validation_data=train_ds,  # Use same dataset for validation
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Plot training curves
    plot_loss(
        history.history['loss'],
        history.history['val_loss'],
        save_path=os.path.join(CHECKPOINT_DIR, 'training_loss.png')
    )

if __name__ == "__main__":
    main() 