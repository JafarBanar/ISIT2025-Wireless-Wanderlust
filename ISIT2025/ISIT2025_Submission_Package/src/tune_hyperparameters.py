import tensorflow as tf
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from src.models.basic_localization import create_basic_localization_model
from src.utils.data_loader import load_csi_data
import numpy as np

# Define the model-building function for KerasTuner
def build_model(hp):
    input_shape = (32, 1024, 2)
    num_classes = 2
    l2_reg = hp.Float('l2_reg', 1e-5, 1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', [1e-4, 5e-4, 1e-3, 5e-3])
    
    model = create_basic_localization_model(
        input_shape=input_shape,
        num_classes=num_classes,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    return model

def main():
    # Load data (use a small subset for tuning)
    train_ds, val_ds, _ = load_csi_data(
        data_dir='data/csi_data',
        batch_size=32,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    # For speed, take a small subset
    train_ds = train_ds.take(256)
    val_ds = val_ds.take(64)

    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        executions_per_trial=1,
        directory='results/hyperparam_tuning',
        project_name='basic_localization'
    )

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        verbose=2
    )

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print("Best hyperparameters found:")
    print(f"Learning rate: {best_hp.get('learning_rate')}")
    print(f"L2 regularization: {best_hp.get('l2_reg')}")
    print(f"Dropout rate: {best_hp.get('dropout_rate')}")

if __name__ == '__main__':
    main() 