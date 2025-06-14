import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from pathlib import Path
from src.utils.data_loader import load_csi_data
from src.models.basic_localization import create_basic_localization_model

def create_improved_model(input_shape, num_outputs=2):
    """Create an improved model with additional layers and regularization."""
    model = models.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block with batch normalization
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Second convolutional block
        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third convolutional block
        layers.Conv1D(256, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Global average pooling
        layers.GlobalAveragePooling1D(),
        
        # Dense layers with regularization
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_outputs)
    ])
    
    return model

def create_residual_model(input_shape, num_outputs=2):
    """Create a model with residual connections."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv1D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Residual blocks
    for filters in [128, 256, 512]:
        # First convolution in residual block
        y = layers.Conv1D(filters, 3, padding='same')(x)
        y = layers.BatchNormalization()(y)
        y = layers.Activation('relu')(y)
        
        # Second convolution in residual block
        y = layers.Conv1D(filters, 3, padding='same')(y)
        y = layers.BatchNormalization()(y)
        
        # Adjust dimensions if needed
        if x.shape[-1] != filters:
            x = layers.Conv1D(filters, 1, padding='same')(x)
        
        # Add residual connection
        x = layers.Add()([x, y])
        x = layers.Activation('relu')(x)
    
    # Global average pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_outputs)(x)
    
    return models.Model(inputs, outputs)

def train_improved_model(model, train_ds, val_ds, epochs=100):
    """Train the improved model with custom learning rate schedule."""
    # Create results directory
    results_dir = Path("results/improved_localization")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    
    # Optimizer with learning rate schedule
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(results_dir / 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save final model and metrics
    model.save(str(results_dir / 'final_model.keras'))
    np.save(str(results_dir / 'metrics.npy'), history.history)
    
    return history

def main():
    # Load data
    train_ds, val_ds, test_ds = load_csi_data()
    
    # Get input shape from dataset
    for x, _ in train_ds.take(1):
        input_shape = x.shape[1:]
        break
    
    # Create and train improved model
    print("Training improved model...")
    improved_model = create_improved_model(input_shape)
    improved_history = train_improved_model(improved_model, train_ds, val_ds)
    
    # Create and train residual model
    print("\nTraining residual model...")
    residual_model = create_residual_model(input_shape)
    residual_history = train_improved_model(residual_model, train_ds, val_ds)
    
    # Evaluate both models
    print("\nEvaluating models...")
    improved_results = improved_model.evaluate(test_ds, return_dict=True)
    residual_results = residual_model.evaluate(test_ds, return_dict=True)
    
    print("\nImproved Model Results:")
    for metric, value in improved_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nResidual Model Results:")
    for metric, value in residual_results.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 