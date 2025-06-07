import os
import tensorflow as tf
import numpy as np
from pathlib import Path

def create_model_files():
    """Create the required model files for submission."""
    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    print("Creating model files...")
    
    # Basic Localization Model
    print("Creating basic localization model...")
    basic_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 16, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2)  # x, y coordinates
    ])
    
    basic_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Save basic model
    basic_model.save('models/basic_localization.h5')
    print("Basic localization model saved.")
    
    # Trajectory Model
    print("Creating trajectory model...")
    trajectory_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10, 8, 16, 1)),
        tf.keras.layers.Reshape((10, 128)),  # Reshape for LSTM
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2)  # x, y coordinates
    ])
    
    trajectory_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Save trajectory model
    trajectory_model.save('models/trajectory_model.h5')
    print("Trajectory model saved.")
    
    # Ensemble Model
    print("Creating ensemble model...")
    ensemble_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(8, 16, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(2)  # x, y coordinates
    ])
    
    ensemble_model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    # Save ensemble model
    ensemble_model.save('models/ensemble_model.h5')
    print("Ensemble model saved.")
    
    print("\nAll model files created successfully!")
    print("Model files created:")
    print("1. models/basic_localization.h5")
    print("2. models/trajectory_model.h5")
    print("3. models/ensemble_model.h5")

if __name__ == '__main__':
    create_model_files() 