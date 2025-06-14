import tensorflow as tf
import os

def create_dummy_model():
    """Create a simple dummy model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(8, 16)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2, activation='linear')  # Output 2D coordinates
    ])
    # Do NOT compile the model for inference-only use
    return model

def main():
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Create and save dummy model
    model = create_dummy_model()
    model.save('models/best_model.h5')
    print("Dummy model saved to models/best_model.h5")

if __name__ == "__main__":
    main() 