import tensorflow as tf
import numpy as np

def test_tensorflow():
    print("TensorFlow version:", tf.__version__)
    print("GPU Available:", tf.config.list_physical_devices('GPU'))
    print("Metal Available:", tf.config.list_physical_devices('GPU'))
    
    # Create a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    
    # Test forward pass
    x = np.random.random((1, 5))
    y = model(x)
    print("\nTest forward pass successful!")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    test_tensorflow() 