import numpy as np
import tensorflow as tf

# Path to TFLite model
model_path = 'results/basic_localization/model.tflite'

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print('TFLite model loaded successfully.')

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input (batch size 1, shape must match training input)
dummy_input = np.zeros(input_details[0]['shape'], dtype=np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], dummy_input)

# Run inference
interpreter.invoke()

# Get output tensor
output = interpreter.get_tensor(output_details[0]['index'])
print('Inference successful.')
print('Output shape:', output.shape)
print('Output values:', output) 