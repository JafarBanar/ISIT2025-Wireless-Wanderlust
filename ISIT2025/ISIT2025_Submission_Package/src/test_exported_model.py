import tensorflow as tf
import numpy as np
from keras.layers import TFSMLayer

# Path to exported SavedModel
export_dir = 'results/basic_localization/export/saved_model'

# Use TFSMLayer for inference-only loading
layer = TFSMLayer(export_dir, call_endpoint='serve')
print('TFSMLayer loaded successfully.')

# Create a dummy input (batch size 1, shape must match training input)
dummy_input = np.zeros((1, 32, 1024, 2), dtype=np.float32)

# Run inference
output = layer(dummy_input)
print('Inference successful.')
print('Output shape:', output.shape)
print('Output values:', output.numpy()) 