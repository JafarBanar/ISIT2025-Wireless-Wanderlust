# Task 4: Grant-Free Access Model Submission

## Model Description
The model implements a grant-free access mechanism for CSI-based localization, combining:
- Channel sensing for interference detection
- Feature selection for efficient CSI processing
- Position prediction with trajectory awareness

## Evaluation Results

### Performance Metrics
- Mean Absolute Error (MAE): 0.0468
- 90th Percentile Error (R90): 0.0868
- Combined Score: 0.0668
- Priority Classification Accuracy: 100%

### Model Architecture
- Input: CSI data with shape [batch_size, sequence_length, n_arrays, n_elements, n_freq, 2]
- Outputs:
  - Position predictions
  - Channel occupancy mask
  - Interference estimation

### Key Features
1. Channel Sensing Layer
   - Energy-based interference detection
   - Adaptive thresholding
   - Multi-antenna processing

2. Feature Selection
   - Attention-based feature weighting
   - Adaptive feature selection
   - Priority-based transmission

3. Position Prediction
   - Trajectory-aware processing
   - Multi-task learning
   - Combined loss optimization

## Files Included
- `best_model.keras`: The trained model weights
- `position_predictions.png`: Visualization of position predictions vs ground truth
- `error_distribution.png`: Distribution of prediction errors
- `training_history.png`: Training metrics over time

## Usage
The model can be loaded and used for inference using:
```python
import tensorflow as tf
model = tf.keras.models.load_model('best_model.keras')
predictions = model.predict(csi_data)
```

## Notes
- The model achieves high accuracy in both position prediction and priority classification
- The low MAE and R90 values indicate robust performance across different scenarios
- The perfect priority classification accuracy suggests effective channel sensing 