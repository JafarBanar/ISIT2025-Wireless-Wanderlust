# ISIT 2025 Wireless Wanderlust - Technical Documentation

## Project Overview
This project implements three localization models for the ISIT 2025 Wireless Wanderlust competition:
1. Vanilla Localization
2. Trajectory-Aware Localization
3. Feature Selection with Grant-free Random Access

## System Architecture

### 1. Vanilla Localization Model
- Architecture: Enhanced CNN with multi-head attention
- Input: CSI data from 4 antenna arrays × 8 elements × 16 frequency bands
- Output: 2D position coordinates (x, y)
- Key Features:
  - Multi-array processing
  - Frequency band attention mechanism
  - L2 regularization (0.01)
  - Dropout layers (0.3, 0.2)

### 2. Trajectory-Aware Model
- LSTM-based architecture with attention
- Temporal feature integration
- Sequence handling for continuous tracking
- Performance metrics:
  - MAE: 0.3370
  - Test Loss: 0.4949
  - R90 metric implementation

### 3. Grant-free Random Access System
- Channel sensing mechanism
- Collision resolution with exponential backoff
- Adaptive channel selection
- Priority-based transmission scheduling

## Implementation Details

### Data Pipeline
```python
Input Shape: (batch_size, num_arrays, elements_per_array, num_freq_bands)
Preprocessing:
1. Normalization
2. Feature extraction
3. Temporal alignment
4. Batch processing
```

### Model Training
- Optimizer: Adam
- Learning rate: 1e-4 with cosine decay
- Batch size: 32
- Training epochs: 100
- Validation split: 20%
- Early stopping patience: 10

### Performance Optimization
1. Feature Selection
   - Importance-based selection
   - Adaptive sampling
   - Priority queuing

2. Random Access
   - Channel state monitoring
   - Collision avoidance
   - Backoff mechanism

## Deployment Guidelines

### System Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy 1.19+
- Minimum 8GB RAM
- CUDA-capable GPU recommended

### Installation Steps
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables
4. Run validation tests

### Usage Instructions
1. Data Preparation
2. Model Training
3. Evaluation
4. Deployment

## Performance Metrics

### Localization Accuracy
- MAE: 0.3370
- R90: [TO BE UPDATED]
- Combined Score: [TO BE UPDATED]

### System Efficiency
- Training time: [TO BE UPDATED]
- Inference latency: [TO BE UPDATED]
- Memory usage: [TO BE UPDATED]

## Testing Framework

### Unit Tests
- Model components
- Data pipeline
- Metrics calculation
- Channel sensing

### Integration Tests
- End-to-end workflow
- Performance validation
- Error handling
- Edge cases

## Maintenance and Updates

### Version Control
- Git repository
- Branch management
- Release tagging

### Documentation Updates
- Change log
- API documentation
- Performance reports
- Bug tracking

## Competition Submission Guidelines

### Required Materials
1. Source code
2. Documentation
3. Test results
4. Performance analysis

### Submission Format
- ZIP archive
- README file
- Installation guide
- Usage examples

## Data Pipeline

### Data Format
- TFRecord files containing:
  - CSI data: (4, 8, 16, 2) shape
  - Location: (x, y) coordinates
  - Timestamps (for trajectory model)

### Preprocessing
1. Data loading from TFRecord files
2. Sequence creation for trajectory model
3. Feature normalization
4. Train/val/test splitting (70/15/15)

## Training Process

### Common Settings
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Early stopping with patience
- Learning rate scheduling
- Model checkpointing

### Model-Specific Settings
1. **Vanilla Model**:
   - Combined loss (MAE + R90)
   - Standard callbacks
   
2. **Trajectory Model**:
   - Sequence length: 5
   - Increased patience for temporal learning
   
3. **Feature Selection**:
   - Feature importance weighting
   - Priority threshold adaptation

## Evaluation Metrics

### Primary Metrics
1. Mean Absolute Error (MAE)
2. R90 (90th percentile of error)
3. Combined metric (weighted average)

### Additional Metrics
1. Training time
2. Inference latency
3. Transmission rate (Feature Selection)

## Error Handling and Logging

### Error Types
1. Model errors
2. Data errors
3. Training errors
4. Metrics errors

### Logging System
- Comprehensive logging
- Error tracking
- Performance monitoring
- Training progress

## Testing Framework

### Unit Tests
1. Model architecture tests
2. Data pipeline tests
3. Training process tests
4. Metrics calculation tests

### Integration Tests
1. End-to-end training
2. Model evaluation
3. Submission generation

## Submission Process

### File Generation
1. Position predictions (CSV)
2. Transmission rates (TXT)
3. Model documentation

### Validation
1. Format checking
2. Metric verification
3. Submission requirements

## Competition Requirements

### Team Formation
- Registration deadline: May 7th, 2025
- IEEE Information Theory Society membership
- Student member requirements

### Timeline
- Competition period: May 7th - June 13th, 2025
- Submission deadlines
- Results announcement

## Future Improvements

### Model Enhancements
1. Architecture optimization
2. Hyperparameter tuning
3. Ensemble methods

### Pipeline Optimization
1. Data augmentation
2. Feature engineering
3. Training efficiency

## Contact Information

### Competition Organizers
- General inquiries: Luca Barletta, Stefano Rini, Farhad Shirani
- Dataset/evaluation: Florian Euchner, Phillip Stephan
- Technical issues: Marian Temprana Alonso, Esteban Schafir 