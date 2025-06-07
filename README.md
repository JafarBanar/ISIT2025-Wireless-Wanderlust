# ISIT2025 Wireless Wanderlust - CSI-based Localization

This repository contains the implementation for the ISIT2025 Wireless Wanderlust competition, focusing on CSI-based localization using massive MIMO systems.

## Features

- **Basic Localization Model**
  - Efficient CSI feature extraction
  - Optimized for MAE and R90 metrics
  - Test Loss: 0.4949
  - Test MAE: 0.3370

- **Trajectory-Aware Model**
  - LSTM-based sequence processing
  - Attention mechanism for temporal features
  - Improved prediction accuracy

- **Feature Selection**
  - Adaptive feature selection
  - Priority-based transmission
  - Collision handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ISIT2025.git
cd ISIT2025
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

1. Basic Localization Model:
```bash
python src/train_basic_model.py
```

2. Trajectory-Aware Model:
```bash
python src/train_trajectory_model.py
```

### Model Optimization

Optimize models for competition:
```bash
python src/optimization/optimize_models.py
```

### Generating Submissions

Create competition submission:
```bash
python src/submission/create_submission.py
```

### Running Tests

Run test suite:
```bash
python -m unittest discover tests
```

## Project Structure

```
ISIT2025/
├── src/
│   ├── models/
│   │   ├── basic_localization.py
│   │   └── trajectory_model.py
│   ├── optimization/
│   │   └── optimize_models.py
│   ├── submission/
│   │   └── create_submission.py
│   └── utils/
│       ├── competition_metrics.py
│       └── model_utils.py
├── tests/
│   └── test_models.py
├── models/
│   ├── best_model.h5
│   └── trajectory_model.h5
├── data/
│   └── test/
├── requirements.txt
└── README.md
```

## Model Architecture

### Basic Localization Model
- Input: CSI data (8 antennas × 16 frequency bands)
- Architecture: CNN + Dense layers
- Output: 2D coordinates (x, y)

### Trajectory-Aware Model
- Input: CSI sequence data
- Architecture: LSTM + Attention + Dense layers
- Output: 2D coordinates (x, y)

## Competition Metrics

- MAE (Mean Absolute Error)
- R90 (90th percentile error)
- Combined Score: 0.7 × MAE + 0.3 × R90

## Performance

### Basic Model
- Test Loss: 0.4949
- Test MAE: 0.3370
- Training Time: ~2 hours
- Inference Time: < 10ms

### Trajectory Model
- Improved accuracy for moving targets
- Better handling of temporal dependencies
- Slightly higher computational cost

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ISIT2025 Wireless Wanderlust Competition
- DICHASUS Massive MIMO CSI Dataset
- IEEE Information Theory Society 