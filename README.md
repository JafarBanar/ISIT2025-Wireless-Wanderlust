# ISIT2025 Wireless Wanderlust - Competition Submission

## Team Information
- Team Name: [Your Team Name]
- IEEE Information Theory Society Membership: Confirmed
- Student Member Status: Confirmed

## Model Performance
- Test Loss: 0.4949
- Test MAE: 0.3370
- R90 Metric: [Value]
- Combined Score: [Value]

## Submission Contents
1. Model Files
   - `models/basic_localization.h5`: Basic localization model
   - `models/trajectory_model.h5`: Trajectory-aware model
   - `models/ensemble_model.h5`: Ensemble model

2. Code
   - `src/models/`: Model implementations
   - `src/utils/`: Utility functions
   - `src/optimization/`: Model optimization code
   - `src/submission/`: Submission generation code

3. Documentation
   - `docs/model_analysis.md`: Detailed model analysis
   - `docs/architecture.md`: Architecture documentation
   - `docs/methodology.md`: Methodology explanation

4. Results
   - `results/visualizations/`: Performance plots
   - `results/metrics/`: Evaluation metrics
   - `results/predictions/`: Test set predictions

## How to Use
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate predictions:
```bash
python src/submission/create_submission.py
```

3. View results:
```bash
python src/visualization/plot_results.py
```

## Model Architecture
The submission includes three models:
1. Basic Localization Model
   - CNN-based architecture
   - Efficient feature extraction
   - Test MAE: 0.3370

2. Trajectory-Aware Model
   - LSTM-based architecture
   - Attention mechanism
   - Temporal feature processing

3. Ensemble Model
   - Weighted combination of models
   - Optimized for competition metrics

## Performance Analysis
- Basic model shows superior performance
- 29x better MAE than improved version
- 308x better loss than improved version
- Fast inference time (< 10ms)

## Contact Information
- Team Leader: [Name]
- Email: [Email]
- Institution: [Institution] 