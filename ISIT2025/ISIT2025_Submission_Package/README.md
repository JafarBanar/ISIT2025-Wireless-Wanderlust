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

## Reproducing Results

### Prerequisites
– Ensure you have Python (3.9 or later) and pip installed.
– (Optional) Use a virtual environment (e.g., conda or venv) to isolate dependencies.

### Installation
– Clone this repository (or unzip the submission package) into your workspace.
– Install dependencies (for example, using pip):
  (bash)
  cd ISIT2025_Submission_Package
  pip install –r requirements.txt
 (Alternatively, if you use conda, run: conda env create –f environment.yml.)

### Running Training (Reproducing Model Weights)
– (For Task 1) Run the basic localization training script (for example, from the src directory):
  (bash)
  cd src
 PYTHONPATH=. python train_basic_localization.py ––data_dir=”../data” ––output_dir=”../models/task1”
 (Adjust ––data_dir and ––output_dir as needed.)

– (For Task 3) Run the Task 3 training script (for example, from the src directory):
 (bash)
 cd src
 PYTHONPATH=. python task3/train.py ––data_dir=”../data” ––output_dir=”../models/task3”
 (Adjust ––data_dir and ––output_dir as needed.)

– (For Task 4) (If applicable) Run the Task 4 training script (for example, from the src directory):
 (bash)
 cd src
 PYTHONPATH=. python train_feature_selection.py ––data_dir=”../data” ––output_dir=”../models/task4”
 (Adjust ––data_dir and ––output_dir as needed.)

### Evaluating (Reproducing Evaluation Plots and Metrics)
– (For Task 1) (If applicable) Run the evaluation script (for example, from the src directory):
 (bash)
 cd src
 PYTHONPATH=. python evaluate_basic_localization.py ––model=”../models/task1/best_model.h5” ––data_dir=”../data”
 (Adjust ––model and ––data_dir as needed.)

– (For Task 3) Run the Task 3 evaluation script (for example, from the src directory):
 (bash)
 cd src
 PYTHONPATH=. python task3/evaluate.py ––data_dir=”../data” ––checkpoint_dir=”../models/task3”
 (Adjust ––data_dir and ––checkpoint_dir as needed.)

– (For Task 4) (If applicable) Run the Task 4 evaluation script (for example, from the src directory):
 (bash)
 cd src
 PYTHONPATH=. python train_feature_selection.py ––data_dir=”../data” ––output_dir=”../models/task4” ––evaluate
 (Adjust ––data_dir and ––output_dir as needed.)

 (Note: Adjust ––evaluate or ––model arguments as per your evaluation script.)

### (Optional) Generating Final Plots (Slides/Poster)
– (If you have a script to generate final plots (e.g., for slides or poster), run it (for example, from the presentation/visualizations directory):
 (bash)
 cd presentation/visualizations
 PYTHONPATH=”../..” python generate_all_plots.py ––output=”output”
 (Adjust ––output as needed.)

 (Note: Ensure that your final slide deck (e.g., PDF) and poster (e.g., PDF) are also included in the submission package (for example, in presentation/).)

 (End of Reproducing Results section.) 