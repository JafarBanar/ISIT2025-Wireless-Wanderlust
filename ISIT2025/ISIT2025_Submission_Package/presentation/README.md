# ISIT 2025 Competition Presentation Materials

This directory contains all presentation materials for the ISIT 2025 Competition submission on Semi-Supervised Indoor Positioning with DICHASUS Massive-MIMO CSI.

## Directory Structure

```
presentation/
├── README.md                 # This file
├── video_and_slides_outline.txt  # Detailed outline for video presentation and slides
├── poster_outline.txt        # Detailed outline for the poster
├── visualizations/           # Result visualizations and plots
│   ├── performance_metrics.py    # Script to generate performance comparison plots
│   ├── training_curves.py        # Script to generate training curves
│   ├── error_analysis.py         # Script to generate error analysis plots
│   └── model_architectures.py    # Script to generate model architecture diagrams
├── slides/                   # Presentation slides
│   └── ISIT2025_Presentation.pptx  # Main presentation file
└── poster/                   # Poster materials
    └── ISIT2025_Poster.pdf   # Final poster in PDF format
```

## Visualization Scripts

The `visualizations/` directory contains Python scripts to generate all result plots and diagrams:

1. `performance_metrics.py`:
   - Bar charts comparing MAE, R90, and Combined metrics across tasks
   - Radar plots showing performance trade-offs
   - Bandwidth reduction vs. accuracy plots

2. `training_curves.py`:
   - Loss evolution curves (training vs. validation)
   - Metric progression plots (MAE, R90, Combined)
   - Learning rate scheduling visualization

3. `error_analysis.py`:
   - Scatter plots of predicted vs. true positions
   - Error heatmaps showing spatial distribution
   - Trajectory visualization with true vs. predicted paths

4. `model_architectures.py`:
   - Block diagrams of all model architectures
   - Feature flow visualization
   - Model component interaction diagrams

## Usage

1. Generate visualizations:
```bash
cd presentation/visualizations
python performance_metrics.py
python training_curves.py
python error_analysis.py
python model_architectures.py
```

2. The generated plots will be saved in `visualizations/output/` and can be used in both the presentation slides and poster.

## Dependencies

Required Python packages:
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- graphviz (for architecture diagrams)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Contact

For questions about the presentation materials, please contact:
[Your contact information] 