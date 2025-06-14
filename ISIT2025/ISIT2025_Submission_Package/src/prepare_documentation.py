"""Generate competition documentation."""

import os
import json
from datetime import datetime
from typing import Dict, Any
from config.competition_config import PATHS, DEADLINES

def generate_model_documentation(model_type: str,
                               metrics: Dict[str, float],
                               output_dir: str):
    """Generate documentation for a specific model."""
    doc_path = os.path.join(output_dir, f'{model_type}_documentation.md')
    
    with open(doc_path, 'w') as f:
        f.write(f"# {model_type.title()} Model Documentation\n\n")
        
        # Model Architecture
        f.write("## Model Architecture\n")
        if model_type == 'vanilla':
            f.write("""
The vanilla localization model uses a simple yet effective architecture:
- Input layer for CSI features (4 arrays × 8 elements × 16 frequencies × 2 components)
- Feature extraction layers with proper regularization
- Dense layers for position prediction
- Optimized for MAE and R90 metrics
""")
        elif model_type == 'trajectory':
            f.write("""
The trajectory-aware model incorporates temporal information:
- LSTM layers for sequence processing
- Attention mechanism for temporal dependencies
- Feature extraction from CSI data
- Position prediction with trajectory awareness
""")
        else:  # feature selection
            f.write("""
The grant-free random access model with feature selection:
- Channel sensing mechanism
- Feature selection layer
- Priority-based transmission
- Central model for position estimation
""")
        
        # Performance Metrics
        f.write("\n## Performance Metrics\n")
        f.write(f"- MAE: {metrics['mae']:.4f}\n")
        f.write(f"- R90: {metrics['r90']:.4f}\n")
        f.write(f"- Combined Score: {metrics['combined_metric']:.4f}\n")
        
        # Implementation Details
        f.write("\n## Implementation Details\n")
        if model_type == 'vanilla':
            f.write("""
- Implemented in TensorFlow 2.x
- Uses custom metrics for competition evaluation
- Optimized with Adam optimizer
- Includes L2 regularization and dropout
""")
        elif model_type == 'trajectory':
            f.write("""
- Sequence-based processing
- Custom attention mechanism
- Trajectory smoothness optimization
- Time-weighted metrics
""")
        else:
            f.write("""
- Channel state estimation
- Adaptive feature selection
- Priority-based transmission protocol
- Interference management
""")
        
        # Training Process
        f.write("\n## Training Process\n")
        f.write("""
- Early stopping with patience
- Learning rate reduction on plateau
- Batch size optimization
- Validation split for monitoring
""")
        
        # Competition Compliance
        f.write("\n## Competition Compliance\n")
        f.write("""
The model adheres to all competition requirements:
- 4 remote antenna arrays
- 8 elements per array
- 16 frequency bands
- Combined metric optimization (0.7×MAE + 0.3×R90)
""")

def generate_team_documentation(output_dir: str):
    """Generate team documentation."""
    doc_path = os.path.join(output_dir, 'team_documentation.md')
    
    with open(doc_path, 'w') as f:
        f.write("# Team Documentation\n\n")
        
        # Team Information
        f.write("## Team Information\n")
        f.write("- Team Name: [Your Team Name]\n")
        f.write("- Institution: [Your Institution]\n")
        f.write("- Team Members:\n")
        f.write("  1. [Member 1 Name] - [Role]\n")
        f.write("  2. [Member 2 Name] - [Role]\n")
        f.write("  3. [Member 3 Name] - [Role]\n")
        
        # IEEE Membership
        f.write("\n## IEEE Information Theory Society Membership\n")
        f.write("- All team members are IEEE Information Theory Society members\n")
        f.write("- Membership numbers will be provided during registration\n")
        
        # Important Dates
        f.write("\n## Important Dates\n")
        f.write(f"- Registration Deadline: {DEADLINES['registration']}\n")
        f.write(f"- Final Submission: {DEADLINES['submission']}\n")

def generate_competition_documentation():
    """Generate all competition documentation."""
    # Create documentation directory
    os.makedirs(PATHS['documentation_dir'], exist_ok=True)
    
    # Load validation results
    results_path = os.path.join(PATHS['output_dir'], 'validation_results.json')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            results = json.load(f)
    else:
        print("Warning: No validation results found. Run validation first.")
        return
    
    # Generate documentation for each model
    for model_type in ['vanilla', 'trajectory', 'feature_selection']:
        if model_type in results:
            generate_model_documentation(
                model_type,
                results[model_type],
                PATHS['documentation_dir']
            )
    
    # Generate team documentation
    generate_team_documentation(PATHS['documentation_dir'])
    
    # Generate main README
    readme_path = os.path.join(PATHS['documentation_dir'], 'README.md')
    with open(readme_path, 'w') as f:
        f.write("# ISIT 2025 Wireless Wanderlust Competition\n\n")
        
        f.write("## Overview\n")
        f.write("This repository contains our submission for the ISIT 2025 Wireless Wanderlust Competition.\n\n")
        
        f.write("## Models\n")
        f.write("We have implemented three models:\n")
        f.write("1. Vanilla Localization Model\n")
        f.write("2. Trajectory-Aware Model\n")
        f.write("3. Grant-Free Random Access Model\n\n")
        
        f.write("## Performance Summary\n")
        f.write("| Model | MAE | R90 | Combined Score |\n")
        f.write("|-------|-----|-----|----------------|\n")
        for model_type in ['vanilla', 'trajectory', 'feature_selection']:
            if model_type in results:
                metrics = results[model_type]
                f.write(f"| {model_type.title()} | {metrics['mae']:.4f} | {metrics['r90']:.4f} | {metrics['combined_metric']:.4f} |\n")
        
        f.write("\n## Documentation\n")
        f.write("- [Vanilla Model Documentation](vanilla_documentation.md)\n")
        f.write("- [Trajectory Model Documentation](trajectory_documentation.md)\n")
        f.write("- [Feature Selection Model Documentation](feature_selection_documentation.md)\n")
        f.write("- [Team Documentation](team_documentation.md)\n")
        
        f.write("\n## Generated\n")
        f.write(f"Documentation generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    generate_competition_documentation() 