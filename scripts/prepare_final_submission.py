import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def create_final_submission():
    """Create the final competition submission package."""
    # Create submission directory
    submission_dir = Path('ISIT2025_Final_Submission')
    if submission_dir.exists():
        shutil.rmtree(submission_dir)
    submission_dir.mkdir()
    
    # Create subdirectories
    (submission_dir / 'code').mkdir()
    (submission_dir / 'models').mkdir()
    (submission_dir / 'documentation').mkdir()
    (submission_dir / 'results').mkdir()
    
    # Copy code files
    print("Copying code files...")
    code_dirs = [
        'src/models',
        'src/utils',
        'src/data_loader',
        'src/optimization',
        'src/submission',
        'src/task1',
        'src/task2',
        'src/task3',
        'src/task4'
    ]
    for code_dir in code_dirs:
        if os.path.exists(code_dir):
            shutil.copytree(code_dir, submission_dir / 'code' / Path(code_dir).name)
    
    # Copy model files
    print("Copying model files...")
    model_files = [
        'models/vanilla_model.h5',
        'models/trajectory_model.h5',
        'models/feature_selection_model.h5',
        'models/ensemble_model.h5'
    ]
    for model_file in model_files:
        if os.path.exists(model_file):
            shutil.copy2(model_file, submission_dir / 'models')
    
    # Copy documentation
    print("Copying documentation...")
    doc_files = [
        'docs/model_analysis.md',
        'docs/architecture.md',
        'docs/methodology.md',
        'README.md',
        'LICENSE'
    ]
    for doc_file in doc_files:
        if os.path.exists(doc_file):
            shutil.copy2(doc_file, submission_dir / 'documentation')
    
    # Copy results
    print("Copying results...")
    results_dirs = [
        'results/visualizations',
        'results/metrics',
        'results/predictions'
    ]
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            shutil.copytree(results_dir, submission_dir / 'results' / Path(results_dir).name)
    
    # Copy environment files
    print("Copying environment files...")
    env_files = [
        'requirements.txt',
        'Dockerfile',
        '.dockerignore'
    ]
    for env_file in env_files:
        if os.path.exists(env_file):
            shutil.copy2(env_file, submission_dir)
    
    # Create a presentation folder (for video, slides, and optional poster) and copy presentation materials (if they exist) into it.
    presentation_dir = submission_dir / "presentation"
    presentation_dir.mkdir(exist_ok=True)
    presentation_src = Path("presentation")
    if presentation_src.exists():
        for item in presentation_src.glob("*"):
            if item.is_file():
                shutil.copy(item, presentation_dir)
    else:
        print("Presentation folder (presentation/) not found. Skipping presentation materials.")
    
    # Create submission metadata
    metadata = {
        'submission_date': datetime.now().isoformat(),
        'competition_info': {
            'name': 'ISIT2025 Wireless Wanderlust',
            'tasks': ['Vanilla Localization', 'Trajectory-Aware Localization', 
                     'Grant-Free RA', 'Feature-Selection + Grant-Free RA'],
            'deadline': '2025-06-13'
        },
        'environment': {
            'os': 'Ubuntu 22.04',
            'cuda': '12.0.1',
            'python': '3.11',
            'tensorflow': '2.15.0',
            'torch': '2.2.0'
        },
        'model_performance': {
            'task1': {
                'mae': None,  # To be filled with actual values
                'r90': None,
                'combined_metric': None
            },
            'task2': {
                'mae': None,
                'r90': None,
                'combined_metric': None
            },
            'task3': {
                'mae': None,
                'r90': None,
                'combined_metric': None
            },
            'task4': {
                'mae': None,
                'r90': None,
                'combined_metric': None
            }
        }
    }
    
    with open(submission_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create zip file
    print("Creating submission package...")
    shutil.make_archive('ISIT2025_Final_Submission', 'zip', submission_dir)
    
    # Clean up
    shutil.rmtree(submission_dir)
    
    print("\nFinal submission package created: ISIT2025_Final_Submission.zip")
    print("\nSubmission Contents:")
    print("1. Code:")
    print("   - Model implementations")
    print("   - Data loaders")
    print("   - Utility functions")
    print("   - Submission generation code")
    print("\n2. Models:")
    print("   - Vanilla localization model")
    print("   - Trajectory-aware model")
    print("   - Feature selection model")
    print("   - Ensemble model")
    print("\n3. Documentation:")
    print("   - Model analysis")
    print("   - Architecture documentation")
    print("   - Methodology explanation")
    print("   - README and LICENSE")
    print("\n4. Results:")
    print("   - Performance visualizations")
    print("   - Evaluation metrics")
    print("   - Test set predictions")
    print("\n5. Environment:")
    print("   - requirements.txt")
    print("   - Dockerfile (Ubuntu 22.04 + CUDA 12)")
    print("\nPlease verify the contents before submitting to the competition.")
    print("\nNext steps:")
    print("1. Prepare 5-minute video presentation")
    print("2. Create presentation slides")
    print("3. Optional: Create one-page poster/diagram")
    print("4. Submit package by June 13, 2025 23:59 AoE")

if __name__ == "__main__":
    create_final_submission() 