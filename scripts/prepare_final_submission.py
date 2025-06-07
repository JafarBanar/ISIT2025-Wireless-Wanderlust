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
    (submission_dir / 'models').mkdir()
    (submission_dir / 'code').mkdir()
    (submission_dir / 'documentation').mkdir()
    (submission_dir / 'results').mkdir()
    
    # Copy model files
    print("Copying model files...")
    model_files = [
        'models/basic_localization.h5',
        'models/trajectory_model.h5',
        'models/ensemble_model.h5'
    ]
    for model_file in model_files:
        if os.path.exists(model_file):
            shutil.copy2(model_file, submission_dir / 'models')
    
    # Copy code
    print("Copying code files...")
    code_dirs = ['src/models', 'src/utils', 'src/optimization', 'src/submission']
    for code_dir in code_dirs:
        if os.path.exists(code_dir):
            shutil.copytree(code_dir, submission_dir / 'code' / Path(code_dir).name)
    
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
    # Copy LICENSE to root as well
    if os.path.exists('LICENSE'):
        shutil.copy2('LICENSE', submission_dir / 'LICENSE')
    
    # Copy results
    print("Copying results...")
    results_dirs = ['results/visualizations', 'results/metrics', 'results/predictions']
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            shutil.copytree(results_dir, submission_dir / 'results' / Path(results_dir).name)
    
    # Copy requirements and scripts
    print("Copying requirements and scripts...")
    shutil.copy2('requirements.txt', submission_dir)
    shutil.copy2('submission/README.md', submission_dir / 'README.md')
    
    # Create submission metadata
    metadata = {
        'submission_date': datetime.now().isoformat(),
        'author': 'Jafar Banar',
        'model_performance': {
            'test_loss': 0.4949,
            'test_mae': 0.3370
        },
        'competition_info': {
            'name': 'ISIT2025 Wireless Wanderlust',
            'track': 'CSI-based Localization',
            'deadline': '2025-05-07'
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
    print("1. Models:")
    print("   - basic_localization.h5")
    print("   - trajectory_model.h5")
    print("   - ensemble_model.h5")
    print("\n2. Code:")
    print("   - Model implementations")
    print("   - Utility functions")
    print("   - Optimization code")
    print("   - Submission generation code")
    print("\n3. Documentation:")
    print("   - Model analysis")
    print("   - Architecture documentation")
    print("   - Methodology explanation")
    print("   - README and LICENSE")
    print("\n4. Results:")
    print("   - Performance plots")
    print("   - Evaluation metrics")
    print("   - Test set predictions")
    print("\nPlease verify the contents before submitting to the competition.")

if __name__ == '__main__':
    create_final_submission() 