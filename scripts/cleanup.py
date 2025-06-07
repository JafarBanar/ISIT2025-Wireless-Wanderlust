import os
import shutil
import zipfile
from pathlib import Path

def cleanup_project():
    """Clean up unnecessary files and verify the final submission package."""
    print("Starting cleanup process...")
    
    # Files to remove
    files_to_remove = [
        'isit2025_submission.zip',  # Old submission package
        '.DS_Store',  # macOS system file
        'ISIT2025_report.ipynb',  # Empty notebook
        'test_inference.py',  # Temporary test file
        'plot_monitoring.py',  # Temporary plotting script
        'organize_files.py',  # Temporary organization script
        'move_files.py',  # Temporary move script
    ]
    
    # Directories to remove
    dirs_to_remove = [
        'submission_package',  # Old submission directory
        'csi_localization.egg-info',  # Build artifacts
        'venv',  # Virtual environment (should be recreated by user)
        'logs',  # Log files
        'outputs',  # Temporary outputs
        'checkpoints',  # Model checkpoints
    ]
    
    # Remove files
    for file in files_to_remove:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Removed file: {file}")
            except Exception as e:
                print(f"Error removing {file}: {e}")
    
    # Remove directories
    for dir in dirs_to_remove:
        if os.path.exists(dir):
            try:
                shutil.rmtree(dir)
                print(f"Removed directory: {dir}")
            except Exception as e:
                print(f"Error removing {dir}: {e}")
    
    # Verify final submission package
    final_zip = 'ISIT2025_Final_Submission.zip'
    if os.path.exists(final_zip):
        print("\nVerifying final submission package...")
        try:
            with zipfile.ZipFile(final_zip, 'r') as zip_ref:
                # Check required directories
                required_dirs = {'models/', 'code/', 'documentation/', 'results/'}
                zip_contents = set(zip_ref.namelist())
                
                # Check required files
                required_files = {
                    'models/basic_localization.h5',
                    'models/trajectory_model.h5',
                    'models/ensemble_model.h5',
                    'README.md',
                    'requirements.txt',
                    'LICENSE'
                }
                
                # Verify contents
                missing_dirs = required_dirs - {f for f in zip_contents if f.endswith('/')}
                missing_files = required_files - set(zip_contents)
                
                if not missing_dirs and not missing_files:
                    print("✓ Final submission package is valid and complete!")
                else:
                    print("! Warning: Some required files/directories are missing:")
                    if missing_dirs:
                        print("  Missing directories:", missing_dirs)
                    if missing_files:
                        print("  Missing files:", missing_files)
        except Exception as e:
            print(f"Error verifying submission package: {e}")
    else:
        print(f"! Warning: Final submission package {final_zip} not found!")

if __name__ == '__main__':
    cleanup_project() 