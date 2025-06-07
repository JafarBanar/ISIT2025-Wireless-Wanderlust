import os
import zipfile
import json
from pathlib import Path

def verify_submission():
    """Verify the contents of the submission package."""
    submission_file = 'ISIT2025_Final_Submission.zip'
    
    if not os.path.exists(submission_file):
        print(f"Error: {submission_file} not found!")
        return False
    
    print(f"Verifying {submission_file}...")
    
    # Check zip file
    try:
        with zipfile.ZipFile(submission_file, 'r') as zip_ref:
            # Get list of files
            file_list = zip_ref.namelist()
            
            # Check required directories
            required_dirs = ['models/', 'code/', 'documentation/', 'results/']
            for dir_name in required_dirs:
                if not any(f.startswith(dir_name) for f in file_list):
                    print(f"Error: Missing required directory {dir_name}")
                    return False
            
            # Check required files
            required_files = [
                'models/basic_localization.h5',
                'models/trajectory_model.h5',
                'models/ensemble_model.h5',
                'README.md',
                'requirements.txt',
                'metadata.json',
                'LICENSE'
            ]
            
            for file_name in required_files:
                if file_name not in file_list:
                    print(f"Error: Missing required file {file_name}")
                    return False
            
            # Check metadata
            try:
                with zip_ref.open('metadata.json') as f:
                    metadata = json.load(f)
                    required_metadata = ['submission_date', 'author', 'model_performance', 'competition_info']
                    for key in required_metadata:
                        if key not in metadata:
                            print(f"Error: Missing required metadata field {key}")
                            return False
            except Exception as e:
                print(f"Error reading metadata: {e}")
                return False
            
            print("\nVerification Results:")
            print("✓ All required directories present")
            print("✓ All required files present")
            print("✓ Metadata file valid")
            print(f"✓ Total files in package: {len(file_list)}")
            
            # Print file structure
            print("\nFile Structure:")
            for file_name in sorted(file_list):
                print(f"  {file_name}")
            
            return True
            
    except Exception as e:
        print(f"Error verifying submission: {e}")
        return False

if __name__ == '__main__':
    if verify_submission():
        print("\nVerification successful! The submission package is ready.")
    else:
        print("\nVerification failed! Please check the errors above.") 