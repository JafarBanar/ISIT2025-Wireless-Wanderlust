#!/usr/bin/env python3
"""
Script to generate all visualization plots for the ISIT 2025 Competition presentation.
This script runs all visualization scripts in the correct order to generate all plots.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle any errors."""
    print(f"\nRunning {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True,
                              text=True)
        if result.returncode == 0:
            print(f"Successfully generated plots from {script_name}")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"Error running {script_name}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Failed to run {script_name}: {str(e)}")
        return False
    return True

def main():
    """Run all visualization scripts in sequence."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    
    # List of scripts to run in order
    scripts = [
        'performance_metrics.py',
        'training_curves.py',
        'error_analysis.py',
        'model_architectures.py'
    ]
    
    # Run each script
    success = True
    for script in scripts:
        script_path = script_dir / script
        if not script_path.exists():
            print(f"Error: Script {script} not found!")
            success = False
            continue
        
        if not run_script(str(script_path)):
            success = False
    
    # Print summary
    if success:
        print("\nAll plots generated successfully!")
        print(f"Plots are saved in: {script_dir / 'output'}")
    else:
        print("\nSome plots failed to generate. Please check the errors above.")
        sys.exit(1)

if __name__ == '__main__':
    main() 