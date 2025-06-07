import os
import argparse
from datetime import datetime
from train_enhanced_model import main as train_model
from prepare_submission import main as prepare_submission
from utils.data_loader import CompetitionDataLoader

def setup_directories():
    """Create necessary directories"""
    dirs = ['data', 'models', 'logs', 'results', 'checkpoints', 'submission']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join('logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def main():
    parser = argparse.ArgumentParser(description='Run complete ISIT 2025 competition pipeline')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing competition dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only prepare submission')
    
    args = parser.parse_args()
    
    # Setup
    print("Setting up directories...")
    setup_directories()
    log_dir = setup_logging()
    
    if not args.skip_training:
        print("\n=== Starting Training Pipeline ===")
        
        # Prepare data
        print("Preparing data...")
        data_loader = CompetitionDataLoader(
            data_dir=args.data_dir,
            batch_size=args.batch_size
        )
        train_dataset, val_dataset, test_dataset = data_loader.prepare_data()
        
        # Get dataset info
        dataset_info = data_loader.get_dataset_info(train_dataset)
        print(f"Dataset loaded: {dataset_info['total_samples']} samples")
        
        # Train model
        print("\nStarting model training...")
        train_model()
        
        print("\nTraining completed!")
    
    # Prepare submission
    print("\n=== Preparing Submission ===")
    prepare_submission()
    
    print("\nComplete pipeline executed successfully!")
    print(f"Logs available at: {log_dir}")

if __name__ == "__main__":
    main() 