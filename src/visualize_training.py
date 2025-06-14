import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_training_results():
    # Load the training history
    history_path = Path('models/task4/training_history.npy')
    if not history_path.exists():
        print(f"Training history not found at {history_path}")
        return
    
    # Load history
    history = np.load(history_path, allow_pickle=True).item()
    
    # Set style
    plt.style.use('seaborn')
    
    # Create subplots for each metric group
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot position metrics
    axes[0].plot(history['positions_loss'], label='Training Loss', color='#2ecc71')
    axes[0].plot(history['val_positions_loss'], label='Validation Loss', color='#27ae60')
    axes[0].plot(history['positions_mae'], label='Training MAE', color='#e74c3c')
    axes[0].plot(history['val_positions_mae'], label='Validation MAE', color='#c0392b')
    axes[0].plot(history['positions_r90'], label='Training R90', color='#3498db')
    axes[0].plot(history['val_positions_r90'], label='Validation R90', color='#2980b9')
    axes[0].set_title('Position Prediction Metrics')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot occupancy metrics
    axes[1].plot(history['occupancy_mask_loss'], label='Training Loss', color='#2ecc71')
    axes[1].plot(history['val_occupancy_mask_loss'], label='Validation Loss', color='#27ae60')
    axes[1].plot(history['occupancy_mask_accuracy'], label='Training Accuracy', color='#e74c3c')
    axes[1].plot(history['val_occupancy_mask_accuracy'], label='Validation Accuracy', color='#c0392b')
    axes[1].set_title('Channel Occupancy Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot interference metrics
    axes[2].plot(history['interference_loss'], label='Training Loss', color='#2ecc71')
    axes[2].plot(history['val_interference_loss'], label='Validation Loss', color='#27ae60')
    axes[2].plot(history['interference_mse'], label='Training MSE', color='#e74c3c')
    axes[2].plot(history['val_interference_mse'], label='Validation MSE', color='#c0392b')
    axes[2].set_title('Interference Prediction Metrics')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/task4/training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print final metrics
    print("\nFinal Training Metrics:")
    print(f"Position MAE: {history['positions_mae'][-1]:.4f}")
    print(f"Position R90: {history['positions_r90'][-1]:.4f}")
    print(f"Occupancy Accuracy: {history['occupancy_mask_accuracy'][-1]:.4f}")
    print(f"Interference MSE: {history['interference_mse'][-1]:.4f}")
    
    print("\nFinal Validation Metrics:")
    print(f"Position MAE: {history['val_positions_mae'][-1]:.4f}")
    print(f"Position R90: {history['val_positions_r90'][-1]:.4f}")
    print(f"Occupancy Accuracy: {history['val_occupancy_mask_accuracy'][-1]:.4f}")
    print(f"Interference MSE: {history['val_interference_mse'][-1]:.4f}")

if __name__ == '__main__':
    visualize_training_results() 