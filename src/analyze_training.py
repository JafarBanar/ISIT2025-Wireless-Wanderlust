import numpy as np
import matplotlib.pyplot as plt

def analyze_training():
    # Load metrics
    metrics = np.load('results/basic_localization/metrics.npy', allow_pickle=True).item()
    history = metrics['history']
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot MAE
    ax2.plot(history['mae'], label='Training MAE')
    ax2.plot(history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig('results/basic_localization/training_history.png')
    print("Training history plot saved to results/basic_localization/training_history.png")
    
    # Print final metrics
    print("\nFinal Metrics:")
    print(f"Test Loss: {metrics['test_loss']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    
    # Print training history summary
    print("\nTraining History Summary:")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Best validation MAE: {min(history['val_mae']):.4f}")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final training MAE: {history['mae'][-1]:.4f}")

if __name__ == "__main__":
    analyze_training() 