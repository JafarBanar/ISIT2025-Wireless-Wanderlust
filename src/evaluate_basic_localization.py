import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from models.basic_localization import BasicLocalizationModel
from utils.data_loader import load_csi_data

def evaluate_model():
    # Load the saved metrics
    metrics = np.load('results/basic_localization/metrics.npy', allow_pickle=True).item()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(metrics['history']['loss'], label='Training Loss')
    plt.plot(metrics['history']['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(metrics['history']['mae'], label='Training MAE')
    plt.plot(metrics['history']['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/basic_localization/training_history.png')
    plt.close()
    
    # Load test data
    _, _, test_ds = load_csi_data(
        data_dir='data/csi_data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=32
    )
    
    # Load the best model
    model = tf.keras.models.load_model('results/basic_localization/best_model.h5')
    
    # Evaluate on test set
    test_loss, test_mae = model.evaluate(test_ds)
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    # Make predictions on a few test samples
    print("\nSample Predictions:")
    for X_batch, y_batch in test_ds.take(3):
        predictions = model.predict(X_batch)
        for true, pred in zip(y_batch[:5], predictions[:5]):
            print(f"True: {true.numpy()}, Predicted: {pred}")
        break

if __name__ == "__main__":
    evaluate_model() 