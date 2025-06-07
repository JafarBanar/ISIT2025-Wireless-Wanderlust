import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from utils.data_loader import load_csi_data
from models.basic_localization import BasicLocalizationModel


def visualize_predictions():
    # Load best model
    model = load_model(
        'results/basic_localization/best_model.keras',
        compile=False,
        custom_objects={'BasicLocalizationModel': BasicLocalizationModel}
    )

    # Load test data (small batch for visualization)
    _, _, test_ds = load_csi_data(
        data_dir='data/csi_data',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=32
    )
    # Take a single batch
    for X_test, y_true in test_ds.take(1):
        y_pred = model.predict(X_test)
        break

    # Plot true vs. predicted for first 20 samples
    plt.figure(figsize=(10, 5))
    plt.scatter(y_true[:20, 0], y_true[:20, 1], c='g', label='True', s=60)
    plt.scatter(y_pred[:20, 0], y_pred[:20, 1], c='r', marker='x', label='Predicted', s=60)
    plt.title('True vs. Predicted Localization (First 20 Samples)')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/basic_localization/prediction_scatter.png')
    print('Prediction scatter plot saved to results/basic_localization/prediction_scatter.png')

    # Print true vs. predicted for first 5 samples
    print('\nTrue vs. Predicted (first 5 samples):')
    for i in range(5):
        print(f'True: {y_true[i].numpy()}, Predicted: {y_pred[i]}')

if __name__ == "__main__":
    visualize_predictions() 