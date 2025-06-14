import os
import tensorflow as tf
from models.basic_localization import BasicLocalizationModel

def export_model():
    # Create export directory
    export_dir = 'results/basic_localization/export/saved_model'
    os.makedirs(export_dir, exist_ok=True)
    
    # Load the trained model
    model = tf.keras.models.load_model('results/basic_localization/best_model.keras')
    
    # Call model once to set input signature
    dummy_input = tf.zeros((1, 32, 1024, 2))  # Batch size 1, same shape as training data
    _ = model(dummy_input)
    
    # Export the model to SavedModel format
    model.export(export_dir)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = 'results/basic_localization/model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Model exported to {export_dir}")
    print(f"TFLite model saved to {tflite_path}")

if __name__ == "__main__":
    export_model() 