import tensorflow as tf
import numpy as np
from pathlib import Path
import json
from src.utils.data_loader import load_csi_data

def optimize_model(model_path, output_dir):
    """Optimize the model for deployment using TensorFlow optimization techniques."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the optimized model
    output_path = Path(output_dir) / 'optimized_model.tflite'
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    return output_path

def create_model_metadata(model_path, output_dir):
    """Create metadata file with model information and usage instructions."""
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Get model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    # Create metadata dictionary
    metadata = {
        'model_name': Path(model_path).stem,
        'input_shape': model.input_shape[1:],
        'output_shape': model.output_shape[1:],
        'model_summary': '\n'.join(model_summary),
        'usage_instructions': {
            'input_format': 'CSI data with shape (sequence_length, num_features)',
            'output_format': '2D coordinates (x, y)',
            'preprocessing': 'Normalize input data to range [-1, 1]',
            'postprocessing': 'Scale output coordinates to original range'
        }
    }
    
    # Save metadata
    metadata_path = Path(output_dir) / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    return metadata_path

def create_inference_script(output_dir):
    """Create a Python script for model inference."""
    script_content = '''import tensorflow as tf
import numpy as np

class LocalizationModel:
    def __init__(self, model_path):
        """Initialize the model for inference."""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
    def preprocess(self, csi_data):
        """Preprocess CSI data for inference."""
        # Normalize input data
        csi_data = (csi_data - np.mean(csi_data)) / np.std(csi_data)
        return csi_data.astype(np.float32)
    
    def predict(self, csi_data):
        """Make predictions using the model."""
        # Preprocess input
        processed_data = self.preprocess(csi_data)
        
        # Set input tensor
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            processed_data.reshape(1, *processed_data.shape)
        )
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output tensor
        output = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        return output[0]

def main():
    # Example usage
    model = LocalizationModel('optimized_model.tflite')
    
    # Example CSI data (replace with actual data)
    csi_data = np.random.randn(100, 64)  # Example shape
    
    # Make prediction
    prediction = model.predict(csi_data)
    print(f"Predicted coordinates: {prediction}")

if __name__ == "__main__":
    main()
'''
    
    # Save inference script
    script_path = Path(output_dir) / 'inference.py'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path

def create_requirements_file(output_dir):
    """Create requirements.txt file for deployment."""
    requirements = [
        'tensorflow>=2.12.0',
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'pandas>=1.3.0'
    ]
    
    # Save requirements
    requirements_path = Path(output_dir) / 'requirements.txt'
    with open(requirements_path, 'w') as f:
        f.write('\n'.join(requirements))
    
    return requirements_path

def create_readme(output_dir):
    """Create README.md file with deployment instructions."""
    readme_content = '''# CSI-based Localization Model

This repository contains a TensorFlow Lite model for CSI-based localization.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Place your optimized model file (`optimized_model.tflite`) in the same directory.

## Usage

Run the inference script:
```bash
python inference.py
```

## Model Information

- Input: CSI data with shape (sequence_length, num_features)
- Output: 2D coordinates (x, y)
- Model type: TensorFlow Lite optimized model

## Performance

The model has been optimized for deployment with the following characteristics:
- Quantized to float16 for reduced memory usage
- Optimized for inference speed
- Compatible with edge devices

## Notes

- Ensure input data is properly preprocessed before inference
- The model expects normalized CSI data
- Output coordinates need to be scaled to the original range
'''
    
    # Save README
    readme_path = Path(output_dir) / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return readme_path

def main():
    # Create deployment directory
    deployment_dir = Path("deployment")
    deployment_dir.mkdir(exist_ok=True)
    
    # Optimize model
    print("Optimizing model...")
    model_path = "results/basic_localization/best_model.keras"
    optimized_model_path = optimize_model(model_path, deployment_dir)
    print(f"Optimized model saved to: {optimized_model_path}")
    
    # Create metadata
    print("\nCreating model metadata...")
    metadata_path = create_model_metadata(model_path, deployment_dir)
    print(f"Metadata saved to: {metadata_path}")
    
    # Create inference script
    print("\nCreating inference script...")
    inference_script_path = create_inference_script(deployment_dir)
    print(f"Inference script saved to: {inference_script_path}")
    
    # Create requirements file
    print("\nCreating requirements file...")
    requirements_path = create_requirements_file(deployment_dir)
    print(f"Requirements file saved to: {requirements_path}")
    
    # Create README
    print("\nCreating README...")
    readme_path = create_readme(deployment_dir)
    print(f"README saved to: {readme_path}")
    
    print("\nDeployment package is ready in the 'deployment' directory!")

if __name__ == "__main__":
    main() 