import requests
import numpy as np
import time
import json
from datetime import datetime

def generate_test_data(batch_size=10):
    """Generate a batch of random CSI data for testing."""
    return {
        "batch": [
            {
                "csi_data": np.random.randn(1, 8, 16).tolist()
            }
            for _ in range(batch_size)
        ]
    }

def test_batch_prediction(batch_size=10):
    """Test the batch prediction endpoint."""
    url = "http://localhost:8000/batch_predict"
    
    # Generate test data
    test_data = generate_test_data(batch_size)
    
    # Make request
    start_time = time.time()
    response = requests.post(url, json=test_data)
    total_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Prediction Results:")
        print(f"Batch Size: {batch_size}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average Time per Request: {(total_time/batch_size):.2f}s")
        print(f"Number of Predictions: {len(result['predictions'])}")
        print("\nFirst Prediction:", result['predictions'][0])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    # Test with different batch sizes
    for batch_size in [1, 5, 10, 20]:
        print(f"\nTesting batch size: {batch_size}")
        test_batch_prediction(batch_size)
        time.sleep(1)  # Wait between tests 