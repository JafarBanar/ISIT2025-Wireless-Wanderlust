import requests
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

def generate_single_test_data():
    """Generate a single random CSI sample (1, 8, 16)."""
    arr = np.random.randn(8, 16)
    arr = np.where(np.isfinite(arr), arr, 0.0)  # Replace NaN/Inf with 0.0
    return {
        "csi_data": [arr.astype(float).tolist()]
    }

def generate_batch_test_data(batch_size=10):
    """Generate a batch of random CSI samples for batch prediction endpoint."""
    batch = []
    for _ in range(batch_size):
        arr = np.random.randn(8, 16)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        batch.append({"csi_data": [arr.astype(float).tolist()]})
    return {"batch": batch}

def test_single_predictions(num_requests=10):
    """Test multiple single predictions."""
    url = "http://localhost:8000/predict"
    times = []
    
    for i in range(num_requests):
        data = generate_single_test_data()
        print(f"Request {i+1} body: {data}")  # Debug: print outgoing request
        start_time = time.time()
        response = requests.post(url, json=data)
        end_time = time.time()
        
        if response.status_code == 200:
            times.append(end_time - start_time)
        else:
            print(f"Request {i+1} failed with status code {response.status_code}")
    
    return times

def test_batch_prediction(batch_size=10):
    """Test a single batch prediction."""
    url = "http://localhost:8000/batch_predict"
    data = generate_batch_test_data(batch_size=batch_size)
    print(f"Batch request body (batch_size={batch_size}): {data}")  # Debug
    start_time = time.time()
    response = requests.post(url, json=data)
    end_time = time.time()
    
    if response.status_code == 200:
        return end_time - start_time
    else:
        print(f"Batch prediction failed with status code {response.status_code}")
        return None

def plot_results(batch_sizes, single_times, batch_times):
    """Plot the performance comparison results."""
    plt.figure(figsize=(10, 6))
    
    # Plot average single prediction time
    plt.plot(batch_sizes, [np.mean(single_times)] * len(batch_sizes), 
             'b--', label='Average Single Prediction Time')
    
    # Plot batch prediction times
    plt.plot(batch_sizes, batch_times, 'r-', label='Batch Prediction Time')
    
    # Plot theoretical sequential time
    theoretical_times = [np.mean(single_times) * size for size in batch_sizes]
    plt.plot(batch_sizes, theoretical_times, 'g--', 
             label='Theoretical Sequential Time')
    
    plt.xlabel('Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: Single vs Batch Predictions')
    plt.legend()
    plt.grid(True)
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs/performance', exist_ok=True)
    
    # Save plot
    plot_path = 'logs/performance/performance_comparison.png'
    plt.savefig(plot_path)
    plt.close()
    return plot_path

def main():
    # Test parameters
    batch_sizes = [1, 5, 10, 20, 50]
    num_single_tests = 10
    
    # Create results directory if it doesn't exist
    os.makedirs('logs/performance', exist_ok=True)
    
    # Test single predictions
    print("Testing single predictions...")
    single_times = test_single_predictions(num_single_tests)
    
    # Test batch predictions
    print("Testing batch predictions...")
    batch_times = []
    for size in batch_sizes:
        print(f"Testing batch size {size}...")
        batch_time = test_batch_prediction(size)
        if batch_time is not None:
            batch_times.append(batch_time)
        else:
            print(f"Failed to test batch size {size}")
            break
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "single_prediction_times": single_times,
        "batch_sizes": batch_sizes[:len(batch_times)],
        "batch_times": batch_times,
        "average_single_time": np.mean(single_times),
        "speedup_ratios": [np.mean(single_times) * size / batch_time 
                          for size, batch_time in zip(batch_sizes[:len(batch_times)], batch_times)]
    }
    
    # Save results to file
    results_path = 'logs/performance/performance_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate and save plot
    plot_path = plot_results(batch_sizes[:len(batch_times)], single_times, batch_times)
    
    # Print summary
    summary = f"""
Performance Test Results:
------------------------
Average Single Prediction Time: {np.mean(single_times):.3f} seconds
Batch Sizes Tested: {batch_sizes[:len(batch_times)]}
Batch Times: {[f'{t:.3f}s' for t in batch_times]}
Speedup Ratios: {[f'{r:.2f}x' for r in results['speedup_ratios']]}

Results saved to: {results_path}
Plot saved to: {plot_path}
"""
    print(summary)
    
    # Save summary to file
    with open('logs/performance/performance_summary.txt', 'w') as f:
        f.write(summary)

if __name__ == "__main__":
    main() 