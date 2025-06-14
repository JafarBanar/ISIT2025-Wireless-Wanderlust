import requests
import numpy as np
import time
import concurrent.futures
import json
from datetime import datetime
import os

def generate_test_data():
    """Generate random CSI data for testing."""
    return {
        "csi_data": np.random.randn(1, 8, 16).tolist()
    }

def make_prediction(url="http://localhost:8000/predict"):
    """Make a single prediction request to the model server."""
    try:
        start_time = time.time()
        response = requests.post(url, json=generate_test_data())
        latency = time.time() - start_time
        
        if response.status_code == 200:
            return {
                "success": True,
                "latency": latency,
                "response": response.json()
            }
        else:
            return {
                "success": False,
                "latency": latency,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "error": str(e)
        }

def run_load_test(num_requests=100, concurrent_requests=10):
    """Run a load test with specified number of total and concurrent requests."""
    print(f"Starting load test with {num_requests} total requests ({concurrent_requests} concurrent)")
    
    results = []
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(make_prediction) for _ in range(num_requests)]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            
            if i % 10 == 0:
                print(f"Completed {i}/{num_requests} requests")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    successful_requests = sum(1 for r in results if r["success"])
    latencies = [r["latency"] for r in results]
    
    stats = {
        "total_requests": num_requests,
        "concurrent_requests": concurrent_requests,
        "successful_requests": successful_requests,
        "failed_requests": num_requests - successful_requests,
        "total_time": total_time,
        "requests_per_second": num_requests / total_time,
        "avg_latency": np.mean(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "p95_latency": np.percentile(latencies, 95),
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results
    os.makedirs("logs/load_test", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"logs/load_test/load_test_{timestamp}.json", "w") as f:
        json.dump({
            "stats": stats,
            "results": results
        }, f, indent=2)
    
    print("\nLoad Test Results:")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Successful Requests: {stats['successful_requests']}")
    print(f"Failed Requests: {stats['failed_requests']}")
    print(f"Total Time: {stats['total_time']:.2f}s")
    print(f"Requests per Second: {stats['requests_per_second']:.2f}")
    print(f"Average Latency: {stats['avg_latency']*1000:.2f}ms")
    print(f"95th Percentile Latency: {stats['p95_latency']*1000:.2f}ms")
    
    return stats

if __name__ == "__main__":
    # Run a load test with 100 total requests and 10 concurrent requests
    run_load_test(num_requests=100, concurrent_requests=10) 