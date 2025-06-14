import unittest
import time
import json
from pathlib import Path
import numpy as np
from monitor import ModelMonitor, APIMonitor

class TestModelMonitor(unittest.TestCase):
    def setUp(self):
        self.model_path = "test_model.h5"
        self.log_dir = "test_logs"
        self.monitor = ModelMonitor(self.model_path, self.log_dir)
        
    def test_metrics_collection(self):
        """Test that metrics are being collected correctly."""
        # Wait for some metrics to be collected
        time.sleep(2)
        
        # Log some test inference data
        test_prediction = np.array([0.1, 0.2, 0.3])
        self.monitor.log_inference(0.5, test_prediction)
        
        # Get summary
        summary = self.monitor.get_summary()
        
        # Verify summary contains expected keys
        self.assertIn('total_inferences', summary)
        self.assertIn('avg_inference_time', summary)
        self.assertIn('avg_memory_usage', summary)
        self.assertIn('avg_cpu_usage', summary)
        
    def test_metrics_saving(self):
        """Test that metrics are being saved correctly."""
        # Log some test data
        test_prediction = np.array([0.1, 0.2, 0.3])
        self.monitor.log_inference(0.5, test_prediction)
        
        # Force save
        self.monitor.save_metrics()
        
        # Check if metrics file exists
        metrics_files = list(Path(self.log_dir).glob('metrics_*.json'))
        self.assertTrue(len(metrics_files) > 0)
        
        # Verify metrics file content
        with open(metrics_files[0], 'r') as f:
            metrics = json.load(f)
            self.assertIn('inference_times', metrics)
            self.assertIn('memory_usage', metrics)
            self.assertIn('cpu_usage', metrics)
            
    def tearDown(self):
        # Clean up test files
        import shutil
        if Path(self.log_dir).exists():
            shutil.rmtree(self.log_dir)

class TestAPIMonitor(unittest.TestCase):
    def setUp(self):
        self.api_url = "http://localhost:8000"
        self.monitor = APIMonitor(self.api_url, check_interval=1)
        
    def test_metrics_structure(self):
        """Test that API metrics have the correct structure."""
        # Force a health check
        self.monitor.check_once()
        
        # Verify metrics structure
        self.assertIn('response_times', self.monitor.metrics)
        self.assertIn('status_codes', self.monitor.metrics)
        self.assertIn('errors', self.monitor.metrics)
        
        # Verify that connection errors are being logged
        self.assertTrue(len(self.monitor.metrics['errors']) > 0)
        
    def test_error_handling(self):
        """Test that errors are being captured correctly."""
        # Force a health check
        self.monitor.check_once()
        
        # Verify errors are being logged
        self.assertIsInstance(self.monitor.metrics['errors'], list)
        self.assertTrue(len(self.monitor.metrics['errors']) > 0)
        
        # Verify error structure
        error = self.monitor.metrics['errors'][0]
        self.assertIn('timestamp', error)
        # error can be either 'error' or 'status_code' depending on the failure
        self.assertTrue('error' in error or 'status_code' in error)

def main():
    unittest.main()

if __name__ == '__main__':
    main() 