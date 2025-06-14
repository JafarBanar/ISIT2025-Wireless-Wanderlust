import psutil
import time
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List
import threading
import queue
import requests
from src.config.server_config import (
    MONITORING_CONFIG, SERVER_PORT, MONITORING_HOST
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, model_path: str, log_dir: str = None):
        """
        Initialize the model monitor.
        
        Args:
            model_path (str): Path to the model to monitor
            log_dir (str): Directory to save monitoring logs
        """
        self.model_path = model_path
        self.log_dir = Path(log_dir or MONITORING_CONFIG["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics with more detailed tracking
        self.metrics = {
            'inference_times': [],
            'memory_usage': [],
            'cpu_usage': [],
            'gpu_usage': [],
            'errors': [],
            'predictions': [],
            'batch_sizes': [],
            'throughput': [],
            'model_size': self._get_model_size(),
            'startup_time': datetime.now().isoformat()
        }
        
        # Initialize queues for thread-safe operations
        self.metrics_queue = queue.Queue()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
    def _get_model_size(self) -> int:
        """Get model file size in bytes."""
        try:
            return Path(self.model_path).stat().st_size
        except:
            return 0
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Collect system metrics
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                gpu_percent = self._get_gpu_usage()
                
                # Update metrics
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                })
                
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'percent': cpu_percent
                })
                
                self.metrics['gpu_usage'].append({
                    'timestamp': datetime.now().isoformat(),
                    'percent': gpu_percent
                })
                
                # Process queued metrics
                while not self.metrics_queue.empty():
                    try:
                        metric = self.metrics_queue.get_nowait()
                        for key, value in metric.items():
                            if key in self.metrics:
                                self.metrics[key].append(value)
                    except queue.Empty:
                        break
                    except Exception as e:
                        logger.error(f"Error processing queued metric: {e}")
                
                # Save metrics periodically
                if len(self.metrics['memory_usage']) % MONITORING_CONFIG["metrics_save_interval"] == 0:
                    self._save_metrics()
                
                time.sleep(MONITORING_CONFIG["monitoring_interval"])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                time.sleep(MONITORING_CONFIG["monitoring_interval"])  # Still sleep on error
        
    def _get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            # This is a placeholder - implement actual GPU monitoring
            # based on your system's GPU monitoring tools
            return 0.0
        except:
            return 0.0
        
    def log_inference(self, inference_time: float, prediction: np.ndarray):
        """
        Log inference metrics.
        
        Args:
            inference_time (float): Time taken for inference in seconds
            prediction (np.ndarray): Model prediction
        """
        self.metrics_queue.put({
            'timestamp': datetime.now().isoformat(),
            'inference_times': {'inference_time': inference_time},
            'predictions': prediction.tolist()
        })
        
    def _save_metrics(self):
        """Save collected metrics to file."""
        try:
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.log_dir / f"metrics_{timestamp}.json"
            
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
                
            logger.info(f"Metrics saved to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        try:
            # Calculate statistics
            inference_times = [m.get('inference_time', 0) for m in self.metrics['inference_times']]
            memory_usage = [m.get('memory_usage', 0) for m in self.metrics['memory_usage']]
            cpu_usage = [m.get('cpu_usage', 0) for m in self.metrics['cpu_usage']]
            throughput = [m.get('inferences_per_second', 0) for m in self.metrics['throughput']]
            
            return {
                'total_inferences': len(inference_times),
                'avg_inference_time': float(np.mean(inference_times)) if inference_times else 0.0,
                'max_inference_time': float(np.max(inference_times)) if inference_times else 0.0,
                'min_inference_time': float(np.min(inference_times)) if inference_times else 0.0,
                'avg_memory_usage': float(np.mean(memory_usage)) if memory_usage else 0.0,
                'max_memory_usage': float(np.max(memory_usage)) if memory_usage else 0.0,
                'avg_cpu_usage': float(np.mean(cpu_usage)) if cpu_usage else 0.0,
                'max_cpu_usage': float(np.max(cpu_usage)) if cpu_usage else 0.0,
                'avg_throughput': float(np.mean(throughput)) if throughput else 0.0,
                'model_size_mb': self.metrics['model_size'] / (1024 * 1024),
                'error_count': len(self.metrics['errors']),
                'uptime_seconds': (datetime.now() - datetime.fromisoformat(self.metrics['startup_time'])).total_seconds(),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {str(e)}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'last_updated': datetime.now().isoformat()
            }

    def log_error(self, error: str):
        """Log an error and append it to the errors list with a timestamp."""
        logger.error(error)
        self.metrics['errors'].append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })

    def update_inference_metrics(self, inference_time: float, prediction):
        """Update metrics for an inference (for compatibility with model server)."""
        self.log_inference(inference_time, np.array(prediction))

class APIMonitor:
    def __init__(self, api_url: str = None, check_interval: int = None):
        """
        Initialize API monitor.
        
        Args:
            api_url (str): URL of the API to monitor
            check_interval (int): Interval between checks in seconds
        """
        self.api_url = api_url or f"http://{MONITORING_HOST}:{SERVER_PORT}"
        self.check_interval = check_interval or MONITORING_CONFIG["api_check_interval"]
        self.metrics = {
            'response_times': [],
            'status_codes': [],
            'errors': []
        }
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop for API health."""
        while True:
            try:
                self.check_once()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                self.metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                time.sleep(self.check_interval)  # Still wait before retrying
    
    def check_once(self):
        """Perform a single API health check."""
        retries = 0
        while retries < MONITORING_CONFIG["max_retries"]:
            try:
                start_time = time.time()
                response = requests.get(
                    f"{self.api_url}/health",
                    timeout=MONITORING_CONFIG["api_timeout"]
                )
                response_time = time.time() - start_time
                
                self.metrics['response_times'].append({
                    'timestamp': datetime.now().isoformat(),
                    'response_time': response_time
                })
                
                self.metrics['status_codes'].append({
                    'timestamp': datetime.now().isoformat(),
                    'status_code': response.status_code
                })
                
                if response.status_code != 200:
                    self.metrics['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'status_code': response.status_code,
                        'response': response.text
                    })
                return  # Success, exit retry loop
                
            except requests.exceptions.RequestException as e:
                retries += 1
                if retries >= MONITORING_CONFIG["max_retries"]:
                    logger.error(f"Error monitoring API after {retries} retries: {str(e)}")
                    self.metrics['errors'].append({
                        'timestamp': datetime.now().isoformat(),
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'url': self.api_url,
                        'retries': retries
                    })
                else:
                    logger.warning(f"Retry {retries}/{MONITORING_CONFIG['max_retries']} for API check: {str(e)}")
                    time.sleep(MONITORING_CONFIG["retry_delay"])
            except Exception as e:
                logger.error(f"Unexpected error monitoring API: {str(e)}")
                self.metrics['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                return  # Don't retry on unexpected errors

    def get_metrics(self) -> Dict[str, Any]:
        """Get current API monitoring metrics."""
        try:
            # Calculate success rate
            total_checks = len(self.metrics['status_codes'])
            successful_checks = sum(1 for m in self.metrics['status_codes'] if m.get('status_code') == 200)
            success_rate = (successful_checks / total_checks * 100) if total_checks > 0 else 0
            
            # Calculate average response time
            response_times = [m.get('response_time', 0) for m in self.metrics['response_times']]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
            
            return {
                'api_url': self.api_url,
                'check_interval': self.check_interval,
                'total_checks': total_checks,
                'successful_checks': successful_checks,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'error_count': len(self.metrics['errors']),
                'last_error': self.metrics['errors'][-1] if self.metrics['errors'] else None,
                'last_status_code': self.metrics['status_codes'][-1] if self.metrics['status_codes'] else None,
                'last_response_time': self.metrics['response_times'][-1] if self.metrics['response_times'] else None,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting API metrics: {str(e)}")
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'timestamp': datetime.now().isoformat()
            }

def main():
    # Example usage
    model_path = "models/best_model.h5"
    api_url = "http://localhost:8000"
    
    # Initialize monitors
    model_monitor = ModelMonitor(model_path)
    api_monitor = APIMonitor(api_url)
    
    try:
        while True:
            # Print monitoring summary every minute
            print("\nMonitoring Summary:")
            print("Model Metrics:")
            print(json.dumps(model_monitor.get_summary(), indent=2))
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
        model_monitor._save_metrics()

if __name__ == "__main__":
    main() 
    main() 