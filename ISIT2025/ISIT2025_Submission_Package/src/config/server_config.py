import os
from pathlib import Path
import tensorflow as tf
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Server Configuration
SERVER_HOST = "localhost"  # Changed from 0.0.0.0 for better security
SERVER_PORT = 8000
API_PREFIX = "/api/v1"
MONITORING_HOST = "localhost"
RELOAD_DIRS = ["src/deployment", "src/monitoring"]  # More specific reload directories

# Model Configuration
MODEL_PATH = Path("models/best_model.h5")
MODEL_COMPILE_CONFIG = {
    "optimizer": "adam",
    "loss": "mse",
    "metrics": ["mae"]
}

# TensorFlow Configuration
TF_CONFIG = {
    "TF_ENABLE_ONEDNN_OPTS": "0",  # Disable oneDNN custom operations
    "TF_CPP_MIN_LOG_LEVEL": "2",   # Reduce TensorFlow logging
    "TF_FORCE_GPU_ALLOW_GROWTH": "true",
    "TF_ENABLE_AUTO_MIXED_PRECISION": "0",
    "TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS": "true",
    "TF_USE_CUDNN": "0",
    "TF_ENABLE_GPU_GARBAGE_COLLECTION": "true",
    "TF_GPU_THREAD_MODE": "gpu_private",
    "TF_GPU_THREAD_COUNT": "1",
    "TF_USE_CUDNN_AUTOTUNE": "0",
    "TF_ENABLE_DEPRECATION_WARNINGS": "0"  # Disable deprecation warnings
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "log_dir": Path("logs/monitoring"),
    "metrics_save_interval": 100,  # Save metrics every 100 samples
    "monitoring_interval": 1.0,    # Collect metrics every second
    "plot_dpi": 300,
    "plot_size": (20, 15),
    "api_check_interval": 60,      # Check API health every 60 seconds
    "api_timeout": 5.0,           # API health check timeout in seconds
    "max_retries": 3,             # Maximum number of retries for API checks
    "retry_delay": 1.0            # Delay between retries in seconds
}

# Create necessary directories
def setup_directories():
    """Create necessary directories for the application."""
    directories = [
        MODEL_PATH.parent,
        MONITORING_CONFIG["log_dir"]
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Apply TensorFlow configurations
def configure_tensorflow():
    """Apply TensorFlow configurations."""
    # Set environment variables
    for key, value in TF_CONFIG.items():
        os.environ[key] = value
    
    # Configure TensorFlow session
    try:
        # Disable GPU if not available
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            logger.info("No GPU found, using CPU only")
        else:
            # Configure GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s), configured for memory growth")
    except Exception as e:
        logger.warning(f"Error configuring GPU: {e}")

# Initialize configuration
def init_config():
    """Initialize all configurations."""
    setup_directories()
    configure_tensorflow() 