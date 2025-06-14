import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(name=None, log_dir='logs'):
    """
    Set up logging configuration.
    
    Args:
        name (str, optional): Logger name. If None, uses the calling module name.
        log_dir (str): Directory to store log files.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    if name is None:
        name = __name__
        
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{timestamp}_{name}.log"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def log_model_summary(logger, model):
    """Log model architecture summary."""
    logger.info("Model Architecture Summary:")
    model.summary(print_fn=logger.info)
    
def log_training_progress(logger, epoch, logs):
    """Log training progress."""
    msg = f"Epoch {epoch + 1} - "
    msg += " - ".join(f"{k}: {v:.4f}" for k, v in logs.items())
    logger.info(msg)
    
def log_evaluation_results(logger, metrics):
    """Log model evaluation results."""
    logger.info("Evaluation Results:")
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name}: {value:.4f}")
        
def log_error(logger, error, context=None):
    """Log error with context."""
    if context:
        logger.error(f"{context}: {str(error)}")
    else:
        logger.error(str(error))
        
def log_warning(logger, message):
    """Log warning message."""
    logger.warning(message)
    
def log_info(logger, message):
    """Log info message."""
    logger.info(message) 