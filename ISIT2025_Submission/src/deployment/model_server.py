import os
import logging
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import tensorflow as tf
from src.monitoring.monitor import ModelMonitor, APIMonitor
from src.config.server_config import (
    SERVER_HOST, SERVER_PORT, API_PREFIX,
    MODEL_PATH, MODEL_COMPILE_CONFIG,
    MONITORING_CONFIG, TF_CONFIG, init_config
)

# Initialize configuration
init_config()

# Configure TensorFlow
tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
tf.config.optimizer.set_experimental_options({
    "layout_optimizer": True,
    "constant_folding": True,
    "shape_optimization": True,
    "remapping": True,
    "arithmetic_optimization": True,
    "dependency_optimization": True,
    "loop_optimization": True,
    "function_optimization": True,
    "debug_stripper": True,
    "disable_model_pruning": False,
    "scoped_allocator_optimization": True,
    "pin_to_host_optimization": True,
    "implementation_selector": True,
    "auto_mixed_precision": False
})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/uvicorn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CSI Model Server",
    description="API for CSI-based localization model",
    version="1.0.0",
    docs_url=f"{API_PREFIX}/docs",
    redoc_url=f"{API_PREFIX}/redoc"
)

# Initialize monitors
model_monitor = ModelMonitor(
    model_path=str(MODEL_PATH),
    log_dir=str(MONITORING_CONFIG["log_dir"])
)
api_monitor = APIMonitor()


class PredictionRequest(BaseModel):
    csi_data: List[List[List[float]]]

class BatchPredictionRequest(BaseModel):
    batch: List[PredictionRequest]

class PredictionResponse(BaseModel):
    prediction: List[float]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[List[float]]
    timestamps: List[str]

class ModelServer:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            # Check if model file exists
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
            # Load model with custom objects if needed
            custom_objects = {}
            self.model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects=custom_objects,
                compile=False  # Load without compilation first
            )
            
            # Compile the model with configuration
            self.model.compile(**MODEL_COMPILE_CONFIG)
            
            # Optimize model for inference
            try:
                if hasattr(self.model, '_make_predict_function'):
                    self.model._make_predict_function()
                else:
                    # For newer TensorFlow versions, use tf.function
                    @tf.function
                    def predict_function(x):
                        return self.model(x, training=False)
                    self.model.predict_function = predict_function
            except Exception as e:
                logger.warning(f"Could not optimize model for inference: {e}")
                # Continue without optimization
            
            logger.info("Model loaded and compiled successfully")
            
            # Verify model can make predictions
            test_input = np.zeros((1, *self.model.input_shape[1:]))
            _ = self.model.predict(test_input, verbose=0)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def preprocess_input(self, data: List[List[List[float]]]) -> np.ndarray:
        """Preprocess input data for model prediction."""
        try:
            # Convert to numpy array and ensure correct shape
            input_data = np.array(data)
            if len(input_data.shape) != 3:
                raise ValueError(f"Expected 3D input, got shape {input_data.shape}")
            return input_data
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid input data format")
    
    def predict(self, data: List[List[List[float]]]) -> List[float]:
        """Make a prediction using the loaded model."""
        try:
            start_time = time.time()
            input_data = self.preprocess_input(data)
            
            # Use the optimized predict function if available
            if hasattr(self.model, 'predict_function'):
                prediction = self.model.predict_function(input_data)
            else:
                prediction = self.model.predict(input_data, verbose=0)
                
            inference_time = time.time() - start_time
            
            # Update monitoring metrics
            model_monitor.update_inference_metrics(
                inference_time=inference_time,
                prediction=prediction.tolist()
            )
            
            return prediction.tolist()
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            model_monitor.log_error(str(e))
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def batch_predict(self, batch_data: List[List[List[List[float]]]]) -> List[List[float]]:
        """Make batch predictions using the loaded model."""
        try:
            start_time = time.time()
            # Stack all inputs into a single batch
            batch_input = np.vstack([self.preprocess_input(data) for data in batch_data])
            predictions = self.model.predict(batch_input, verbose=0)
            inference_time = time.time() - start_time
            
            # Update monitoring metrics
            model_monitor.update_inference_metrics(
                inference_time=inference_time,
                prediction=predictions.tolist()
            )
            
            return predictions.tolist()
        except Exception as e:
            logger.error(f"Error making batch prediction: {str(e)}")
            model_monitor.log_error(str(e))
            raise HTTPException(status_code=500, detail="Batch prediction failed")

# Initialize model server
model_server = ModelServer()

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"status": "healthy", "message": "CSI Model Server is running"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single prediction endpoint."""
    try:
        prediction = model_server.predict(request.csi_data)
        return {
            "prediction": prediction[0],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Batch prediction endpoint."""
    try:
        batch_data = [req.csi_data for req in request.batch]
        predictions = model_server.batch_predict(batch_data)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        return {
            "predictions": predictions,
            "timestamps": [current_time] * len(predictions)
        }
    except Exception as e:
        logger.error(f"Error in batch_predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        # Check if model is loaded
        if model_server.model is None:
            return {"status": "unhealthy", "message": "Model not loaded"}
        
        # Check model server status
        return {
            "status": "healthy",
            "message": "CSI Model Server is running",
            "model_loaded": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "message": str(e)}

@app.get("/metrics")
async def get_metrics():
    """Get current monitoring metrics."""
    try:
        metrics = model_monitor.get_metrics()
        api_metrics = api_monitor.get_metrics()
        return {
            "model_metrics": metrics,
            "api_metrics": api_metrics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get metrics") 