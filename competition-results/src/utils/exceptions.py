class ModelError(Exception):
    """Base class for model-related errors."""
    pass

class DataError(Exception):
    """Base class for data-related errors."""
    pass

class TrainingError(Exception):
    """Base class for training-related errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when a model file cannot be found."""
    pass

class InvalidDataError(DataError):
    """Raised when data format is invalid."""
    pass

class DataNotFoundError(DataError):
    """Raised when data file cannot be found."""
    pass

class TrainingConfigurationError(TrainingError):
    """Raised when training configuration is invalid."""
    pass

class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass

class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass

class MetricsCalculationError(Exception):
    """Raised when metrics calculation fails."""
    pass

class VisualizationError(Exception):
    """Raised when visualization fails."""
    pass 