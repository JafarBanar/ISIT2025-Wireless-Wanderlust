import tensorflow as tf
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

class OptimizedLocalizationModel:
    def __init__(self, input_shape=(32, 1024, 2)):
        """
        Initialize the optimized localization model with best hyperparameters.
        
        Args:
            input_shape (tuple): Shape of input data (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = self._build_model()
        logger.info("Optimized localization model initialized with best hyperparameters")
        
    def _build_model(self):
        """Build the model architecture with optimized hyperparameters."""
        # Best hyperparameters from tuning
        l2_reg = 0.00043799
        dropout_rate = 0.2
        learning_rate = 0.0005
        
        model = tf.keras.models.Sequential([
            # First convolutional block
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                         input_shape=self.input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Second convolutional block
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Third convolutional block
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Flatten and dense layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(128, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            # Output layer
            tf.keras.layers.Dense(2)  # x, y coordinates
        ])
        
        # Compile model with optimized learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Model built with optimized parameters: l2_reg={l2_reg}, "
                   f"dropout_rate={dropout_rate}, learning_rate={learning_rate}")
        return model
    
    def train(self, train_dataset, val_dataset, epochs=100, batch_size=32):
        """
        Train the model with optimized parameters.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            History object containing training metrics
        """
        logger.info(f"Starting model training with {epochs} epochs")
        
        # Add callbacks for better training
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        
        logger.info("Model training completed")
        return history
    
    def evaluate(self, test_dataset):
        """
        Evaluate the model on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on test data")
        metrics = self.model.evaluate(test_dataset, return_dict=True)
        logger.info(f"Test metrics: {metrics}")
        return metrics
    
    def predict(self, data):
        """
        Make predictions using the model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Model predictions
        """
        return self.model.predict(data)
    
    def save(self, filepath):
        """
        Save the model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a saved model from disk.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model = cls()
        model.model = tf.keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model 