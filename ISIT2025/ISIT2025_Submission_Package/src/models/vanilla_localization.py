import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.utils.metrics import CombinedMetric, combined_loss
from src.utils.competition_metrics import R90Metric
from src.utils.logging_utils import setup_logging

# Set up logging
logger = setup_logging(__name__)

layers = keras.layers
Model = keras.Model

class VanillaLocalizationModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        logger.info(f"Initializing VanillaLocalizationModel with input shape {input_shape} and {num_classes} classes")
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # CNN layers for feature extraction
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        # Flatten and Dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.3)
        
        # Output layer for coordinates (x, y)
        self.output_layer = layers.Dense(2)
        
        # Compile the model
        self.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def call(self, inputs, training=False):
        # If input is 4D: (batch, 32, 1024, 2)
        if len(inputs.shape) == 4:
            batch_size = tf.shape(inputs)[0]
            x = tf.reshape(inputs, [batch_size, 32, 32, 64])
        # If input is 3D: (32, 1024, 2)
        elif len(inputs.shape) == 3:
            x = tf.reshape(inputs, [32, 32, 64])
            x = tf.expand_dims(x, axis=0)  # Add batch dimension
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")

        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)

        # Flatten and Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        # Output layer
        x = self.output_layer(x)
        # If input was 3D, remove batch dimension
        if len(inputs.shape) == 3:
            x = tf.squeeze(x, axis=0)
        return x
    
    def train(self, train_data, train_labels, validation_data=None, epochs=100, batch_size=32):
        history = self.fit(
            train_data,
            train_labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
        )
        return history
    
    def predict(self, test_data):
        return super().predict(test_data)
    
    def evaluate(self, test_data, test_labels):
        return super().evaluate(test_data, test_labels)
    
    def save_model(self, filepath):
        self.save(filepath)
    
    @classmethod
    def load_model(cls, filepath, input_shape=(32, 1024, 2), num_classes=2):
        """Load a saved model with custom objects."""
        logger.info(f"Loading model from {filepath}")
        try:
            instance = cls(input_shape, num_classes)
            instance.model = tf.keras.models.load_model(
                filepath,
                custom_objects={
                    'VanillaLocalizationModel': VanillaLocalizationModel,
                    'CombinedMetric': CombinedMetric,
                    'combined_loss': combined_loss,
                    'R90Metric': R90Metric
                },
                compile=False
            )
            logger.info("Model loaded successfully")
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 