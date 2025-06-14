import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

class FeatureSelectionModel:
    def __init__(self, input_shape, num_features, priority_threshold=0.5):
        self.input_shape = input_shape
        self.num_features = num_features
        self.priority_threshold = priority_threshold
        self.model = self._build_model()
        
    def _build_model(self):
        # Input for CSI data
        inputs = layers.Input(shape=self.input_shape)
        
        # Feature extraction layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Feature selection branch
        feature_importance = layers.Conv2D(self.num_features, (1, 1), activation='sigmoid')(x)
        feature_importance = layers.GlobalAveragePooling2D()(feature_importance)
        
        # Priority-based feature selection
        priority_scores = layers.Dense(self.num_features, activation='sigmoid')(feature_importance)
        selected_features = layers.Lambda(
            lambda x: tf.cast(x > self.priority_threshold, tf.float32)
        )(priority_scores)
        
        # Feature fusion
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layers
        location_output = layers.Dense(2, name='location')(x)  # x, y coordinates
        priority_output = layers.Dense(self.num_features, activation='sigmoid', name='priority')(x)
        
        model = Model(
            inputs=inputs,
            outputs=[location_output, priority_output]
        )
        
        model.compile(
            optimizer='adam',
            loss={
                'location': 'mse',
                'priority': 'binary_crossentropy'
            },
            loss_weights={
                'location': 1.0,
                'priority': 0.5
            },
            metrics={
                'location': 'mae',
                'priority': 'accuracy'
            }
        )
        
        return model
    
    def train(self, train_data, train_labels, validation_data=None, epochs=100, batch_size=32):
        history = self.model.fit(
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
        location_pred, priority_pred = self.model.predict(test_data)
        return location_pred, priority_pred
    
    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)
    
    def get_feature_importance(self, test_data):
        """Get feature importance scores for the input data."""
        _, priority_scores = self.predict(test_data)
        return priority_scores
    
    def select_features(self, test_data, threshold=None):
        """Select features based on priority scores."""
        if threshold is None:
            threshold = self.priority_threshold
            
        priority_scores = self.get_feature_importance(test_data)
        selected_features = (priority_scores > threshold).astype(np.float32)
        return selected_features
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath):
        model = tf.keras.models.load_model(filepath)
        return model 