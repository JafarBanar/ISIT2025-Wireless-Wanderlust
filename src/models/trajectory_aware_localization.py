import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from src.isit2025.utils.metrics import combined_loss, CombinedMetric
from src.isit2025.models.vanilla_localization import VanillaLocalizationModel

class TrajectoryAwareLocalizationModel:
    def __init__(self, input_shape, sequence_length=5):
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.model = self._build_model()
        
    def _build_model(self):
        # Input for sequence of CSI data
        inputs = layers.Input(shape=(self.sequence_length, *self.input_shape))
        
        # CNN layers for spatial feature extraction
        x = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(inputs)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))(x)
        x = layers.TimeDistributed(layers.BatchNormalization())(x)
        x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
        
        # Flatten spatial features
        x = layers.TimeDistributed(layers.Flatten())(x)
        
        # LSTM layers for temporal modeling
        x = layers.LSTM(256, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(128)(x)
        x = layers.Dropout(0.3)(x)
        
        # Attention mechanism
        attention = layers.Dense(1, activation='tanh')(x)
        attention = layers.Flatten()(attention)
        attention_weights = layers.Activation('softmax')(attention)
        attention_mul = layers.Multiply()([x, attention_weights])
        
        # Dense layers for final prediction
        x = layers.Dense(256, activation='relu')(attention_mul)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer for coordinates (x, y)
        outputs = layers.Dense(2)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
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
        return self.model.predict(test_data)
    
    def evaluate(self, test_data, test_labels):
        return self.model.evaluate(test_data, test_labels)
    
    def save_model(self, filepath):
        self.model.save(filepath)
    
    @classmethod
    def load_model(cls, filepath):
        model = tf.keras.models.load_model(filepath)
        return model

class TrajectoryAwareLocalizationModel(tf.keras.Model):
    """
    Trajectory-aware localization model that incorporates temporal information
    for improved localization accuracy.
    """
    
    def __init__(self, sequence_length=5, l2_reg=0.01, dropout_rate=0.3):
        super().__init__()
        
        self.sequence_length = sequence_length
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # Base CSI processing model (shared weights for all timesteps)
        self.csi_processor = VanillaLocalizationModel(l2_reg=l2_reg, dropout_rate=dropout_rate)
        
        # Temporal processing layers
        self.lstm1 = layers.LSTM(256, return_sequences=True,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.lstm_bn1 = layers.BatchNormalization()
        self.lstm_dropout1 = layers.Dropout(dropout_rate)
        
        self.lstm2 = layers.LSTM(128, return_sequences=False,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.lstm_bn2 = layers.BatchNormalization()
        self.lstm_dropout2 = layers.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            dropout=dropout_rate
        )
        self.attention_bn = layers.BatchNormalization()
        
        # Final prediction layers
        self.fusion_dense = layers.Dense(128, activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.fusion_bn = layers.BatchNormalization()
        self.fusion_dropout = layers.Dropout(dropout_rate)
        
        self.output_layer = layers.Dense(2)  # x,y coordinates
        
        # Compile with competition metrics
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=combined_loss,
            metrics=[CombinedMetric(), 'mae']
        )
    
    def call(self, inputs, training=False):
        # inputs shape: (batch_size, sequence_length, 4, 8, 16, 2)
        batch_size = tf.shape(inputs)[0]
        
        # Process each timestep with the CSI processor
        sequence_features = []
        for t in range(self.sequence_length):
            features = self.csi_processor(inputs[:, t], training=training)
            sequence_features.append(features)
        
        # Stack features into a sequence
        x = tf.stack(sequence_features, axis=1)  # (batch_size, sequence_length, feature_dim)
        
        # Apply attention mechanism
        attention_output = self.attention(x, x, x, training=training)
        attention_output = self.attention_bn(attention_output, training=training)
        
        # Process temporal sequence
        x = self.lstm1(attention_output, training=training)
        x = self.lstm_bn1(x, training=training)
        x = self.lstm_dropout1(x, training=training)
        
        x = self.lstm2(x, training=training)
        x = self.lstm_bn2(x, training=training)
        x = self.lstm_dropout2(x, training=training)
        
        # Final prediction
        x = self.fusion_dense(x)
        x = self.fusion_bn(x, training=training)
        x = self.fusion_dropout(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "l2_reg": self.l2_reg,
            "dropout_rate": self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_callbacks(self):
        """Get training callbacks with competition-specific settings."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_combined_metric',
                patience=15,  # Increased patience for temporal model
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_combined_metric',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'results/isit2025/trajectory_aware/best_model.keras',
                monitor='val_combined_metric',
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='results/isit2025/trajectory_aware/logs',
                histogram_freq=1
            )
        ] 