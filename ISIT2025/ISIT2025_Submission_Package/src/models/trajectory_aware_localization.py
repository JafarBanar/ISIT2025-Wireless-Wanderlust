import tensorflow as tf
import numpy as np
from src.utils.metrics import combined_loss, CombinedMetric
from src.utils.competition_metrics import R90Metric
from src.utils.logging_utils import setup_logging
from .vanilla_localization import VanillaLocalizationModel

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
        self.csi_processor = VanillaLocalizationModel(input_shape=(32, 1024, 2), num_classes=2)
        
        # Temporal processing layers
        self.lstm1 = tf.keras.layers.LSTM(256, return_sequences=True,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.lstm_bn1 = tf.keras.layers.BatchNormalization()
        self.lstm_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.lstm2 = tf.keras.layers.LSTM(128, return_sequences=True,
                               kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.lstm_bn2 = tf.keras.layers.BatchNormalization()
        self.lstm_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        # Attention mechanism
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=dropout_rate
        )
        self.attention_bn = tf.keras.layers.BatchNormalization()
        
        # Final prediction layers
        self.fusion_dense = tf.keras.layers.Dense(128, activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.fusion_bn = tf.keras.layers.BatchNormalization()
        self.fusion_dropout = tf.keras.layers.Dropout(dropout_rate)
        
        self.output_layer = tf.keras.layers.Dense(2)
        
        # Compile the model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=combined_loss,
            metrics=[CombinedMetric(), R90Metric()]
        )
    
    def call(self, inputs, training=False):
        # Process each timestep through the CSI processor
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        
        # Reshape input for processing
        x = tf.reshape(inputs, [-1, 32, 1024, 2])
        
        # Process through CSI model
        x = self.csi_processor(x)
        
        # Reshape back to sequence
        x = tf.reshape(x, [batch_size, timesteps, -1])
        
        # LSTM layers
        x = self.lstm1(x)
        x = self.lstm_bn1(x, training=training)
        x = self.lstm_dropout1(x, training=training)
        
        x = self.lstm2(x)
        x = self.lstm_bn2(x, training=training)
        x = self.lstm_dropout2(x, training=training)
        
        # Self-attention
        attention_output = self.attention(x, x, x)
        x = self.attention_bn(attention_output + x, training=training)
        
        # Final prediction
        x = self.fusion_dense(x)
        x = self.fusion_bn(x, training=training)
        x = self.fusion_dropout(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @classmethod
    def load_model(cls, filepath, sequence_length=5):
        """Load a saved model with custom objects."""
        instance = cls(sequence_length=sequence_length)
        instance.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                'TrajectoryAwareLocalizationModel': TrajectoryAwareLocalizationModel,
                'VanillaLocalizationModel': VanillaLocalizationModel,
                'CombinedMetric': CombinedMetric,
                'combined_loss': combined_loss,
                'R90Metric': R90Metric
            },
            compile=False
        )
        return instance
    
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