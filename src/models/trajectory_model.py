import tensorflow as tf
from tensorflow.keras import layers, Model
from src.isit2025.utils.competition_metrics import R90Metric, CombinedCompetitionMetric
import numpy as np
from typing import Dict, Tuple

class TemporalAttention(layers.Layer):
    """
    Temporal attention mechanism for trajectory prediction.
    Attends to previous timesteps to better predict current location.
    """
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
        # Attention layers
        self.W1 = layers.Dense(units)
        self.W2 = layers.Dense(units)
        self.V = layers.Dense(1)
    
    def call(self, query, values):
        """
        Apply attention mechanism.
        
        Args:
            query: Current hidden state (batch_size, query_dim)
            values: Sequence of previous states (batch_size, seq_len, value_dim)
        """
        # Expand query dims for broadcasting
        query_expanded = tf.expand_dims(query, 1)
        
        # Calculate attention scores
        score = self.V(tf.nn.tanh(
            self.W1(query_expanded) + self.W2(values)
        ))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention weights to values
        context = tf.reduce_sum(attention_weights * values, axis=1)
        
        return context, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

class TrajectoryAwareModel(tf.keras.Model):
    """
    Trajectory-aware localization model with LSTM and attention.
    """
    
    def __init__(self,
                 lstm_units: int = 128,
                 attention_heads: int = 4,
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.01,
                 sequence_length: int = 10,
                 **kwargs):
        """
        Initialize model.
        
        Args:
            lstm_units: Number of LSTM units
            attention_heads: Number of attention heads
            dropout_rate: Dropout rate
            l2_reg: L2 regularization factor
            sequence_length: Length of input sequences
        """
        super().__init__(**kwargs)
        
        self.lstm_units = lstm_units
        self.attention_heads = attention_heads
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.sequence_length = sequence_length
        
        # Feature extraction
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        # Calculate feature dimensions
        self.feature_dim = 128  # From GlobalAveragePooling2D
        
        # LSTM layers with input shapes
        self.lstm_1 = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
            input_shape=(None, self.feature_dim + 2)  # feature_dim + 2 for position coordinates
        )
        
        self.lstm_2 = tf.keras.layers.LSTM(
            lstm_units // 2,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_regularizer=tf.keras.regularizers.l2(l2_reg),
            input_shape=(None, lstm_units)  # Input from lstm_1
        )
        
        # Multi-head attention
        self.attention_layers = [
            TemporalAttention(lstm_units // attention_heads)
            for _ in range(attention_heads)
        ]
        
        # Output layers
        self.location_head = tf.keras.layers.Dense(
            2, name='location',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        )
        
        # Compile with competition metrics
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={'location': 'mse'},
            metrics={'location': [R90Metric(), CombinedCompetitionMetric(), 'mae']}
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: Dictionary containing:
                - csi_features: CSI features (batch, seq_len, n_arrays, n_elements, n_freq, 2)
                - prev_positions: Previous positions (batch, seq_len-1, 2)
            training: Whether in training mode
            
        Returns:
            Dictionary with predictions
        """
        # Extract CSI features
        csi_features = inputs['csi_features']
        prev_positions = inputs.get('prev_positions', None)
        
        # Get dimensions
        batch_size = tf.shape(csi_features)[0]
        seq_len = tf.shape(csi_features)[1]
        n_arrays = tf.shape(csi_features)[2]
        n_elements = tf.shape(csi_features)[3]
        n_freq = tf.shape(csi_features)[4]
        
        # Reshape CSI features for CNN
        csi_reshaped = tf.reshape(
            csi_features,
            [-1, n_arrays, n_elements, n_freq * 2]  # Combine freq and real/imag
        )
        
        # Extract features
        features = self.feature_extractor(csi_reshaped)  # (batch*seq_len, feature_dim)
        features = tf.reshape(features, [batch_size, seq_len, self.feature_dim])  # (batch, seq_len, feature_dim)
        
        # Handle previous positions
        if prev_positions is not None:
            # Concatenate with previous positions
            prev_pos_padded = tf.pad(
                prev_positions,
                [[0, 0], [0, 1], [0, 0]],  # Pad last timestep
                constant_values=0.0  # Use 0 for padding
            )
            features = tf.concat([features, prev_pos_padded], axis=-1)  # (batch, seq_len, feature_dim + 2)
        else:
            # If no previous positions, pad with zeros
            zero_pad = tf.zeros([batch_size, seq_len, 2])
            features = tf.concat([features, zero_pad], axis=-1)
        
        # Apply LSTM layers
        lstm_out_1 = self.lstm_1(features, training=training)
        lstm_out_2 = self.lstm_2(lstm_out_1, training=training)
        
        # Apply multi-head attention
        attention_outputs = []
        for attention_layer in self.attention_layers:
            context, _ = attention_layer(lstm_out_2[:, -1], lstm_out_2)  # Use last timestep as query
            attention_outputs.append(context)
        
        # Concatenate attention outputs
        context = tf.concat(attention_outputs, axis=-1)
        
        # Generate predictions
        location = self.location_head(context)  # (batch, 2)
        
        return {'location': location}
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'attention_heads': self.attention_heads,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'sequence_length': self.sequence_length
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    def get_callbacks(self, model_dir: str):
        """Get training callbacks with competition-specific settings."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_location_combined_competition_metric',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_location_combined_competition_metric',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{model_dir}/best_model.keras',
                monitor='val_location_combined_competition_metric',
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'{model_dir}/logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ] 