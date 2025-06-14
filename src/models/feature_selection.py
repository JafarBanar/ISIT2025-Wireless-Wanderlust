import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Dict
try:
    from src.utils.competition_metrics import R90Metric, CombinedCompetitionMetric
    from src.utils.metrics import combined_loss, CombinedMetric
    from src.utils.logging_utils import setup_logging
except ImportError:
    from utils.competition_metrics import R90Metric, CombinedCompetitionMetric
    from utils.metrics import combined_loss, CombinedMetric
    from utils.logging_utils import setup_logging

class FeatureSelector(tf.keras.layers.Layer):
    """
    Feature selection layer for grant-free random access.
    Implements a trainable attention-based mechanism to select important features.
    """
    
    def __init__(self,
                 n_features_to_select: int = 32,
                 temperature: float = 1.0,
                 l1_reg: float = 0.01,
                 **kwargs):
        """
        Initialize feature selector.
        
        Args:
            n_features_to_select: Number of features to select
            temperature: Temperature for Gumbel-Softmax
            l1_reg: L1 regularization for sparsity
        """
        super().__init__(**kwargs)
        self.n_features = n_features_to_select
        self.temperature = temperature
        self.l1_reg = l1_reg
        
        # Feature importance scores
        self.importance = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation=None,
                                kernel_regularizer=tf.keras.regularizers.l1(l1_reg))
        ])
        
        # Feature projection
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_features_to_select, use_bias=False)
        ])
    
    def build(self, input_shape):
        """Build the layer."""
        # Add dense layers with proper input shapes
        self.importance.build(input_shape)
        self.projection.build(input_shape)
        super().build(input_shape)
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply feature selection.
        
        Args:
            inputs: Input tensor (batch_size, feature_dim)
            training: Whether in training mode
            
        Returns:
            Tuple of (selected features, selection mask)
        """
        # Calculate feature importance scores
        scores = self.importance(inputs)  # (batch, 1)
        
        # Apply Gumbel-Softmax for differentiable selection
        if training:
            logits = scores / self.temperature
            mask = tf.nn.softmax(logits, axis=-1)
            
            # Get top-k features
            _, indices = tf.math.top_k(mask, k=self.n_features)
        else:
            # During inference, select top-k features deterministically
            _, indices = tf.math.top_k(scores, k=self.n_features)
        
        # Create mask for visualization
        mask = tf.reduce_sum(tf.one_hot(indices, tf.shape(scores)[1]), axis=1)
        mask = tf.expand_dims(mask, -1)  # Add feature dimension
        
        # Project features
        selected = self.projection(inputs)  # Project all features
        
        return selected, mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_features_to_select': self.n_features,
            'temperature': self.temperature,
            'l1_reg': self.l1_reg
        })
        return config

class ChannelSensor(tf.keras.layers.Layer):
    """
    Channel sensing layer for grant-free random access.
    Estimates channel occupancy and interference.
    """
    
    def __init__(self, n_arrays: int = 4, n_elements: int = 8, n_freq: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_arrays = n_arrays
        self.n_elements = n_elements
        self.n_freq = n_freq
        
        # Energy computation layers
        self.energy_conv = None  # Will be created in build()
        self.energy_bn = tf.keras.layers.BatchNormalization()
        
        # Attention mechanism
        self.attention_dense = None  # Will be created in build()
        self.attention_bn = tf.keras.layers.BatchNormalization()
        
        # Interference estimation
        self.interference_estimator = None  # Will be created in build()
    
    def build(self, input_shape):
        """Build the layer with proper input shape handling."""
        # Debug print for input shape
        print("ChannelSensor input_shape:", input_shape)
        
        # Validate input shape
        if not isinstance(input_shape, (tuple, list)):
            raise ValueError(f"Expected tuple/list input shape, got {type(input_shape)}")
        
        # Ensure input shape has correct dimensions
        if len(input_shape) != 6:  # (batch, seq_len, n_arrays, n_elements, n_freq, 2)
            raise ValueError(f"Expected input shape with 6 dimensions, got {len(input_shape)}")
        
        # Create energy computation layers
        self.energy_conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation='relu',
            padding='same',
            name='energy_conv'
        )
        
        # Create attention mechanism
        self.attention_dense = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='attention_dense'
        )
        
        # Create interference estimator
        self.interference_estimator = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2, padding='same'),
            tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: CSI features tensor (batch_size, seq_len, n_arrays, n_elements, n_freq, 2)
            training: Whether in training mode
            
        Returns:
            Tuple of (occupancy_mask, interference)
        """
        # Get batch size and sequence length
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        # Compute energy (magnitude squared)
        energy = tf.reduce_sum(tf.square(inputs), axis=-1)  # (batch, seq_len, n_arrays, n_elements, n_freq)
        
        # Reshape for Conv2D
        energy = tf.reshape(
            energy,
            [batch_size * seq_len, self.n_arrays * self.n_elements, self.n_freq, 1]
        )
        
        # Apply energy computation
        energy = self.energy_conv(energy)
        energy = self.energy_bn(energy, training=training)
        
        # Compute attention weights
        attention = self.attention_dense(energy)
        attention = self.attention_bn(attention, training=training)
        
        # Apply attention to energy features
        weighted_energy = energy * attention  # Shape: [batch_size, seq_len, n_arrays*n_elements, n_freq, 1]
        
        # Debug prints for shapes
        print("Weighted energy shape:", tf.shape(weighted_energy))
        
        # Reshape for interference estimation
        # First flatten the spatial dimensions: [batch_size*seq_len, n_arrays*n_elements*n_freq]
        energy_flat = tf.reshape(
            weighted_energy,
            [batch_size * seq_len, self.n_arrays * self.n_elements * self.n_freq]
        )
        
        # Add channel dimension for interference estimator
        energy_for_interference = tf.expand_dims(energy_flat, axis=-1)  # Shape: [batch_size*seq_len, n_arrays*n_elements*n_freq, 1]
        
        print("Energy for interference shape:", tf.shape(energy_for_interference))
        
        # Estimate interference
        interference = self.interference_estimator(energy_for_interference)  # Shape: [batch_size*seq_len, 1]
        
        print("Interference shape before reshape:", tf.shape(interference))
        
        # Reshape interference to (batch_size, seq_len, 1)
        interference = tf.reshape(interference, [batch_size, seq_len, 1])
        
        print("Interference shape after reshape:", tf.shape(interference))
        
        # Compute occupancy mask from attention weights
        # First reshape attention to (batch_size*seq_len, n_arrays*n_elements, n_freq)
        attention_reshaped = tf.reshape(
            attention,
            [batch_size * seq_len, self.n_arrays * self.n_elements, self.n_freq]
        )
        
        # Average attention across frequency: (batch_size*seq_len, n_arrays*n_elements)
        occupancy_mask = tf.reduce_mean(attention_reshaped, axis=2)
        
        # Add channel dimension: (batch_size*seq_len, n_arrays*n_elements, 1)
        occupancy_mask = tf.expand_dims(occupancy_mask, axis=-1)
        
        # Reshape to original dimensions: (batch_size, seq_len, n_arrays, n_elements, 1)
        occupancy_mask = tf.reshape(
            occupancy_mask,
            [batch_size, seq_len, self.n_arrays, self.n_elements, 1]
        )
        
        # Debug prints
        print("Attention shape:", tf.shape(attention))
        print("Attention reshaped shape:", tf.shape(attention_reshaped))
        print("Occupancy mask shape before final reshape:", tf.shape(occupancy_mask))
        
        return occupancy_mask, interference
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_arrays': self.n_arrays,
            'n_elements': self.n_elements,
            'n_freq': self.n_freq
        })
        return config

class FeatureSelectionModel(tf.keras.Model):
    """
    Feature selection model that learns to identify and use the most relevant
    CSI features for localization.
    """
    
    def __init__(self, input_shape=(32, 1024, 2), num_classes=2, l2_reg=0.01, dropout_rate=0.3):
        super().__init__()
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        
        # Feature selection layers
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        
        # Feature importance layers
        self.importance_conv = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        self.importance_bn = tf.keras.layers.BatchNormalization()
        
        # Dense layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dense_bn1 = tf.keras.layers.BatchNormalization()
        self.dense_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        self.dense2 = tf.keras.layers.Dense(128, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.dense_bn2 = tf.keras.layers.BatchNormalization()
        self.dense_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        
        self.output_layer = tf.keras.layers.Dense(num_classes)
        
        # Compile the model
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=combined_loss,
            metrics=[CombinedMetric(), R90Metric()]
        )
    
    def call(self, inputs, training=False):
        # Feature extraction
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        x = self.dropout1(x, training=training)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        x = self.dropout2(x, training=training)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        x = self.dropout3(x, training=training)
        
        # Feature importance
        importance = self.importance_conv(x)
        importance = self.importance_bn(importance, training=training)
        
        # Apply feature importance
        x = x * importance
        
        # Dense layers
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense_bn1(x, training=training)
        x = self.dense_dropout1(x, training=training)
        
        x = self.dense2(x)
        x = self.dense_bn2(x, training=training)
        x = self.dense_dropout2(x, training=training)
        
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    @classmethod
    def load_model(cls, filepath, input_shape=(32, 1024, 2), num_classes=2):
        """Load a saved model with custom objects."""
        instance = cls(input_shape=input_shape, num_classes=num_classes)
        instance.model = tf.keras.models.load_model(
            filepath,
            custom_objects={
                'FeatureSelectionModel': FeatureSelectionModel,
                'CombinedMetric': CombinedMetric,
                'combined_loss': combined_loss,
                'R90Metric': R90Metric
            },
            compile=False
        )
        return instance

@keras.utils.register_keras_serializable()
class GrantFreeAccessModel(keras.Model):
    """
    Model for grant-free random access with feature selection.
    """
    
    def __init__(
        self,
        n_arrays: int = 4,
        n_elements: int = 8,
        n_freq: int = 16,
        seq_len: int = 10,
        **kwargs
    ):
        """
        Initialize model.
        
        Args:
            n_arrays: Number of antenna arrays
            n_elements: Number of elements per array
            n_freq: Number of frequency bins
            seq_len: Sequence length
        """
        super().__init__(**kwargs)
        
        self.n_arrays = n_arrays
        self.n_elements = n_elements
        self.n_freq = n_freq
        self.seq_len = seq_len
        
        # Channel sensing layer
        self.channel_sensor = ChannelSensor(n_arrays, n_elements, n_freq)
        
        # Feature selection layers with TimeDistributed
        self.feature_selector = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(256, activation='relu'),
                name='feature_dense1'
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization(),
                name='feature_bn1'
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(128, activation='relu'),
                name='feature_dense2'
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.BatchNormalization(),
                name='feature_bn2'
            ),
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(n_arrays * n_elements * n_freq * 2, activation='sigmoid'),
                name='feature_output'
            )
        ])
        
        # Position prediction layers
        self.position_predictor = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, name='position_lstm1'),
            tf.keras.layers.BatchNormalization(name='position_bn1'),
            tf.keras.layers.LSTM(64, return_sequences=True, name='position_lstm2'),
            tf.keras.layers.BatchNormalization(name='position_bn2'),
            tf.keras.layers.Dense(32, activation='relu', name='position_dense1'),
            tf.keras.layers.Dense(2, name='position_output')  # x, y coordinates
        ])
    
    def build(self, input_shape):
        """Build the model layers."""
        # Input shapes should be a dictionary with 'csi_features' and 'prev_positions'
        if not isinstance(input_shape, dict):
            raise ValueError("Model expects a dictionary input with 'csi_features' and 'prev_positions'")
        
        if 'csi_features' not in input_shape or 'prev_positions' not in input_shape:
            raise ValueError("Input shape must contain 'csi_features' and 'prev_positions'")
        
        csi_shape = input_shape['csi_features']
        prev_pos_shape = input_shape['prev_positions']
        
        # Build channel sensor with CSI shape
        self.channel_sensor.build(csi_shape)
        
        # Build feature selector
        # Reshape CSI to (batch_size, seq_len, features)
        feature_shape = (csi_shape[0], csi_shape[1], self.n_arrays * self.n_elements * self.n_freq * 2)
        self.feature_selector.build(feature_shape)
        
        # Build position predictor
        # Input shape: (batch_size, seq_len, features + prev_pos)
        position_shape = (csi_shape[0], csi_shape[1], self.n_arrays * self.n_elements * self.n_freq * 2 + 2)
        self.position_predictor.build(position_shape)
        
        super().build(input_shape)
    
    def call(self, inputs, training=False):
        # Unpack inputs
        csi_features = inputs['csi_features']  # (batch_size, seq_len, n_arrays, n_elements, n_freq, 2)
        prev_positions = inputs['prev_positions']  # (batch_size, seq_len-1, 2)
        
        # Get batch size and sequence length
        batch_size = tf.shape(csi_features)[0]
        seq_len = tf.shape(csi_features)[1]
        
        # Channel sensing
        occupancy_mask, interference = self.channel_sensor(csi_features)
        
        # Reshape CSI to (batch_size, seq_len, features)
        csi_reshaped = tf.reshape(
            csi_features,
            [batch_size, seq_len, self.n_arrays * self.n_elements * self.n_freq * 2]
        )
        
        # Debug prints
        print("CSI reshaped shape:", tf.shape(csi_reshaped))
        
        # Feature selection
        feature_weights = self.feature_selector(csi_reshaped)  # (batch_size, seq_len, n_arrays*n_elements*n_freq*2)
        
        print("Feature weights shape:", tf.shape(feature_weights))
        
        # Apply feature weights
        selected_features = csi_reshaped * feature_weights
        
        # Concatenate with previous positions
        # Pad prev_positions to match sequence length
        prev_pos_padded = tf.pad(
            prev_positions,
            [[0, 0], [1, 0], [0, 0]],  # Add one position at the start
            constant_values=0.0
        )
        
        # Concatenate features and positions
        combined_features = tf.concat([selected_features, prev_pos_padded], axis=-1)
        
        # Predict positions
        positions = self.position_predictor(combined_features)
        
        return {
            'positions': positions,  # (batch_size, seq_len, 2)
            'occupancy_mask': occupancy_mask,  # (batch_size, seq_len, n_arrays, n_elements, 1)
            'interference': interference  # (batch_size, seq_len, 1)
        }
    
    def compile(self, **kwargs):
        """Compile model with appropriate losses and metrics."""
        super().compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                'positions': tf.keras.losses.MeanSquaredError(),
                'occupancy_mask': tf.keras.losses.BinaryCrossentropy(),
                'interference': tf.keras.losses.MeanSquaredError()
            },
            loss_weights={
                'positions': 1.0,
                'occupancy_mask': 0.2,
                'interference': 0.1
            },
            metrics={
                'positions': [
                    tf.keras.metrics.MeanAbsoluteError(name='mae'),
                    R90Metric(name='r90'),
                    CombinedCompetitionMetric(name='combined_score')
                ],
                'occupancy_mask': ['accuracy'],
                'interference': ['mse']
            },
            **kwargs
        )
    
    def get_callbacks(self, model_dir: str):
        """Get training callbacks with competition-specific settings."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_positions_mae',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_positions_mae',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{model_dir}/best_model.keras',
                monitor='val_positions_mae',
                save_best_only=True,
                mode='min'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=f'{model_dir}/logs',
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_arrays': self.n_arrays,
            'n_elements': self.n_elements,
            'n_freq': self.n_freq,
            'seq_len': self.seq_len
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 