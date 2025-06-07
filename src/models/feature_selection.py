import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.isit2025.utils.competition_metrics import R90Metric, CombinedCompetitionMetric

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
    Implements energy detection and interference estimation.
    """
    
    def __init__(self,
                 energy_threshold: float = 0.1,
                 interference_window: int = 5,
                 **kwargs):
        """
        Initialize channel sensor.
        
        Args:
            energy_threshold: Threshold for channel occupancy detection
            interference_window: Window size for interference estimation
        """
        super().__init__(**kwargs)
        self.energy_threshold = energy_threshold
        self.interference_window = interference_window
        
        # Channel state estimation with attention
        self.channel_attention = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Energy detection with spectral analysis
        self.spectral_analyzer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D(),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Interference estimation with temporal convolution
        self.interference_estimator = tf.keras.Sequential([
            # Temporal convolution
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # Upsampling path
            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation('relu'),
            
            # Final interference estimation
            tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')
        ])
    
    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Perform channel sensing.
        
        Args:
            inputs: CSI features (batch, seq_len, n_arrays, n_elements, n_freq, 2)
            
        Returns:
            Tuple of (channel occupancy mask, interference estimates)
        """
        # Get dimensions
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        n_arrays = input_shape[2]
        n_elements = input_shape[3]
        n_freq = input_shape[4]
        
        # Calculate signal energy
        energy = tf.reduce_sum(tf.square(inputs), axis=-1)  # Sum over real/imag
        
        # Apply channel attention
        energy_flat = tf.reshape(energy, [-1, n_arrays * n_elements * n_freq])
        attention_weights = self.channel_attention(energy_flat)
        attention_weights = tf.reshape(attention_weights, [batch_size, seq_len, n_arrays, n_elements, n_freq])
        
        # Detect channel occupancy with spectral analysis
        # Reshape energy for 2D convolutions
        energy_reshaped = tf.reshape(energy, [-1, n_elements, n_freq, 1])
        occupancy = self.spectral_analyzer(energy_reshaped)
        occupancy = tf.reshape(occupancy, [batch_size, seq_len, n_arrays, n_elements, n_freq])
        occupancy_mask = tf.cast(occupancy > self.energy_threshold, tf.float32)
        
        # Estimate interference with temporal convolution
        # Reshape energy for interference estimation
        energy_temporal = tf.reshape(energy, [-1, n_elements, n_freq, 1])
        interference = self.interference_estimator(energy_temporal)
        interference = tf.reshape(interference, [batch_size, seq_len, n_arrays, n_elements, n_freq])
        
        # Apply attention to interference estimates
        interference = interference * attention_weights
        
        return occupancy_mask, interference
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'energy_threshold': self.energy_threshold,
            'interference_window': self.interference_window
        })
        return config

class GrantFreeAccessModel(tf.keras.Model):
    """
    Model for grant-free random access with feature selection.
    """
    
    def __init__(self,
                 n_features_to_select: int = 32,
                 n_priority_levels: int = 4,
                 dropout_rate: float = 0.3,
                 l2_reg: float = 0.01,
                 sequence_length: int = 10,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Store parameters
        self.n_features = n_features_to_select
        self.n_priorities = n_priority_levels
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.sequence_length = sequence_length
        
        # Channel sensing
        self.channel_sensor = ChannelSensor()
        
        # Feature selection
        self.feature_selector = FeatureSelector(
            n_features_to_select=n_features_to_select
        )
        
        # Localization head
        self.location_head = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(128, activation='relu',
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_rate),
            
            tf.keras.layers.Dense(2, name='location')  # x,y coordinates
        ])
        
        # Priority prediction
        self.priority_head = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(n_priority_levels, activation='softmax', name='priority')
        ])
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> Dict[str, tf.Tensor]:
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
        # Get CSI features
        csi_features = inputs['csi_features']  # (batch, seq_len, n_arrays, n_elements, n_freq, 2)
        
        # Get dimensions
        batch_size = tf.shape(csi_features)[0]
        seq_len = tf.shape(csi_features)[1]
        n_arrays = tf.shape(csi_features)[2]
        n_elements = tf.shape(csi_features)[3]
        n_freq = tf.shape(csi_features)[4]
        
        # Perform channel sensing
        occupancy_mask, interference = self.channel_sensor(csi_features)
        
        # Reshape CSI features for feature selection
        csi_reshaped = tf.reshape(
            csi_features,
            [batch_size * seq_len, n_arrays * n_elements * n_freq * 2]
        )
        
        # Apply feature selection
        selected_features, selection_mask = self.feature_selector(csi_reshaped, training=training)
        
        # Reshape back to sequence
        selected_features = tf.reshape(selected_features, [batch_size, seq_len, -1])
        
        # Generate predictions using last timestep
        location = self.location_head(selected_features[:, -1])  # Use last timestep
        priority = self.priority_head(selected_features[:, -1])
        
        return {
            'location': location,
            'priority': priority,
            'channel_occupancy': occupancy_mask,
            'interference': interference,
            'selection_mask': selection_mask
        }
    
    def compile(self, optimizer=None, *args, **kwargs):
        """
        Compile model with competition metrics.
        
        Args:
            optimizer: Optimizer instance
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
        # Add competition metrics
        metrics = {
            'location': [
                tf.keras.metrics.MeanAbsoluteError(name='mae'),
                R90Metric(name='r90'),
                CombinedCompetitionMetric(name='combined_score')
            ]
        }
        
        super().compile(
            optimizer=optimizer,
            loss={'location': 'mse'},
            metrics=metrics,
            *args,
            **kwargs
        )
    
    def get_callbacks(self, model_dir: str):
        """Get training callbacks with competition-specific settings."""
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_location_mae',
                patience=10,
                restore_best_weights=True,
                mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_location_mae',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                mode='min'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                f'{model_dir}/best_model.keras',
                monitor='val_location_mae',
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
            'n_features_to_select': self.feature_selector.n_features,
            'n_priority_levels': self.priority_head.output_shape[-1],
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config) 