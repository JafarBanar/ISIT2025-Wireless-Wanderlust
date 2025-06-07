import tensorflow as tf
from keras import layers, Model
import numpy as np
from src.utils.channel_sensing import AdvancedChannelSensor
from src.utils.attention import MultiHeadAttention

class EnhancedLocalizationModel(Model):
    def __init__(self, num_antenna_arrays=4, elements_per_array=8, num_freq_bands=16):
        super(EnhancedLocalizationModel, self).__init__()
        
        self.num_antenna_arrays = num_antenna_arrays
        self.elements_per_array = elements_per_array
        self.num_freq_bands = num_freq_bands
        
        # Feature extraction layers
        self.conv1 = layers.Conv2D(64, (3, 3), activation='relu')
        self.conv2 = layers.Conv2D(128, (3, 3), activation='relu')
        self.conv3 = layers.Conv2D(256, (3, 3), activation='relu')
        
        # Antenna array specific processing
        self.array_dense = [layers.Dense(128, activation='relu') for _ in range(num_antenna_arrays)]
        
        # Frequency band attention
        self.freq_attention = layers.MultiHeadAttention(num_heads=8, key_dim=32)
        
        # Position estimation layers
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(0.2)
        self.output_layer = layers.Dense(2)  # x, y coordinates
        
        # L2 regularization
        self.regularizer = tf.keras.regularizers.l2(0.01)
        
    def call(self, inputs, training=False):
        # Reshape input for antenna arrays and frequency bands
        x = tf.reshape(inputs, (-1, self.num_antenna_arrays, self.elements_per_array, self.num_freq_bands))
        
        # Apply convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Process each antenna array
        array_features = []
        for i in range(self.num_antenna_arrays):
            array_feat = self.array_dense[i](x[:, i, :, :])
            array_features.append(array_feat)
        
        # Combine array features
        x = tf.concat(array_features, axis=-1)
        
        # Apply frequency attention
        x = self.freq_attention(x, x, x)
        
        # Final position estimation
        x = self.dense1(x)
        if training:
            x = self.dropout1(x)
        x = self.dense2(x)
        if training:
            x = self.dropout2(x)
        return self.output_layer(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_antenna_arrays": self.num_antenna_arrays,
            "elements_per_array": self.elements_per_array,
            "num_freq_bands": self.num_freq_bands,
        })
        return config 

class ChannelAwareLocalizationModel:
    """Enhanced localization model with channel sensing integration."""
    
    def __init__(self, num_channels=16, sensing_threshold=0.3):
        self.localization_model = EnhancedLocalizationModel()
        self.channel_sensor = AdvancedChannelSensor(
            num_channels=num_channels,
            sensing_threshold=sensing_threshold
        )
        self.channel_weights = np.ones(num_channels) / num_channels
        
    def update_channel_weights(self, csi_data: np.ndarray):
        """Update channel weights based on CSI data quality."""
        # Calculate SNR for each channel
        snr_values = np.mean(np.abs(csi_data), axis=(1, 2))  # Average across antenna arrays and elements
        
        # Update channel sensor with SNR information
        for i, snr in enumerate(snr_values):
            self.channel_sensor.update_channel_state(i, False, snr)
        
        # Get channel quality metrics
        channel_stats = self.channel_sensor.get_channel_stats()
        quality_scores = np.array(channel_stats['channel_quality'])
        
        # Update weights using softmax
        self.channel_weights = np.exp(quality_scores) / np.sum(np.exp(quality_scores))
    
    def predict(self, csi_data: np.ndarray, training=False):
        """Make predictions with channel-aware processing."""
        # Update channel weights
        self.update_channel_weights(csi_data)
        
        # Apply channel weights to CSI data
        weighted_csi = csi_data * self.channel_weights[:, np.newaxis, np.newaxis]
        
        # Get localization prediction
        position = self.localization_model(weighted_csi, training=training)
        
        # Get channel statistics for monitoring
        channel_stats = self.channel_sensor.get_channel_stats()
        
        return {
            'position': position,
            'channel_quality': channel_stats['channel_quality'],
            'channel_weights': self.channel_weights
        }
    
    def train_step(self, csi_data: np.ndarray, true_position: np.ndarray):
        """Training step with channel-aware processing."""
        with tf.GradientTape() as tape:
            # Get prediction
            prediction = self.predict(csi_data, training=True)
            position = prediction['position']
            
            # Calculate loss
            mse_loss = tf.reduce_mean(tf.square(position - true_position))
            
            # Add regularization for channel weights
            weight_entropy = -tf.reduce_sum(self.channel_weights * tf.math.log(self.channel_weights + 1e-10))
            total_loss = mse_loss + 0.1 * weight_entropy
        
        # Update model weights
        gradients = tape.gradient(total_loss, self.localization_model.trainable_variables)
        self.localization_model.optimizer.apply_gradients(
            zip(gradients, self.localization_model.trainable_variables)
        )
        
        return {
            'loss': total_loss,
            'mse_loss': mse_loss,
            'weight_entropy': weight_entropy
        }
    
    def get_channel_stats(self):
        """Get current channel statistics."""
        return self.channel_sensor.get_channel_stats()
    
    def reset_channel_stats(self):
        """Reset channel statistics."""
        self.channel_sensor.reset_statistics() 