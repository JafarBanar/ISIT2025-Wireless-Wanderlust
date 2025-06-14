import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import tensorflow as tf
import logging
from tensorflow.keras import layers

class AdvancedChannelSensor(layers.Layer):
    """Advanced channel sensing and quality monitoring system."""
    
    def __init__(self, num_filters=64, **kwargs):
        super(AdvancedChannelSensor, self).__init__(**kwargs)
        self.num_filters = num_filters
        
        # Convolutional layers for channel feature extraction
        self.conv1 = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )
        self.conv2 = layers.Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )
        
        # Batch normalization
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        
        # Pooling
        self.pool = layers.MaxPooling2D(pool_size=(2, 2))
        
    def call(self, inputs, training=None):
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        
        return x
        
    def get_config(self):
        config = super(AdvancedChannelSensor, self).get_config()
        config.update({
            'num_filters': self.num_filters
        })
        return config
    
    def update_channel_state(self, channel_idx: int, is_occupied: bool, snr: float):
        """Update the state of a specific channel."""
        if not 0 <= channel_idx < self.num_channels:
            raise ValueError(f"Channel index {channel_idx} out of range [0, {self.num_channels-1}]")
        
        self.channel_states['occupied'][channel_idx] = is_occupied
        self.channel_states['snr'][channel_idx] = snr
        self.channel_states['last_update'][channel_idx] = self.time_step
        
        # Update quality based on SNR and occupancy
        quality = self._calculate_quality(snr, is_occupied)
        self.channel_states['quality'][channel_idx] = quality
        
        # Update collision count
        if is_occupied:
            self.channel_states['collision_count'][channel_idx] += 1
        
        self.time_step += 1
    
    def _calculate_quality(self, snr: float, is_occupied: bool) -> float:
        """Calculate channel quality based on SNR and occupancy."""
        if is_occupied:
            return 0.0
        
        # Normalize SNR to [0, 1] range
        normalized_snr = min(max(snr / 20.0, 0.0), 1.0)  # Assuming max SNR of 20dB
        
        # Apply sigmoid function for smooth transition
        quality = 1.0 / (1.0 + np.exp(-10 * (normalized_snr - 0.5)))
        
        return quality
    
    def get_channel_stats(self) -> Dict[str, Any]:
        """Get current channel statistics."""
        return {
            'channel_quality': np.array(self.channel_states['quality']),
            'snr': np.array(self.channel_states['snr']),
            'collision_rate': np.mean(self.channel_states['occupied']),
            'average_quality': np.mean(self.channel_states['quality']),
            'quality_std': np.std(self.channel_states['quality']),
            'collision_counts': np.array(self.channel_states['collision_count'])
        }
    
    def reset_statistics(self):
        """Reset all channel statistics."""
        self.channel_states = {
            'occupied': [False] * self.num_channels,
            'snr': [0.0] * self.num_channels,
            'quality': [1.0] * self.num_channels,
            'collision_count': [0] * self.num_channels,
            'last_update': [0] * self.num_channels
        }
        self.time_step = 0
    
    def get_best_channels(self, num_channels: int = None) -> List[int]:
        """Get indices of the best quality channels."""
        if num_channels is None:
            num_channels = self.num_channels
        
        # Sort channels by quality
        channel_qualities = list(enumerate(self.channel_states['quality']))
        sorted_channels = sorted(channel_qualities, key=lambda x: x[1], reverse=True)
        
        return [channel[0] for channel in sorted_channels[:num_channels]]
    
    def get_channel_weights(self) -> np.ndarray:
        """Get normalized channel weights based on quality."""
        qualities = np.array(self.channel_states['quality'])
        weights = np.exp(qualities) / np.sum(np.exp(qualities))
        return weights

    def update_channel_memory(self, channel_states: np.ndarray):
        """Update channel state memory"""
        self.channel_memory[self.current_memory_idx] = channel_states
        self.current_memory_idx = (self.current_memory_idx + 1) % self.memory_length
        
    def estimate_channel_quality(self, channel_idx: int) -> float:
        """Estimate channel quality based on history"""
        channel_history = self.channel_memory[:, channel_idx]
        recent_history = channel_history[-20:]  # Last 20 observations
        
        # Calculate metrics
        occupancy_rate = np.mean(recent_history)
        stability = 1.0 - np.std(recent_history)
        collision_rate = sum(1 for x in self.collision_history[-50:] 
                           if x == channel_idx) / 50.0
        
        # Combine metrics
        quality = (0.4 * (1 - occupancy_rate) +  # Prefer less occupied channels
                  0.3 * stability +               # Prefer stable channels
                  0.3 * (1 - collision_rate))     # Avoid channels with collisions
        
        return quality
    
    def sense_channel(self, channel_idx: int, snr: float = None) -> bool:
        """Enhanced channel sensing with SNR consideration"""
        # Basic sensing
        noise_level = np.random.normal(0, 0.1)
        basic_availability = self.channel_states['occupied'][channel_idx] + noise_level < self.sensing_threshold
        
        # SNR check if provided
        if snr is not None:
            snr_check = snr > 10.0
            return basic_availability and snr_check
        
        return basic_availability
    
    def predict_channel_state(self, channel_idx: int) -> float:
        """Predict future channel state using historical data"""
        channel_history = self.channel_memory[:, channel_idx]
        
        # Simple prediction using exponential moving average
        alpha = 0.3
        prediction = 0.0
        for i in range(len(channel_history) - 1, -1, -1):
            prediction = alpha * channel_history[i] + (1 - alpha) * prediction
            
        return prediction
    
    def get_best_channel(self) -> Tuple[int, float]:
        """Get best channel based on multiple criteria"""
        channel_scores = np.zeros(self.num_channels)
        
        for i in range(self.num_channels):
            # Combine multiple metrics
            quality = self.estimate_channel_quality(i)
            prediction = self.predict_channel_state(i)
            current_state = float(self.sense_channel(i))
            
            # Weight the factors
            channel_scores[i] = (0.4 * quality +          # Historical quality
                               0.3 * (1 - prediction) +   # Predicted availability
                               0.3 * current_state)       # Current state
            
        best_channel = np.argmax(channel_scores)
        return best_channel, channel_scores[best_channel]
    
    def handle_collision(self, channel_idx: int) -> int:
        """Enhanced collision handling with adaptive backoff"""
        self.collision_history.append(channel_idx)
        
        # Count recent collisions
        recent_collisions = sum(1 for x in self.collision_history[-20:] 
                              if x == channel_idx)
        
        # Adaptive backoff window based on collision history
        base_window = 2 ** recent_collisions - 1
        channel_quality = self.estimate_channel_quality(channel_idx)
        
        # Adjust window based on channel quality
        adjusted_window = int(base_window * (1 + (1 - channel_quality)))
        max_window = 64
        
        backoff_window = min(adjusted_window, max_window)
        return np.random.randint(0, backoff_window + 1)
    
    def get_transmission_opportunity(self) -> Optional[int]:
        """
        Get the next available transmission opportunity
        Returns channel index if available, None if should wait
        """
        # Check backoff counters
        self.backoff_counters = np.maximum(0, self.backoff_counters - 1)
        
        # Find available channels
        available_channels = [i for i in range(self.num_channels) 
                            if self.backoff_counters[i] == 0 and self.sense_channel(i)]
        
        if not available_channels:
            return None
            
        # Select best available channel
        best_channel = min(available_channels, 
                          key=lambda x: len([c for c in self.collision_history[-10:] if c == x]))
        return best_channel 