import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple
from .channel_sensing import AdvancedChannelSensor

class RandomAccessSimulator:
    """Simulator for grant-free random access with channel sensing."""
    
    def __init__(self, 
                 num_devices: int = 100,
                 num_channels: int = 16,
                 sensing_threshold: float = -80.0,
                 max_retries: int = 3):
        self.num_devices = num_devices
        self.num_channels = num_channels
        self.sensing_threshold = sensing_threshold
        self.max_retries = max_retries
        self.channel_sensor = AdvancedChannelSensor()
        
        # Initialize device states
        self.device_states = {
            'active': np.zeros(num_devices, dtype=bool),
            'channel': np.zeros(num_devices, dtype=int),
            'retries': np.zeros(num_devices, dtype=int),
            'success': np.zeros(num_devices, dtype=bool)
        }
        
        # Statistics
        self.stats = {
            'total_attempts': 0,
            'successful_transmissions': 0,
            'collisions': 0,
            'channel_busy': 0,
            'failed_after_retries': 0
        }
    
    def _sense_channels(self, device_idx: int) -> Tuple[int, float]:
        """Sense available channels for a device."""
        # Get CSI for the device (simulated)
        csi = self._get_device_csi(device_idx)
        
        # Use channel sensor to find best channel
        channel_quality = self.channel_sensor.estimate_channel_quality(csi)
        best_channel = np.argmax(channel_quality)
        best_quality = channel_quality[best_channel]
        
        return best_channel, best_quality
    
    def _get_device_csi(self, device_idx: int) -> np.ndarray:
        """Simulate CSI for a device (replace with actual CSI data)."""
        # Simulate CSI with some randomness
        csi = np.random.normal(0, 1, (4, 8, 16, 2)) + 1j * np.random.normal(0, 1, (4, 8, 16, 2))
        return np.abs(csi)  # Return magnitude
    
    def _handle_collision(self, channel: int) -> None:
        """Handle collision on a channel."""
        self.stats['collisions'] += 1
        # Mark all devices on this channel as failed
        colliding_devices = np.where(
            (self.device_states['active']) & 
            (self.device_states['channel'] == channel)
        )[0]
        
        for device_idx in colliding_devices:
            self.device_states['retries'][device_idx] += 1
            if self.device_states['retries'][device_idx] >= self.max_retries:
                self.stats['failed_after_retries'] += 1
                self.device_states['active'][device_idx] = False
            else:
                # Device will retry in next round
                self.device_states['channel'][device_idx] = -1
    
    def simulate_round(self) -> Dict[str, int]:
        """Simulate one round of random access."""
        # Reset active devices for this round
        self.device_states['active'] = np.ones(self.num_devices, dtype=bool)
        self.device_states['channel'] = np.zeros(self.num_devices, dtype=int)
        
        # First phase: Channel sensing
        for device_idx in range(self.num_devices):
            if not self.device_states['active'][device_idx]:
                continue
                
            channel, quality = self._sense_channels(device_idx)
            if quality < self.sensing_threshold:
                self.stats['channel_busy'] += 1
                self.device_states['active'][device_idx] = False
            else:
                self.device_states['channel'][device_idx] = channel
        
        # Second phase: Transmission
        for channel in range(self.num_channels):
            devices_on_channel = np.where(
                (self.device_states['active']) & 
                (self.device_states['channel'] == channel)
            )[0]
            
            if len(devices_on_channel) > 1:
                # Collision detected
                self._handle_collision(channel)
            elif len(devices_on_channel) == 1:
                # Successful transmission
                device_idx = devices_on_channel[0]
                self.device_states['success'][device_idx] = True
                self.stats['successful_transmissions'] += 1
                self.device_states['active'][device_idx] = False
        
        self.stats['total_attempts'] += self.num_devices
        return self.stats.copy()
    
    def run_simulation(self, num_rounds: int = 100) -> Dict[str, List[int]]:
        """Run multiple rounds of simulation."""
        history = {
            'successful_transmissions': [],
            'collisions': [],
            'channel_busy': [],
            'failed_after_retries': []
        }
        
        for _ in range(num_rounds):
            stats = self.simulate_round()
            for key in history:
                history[key].append(stats[key])
        
        return history
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from simulation results."""
        total_attempts = self.stats['total_attempts']
        if total_attempts == 0:
            return {
                'success_rate': 0.0,
                'collision_rate': 0.0,
                'channel_busy_rate': 0.0,
                'failure_rate': 0.0
            }
        
        return {
            'success_rate': self.stats['successful_transmissions'] / total_attempts,
            'collision_rate': self.stats['collisions'] / total_attempts,
            'channel_busy_rate': self.stats['channel_busy'] / total_attempts,
            'failure_rate': self.stats['failed_after_retries'] / total_attempts
        }

def main():
    # Example usage
    simulator = RandomAccessSimulator(
        num_devices=100,
        num_channels=16,
        sensing_threshold=-80.0,
        max_retries=3
    )
    
    # Run simulation
    history = simulator.run_simulation(num_rounds=100)
    
    # Print results
    metrics = simulator.get_performance_metrics()
    print("\nSimulation Results:")
    print(f"Success Rate: {metrics['success_rate']:.2%}")
    print(f"Collision Rate: {metrics['collision_rate']:.2%}")
    print(f"Channel Busy Rate: {metrics['channel_busy_rate']:.2%}")
    print(f"Failure Rate: {metrics['failure_rate']:.2%}")

if __name__ == '__main__':
    main() 