import tensorflow as tf
import numpy as np
from models.enhanced_localization import EnhancedLocalizationModel
from utils.channel_sensing import AdvancedChannelSensor
from tests.error_analysis import ErrorAnalyzer
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, model_path: str, test_data_path: str):
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.model = None
        self.channel_sensor = None
        self.results_dir = os.path.join('results', 
                                      f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model(self):
        """Load the trained model"""
        self.model = tf.keras.models.load_model(self.model_path)
        
    def evaluate_localization(self, X_test, y_test):
        """Evaluate localization performance"""
        predictions = self.model.predict(X_test)
        
        # Calculate basic metrics
        mae = np.mean(np.abs(predictions - y_test))
        mse = np.mean(np.square(predictions - y_test))
        rmse = np.sqrt(mse)
        
        # Calculate R90 metric
        errors = np.linalg.norm(predictions - y_test, axis=1)
        r90 = np.percentile(errors, 90)
        
        # Calculate combined score
        combined_score = 0.7 * mae + 0.3 * r90
        
        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r90': float(r90),
            'combined_score': float(combined_score)
        }
    
    def evaluate_trajectory(self, X_test_seq, y_test_seq):
        """Evaluate trajectory prediction performance"""
        predictions = self.model.predict(X_test_seq)
        
        # Calculate trajectory metrics
        endpoint_error = np.mean(np.linalg.norm(predictions[:, -1] - y_test_seq[:, -1], axis=1))
        smoothness = np.mean(np.linalg.norm(np.diff(predictions, axis=1), axis=2))
        
        # Calculate time-weighted error
        sequence_length = predictions.shape[1]
        weights = np.linspace(0.5, 1.0, sequence_length)
        weighted_errors = np.mean(
            weights * np.linalg.norm(predictions - y_test_seq, axis=2),
            axis=1
        )
        
        return {
            'endpoint_error': float(endpoint_error),
            'smoothness': float(smoothness),
            'weighted_error': float(np.mean(weighted_errors))
        }
    
    def evaluate_channel_sensing(self, num_channels=16, num_trials=1000):
        """Evaluate channel sensing performance"""
        self.channel_sensor = AdvancedChannelSensor(num_channels)
        
        # Simulate channel access
        collisions = 0
        successful_transmissions = 0
        channel_utilization = np.zeros(num_channels)
        
        for _ in range(num_trials):
            # Simulate random channel occupancy
            channel_states = np.random.binomial(1, 0.3, num_channels)
            snr_values = np.random.normal(15, 5, num_channels)
            
            # Update channel states
            for i in range(num_channels):
                self.channel_sensor.update_channel_state(i, bool(channel_states[i]), snr_values[i])
            
            # Try to transmit
            selected_channel = self.channel_sensor.get_transmission_opportunity()
            if selected_channel is not None:
                channel_utilization[selected_channel] += 1
                if channel_states[selected_channel]:
                    collisions += 1
                else:
                    successful_transmissions += 1
        
        return {
            'collision_rate': float(collisions / num_trials),
            'success_rate': float(successful_transmissions / num_trials),
            'channel_utilization': channel_utilization.tolist(),
            'channel_stats': self.channel_sensor.get_channel_stats()
        }
    
    def generate_visualizations(self, results):
        """Generate performance visualizations"""
        # Localization error distribution
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Error distribution
        plt.subplot(131)
        sns.histplot(results['error_distribution'], bins=50)
        plt.title('Localization Error Distribution')
        plt.xlabel('Error (meters)')
        plt.ylabel('Count')
        
        # Plot 2: Channel utilization
        plt.subplot(132)
        sns.barplot(x=list(range(len(results['channel_sensing']['channel_utilization']))),
                   y=results['channel_sensing']['channel_utilization'])
        plt.title('Channel Utilization')
        plt.xlabel('Channel')
        plt.ylabel('Usage Count')
        
        # Plot 3: Performance metrics
        plt.subplot(133)
        metrics = ['mae', 'rmse', 'r90']
        values = [results['localization'][m] for m in metrics]
        sns.barplot(x=metrics, y=values)
        plt.title('Performance Metrics')
        plt.ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_analysis.png'))
        plt.close()
    
    def evaluate(self):
        """Run complete evaluation"""
        print("Loading model...")
        self.load_model()
        
        print("Loading test data...")
        test_data = tf.data.load(self.test_data_path)
        X_test, y_test = test_data
        
        print("Evaluating localization performance...")
        localization_results = self.evaluate_localization(X_test, y_test)
        
        print("Evaluating trajectory prediction...")
        trajectory_results = self.evaluate_trajectory(X_test, y_test)
        
        print("Evaluating channel sensing...")
        channel_results = self.evaluate_channel_sensing()
        
        # Combine results
        results = {
            'localization': localization_results,
            'trajectory': trajectory_results,
            'channel_sensing': channel_results,
            'error_distribution': np.linalg.norm(
                self.model.predict(X_test) - y_test, 
                axis=1
            ).tolist()
        }
        
        # Save results
        with open(os.path.join(self.results_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate visualizations
        self.generate_visualizations(results)
        
        print("\nEvaluation Results:")
        print(f"MAE: {results['localization']['mae']:.4f}")
        print(f"R90: {results['localization']['r90']:.4f}")
        print(f"Combined Score: {results['localization']['combined_score']:.4f}")
        print(f"Channel Success Rate: {results['channel_sensing']['success_rate']:.4f}")
        
        return results

def main():
    # Configuration
    model_path = 'models/enhanced_localization_final.h5'
    test_data_path = 'data/test'
    
    # Run evaluation
    evaluator = ModelEvaluator(model_path, test_data_path)
    results = evaluator.evaluate()
    
    print(f"\nResults saved to: {evaluator.results_dir}")

if __name__ == "__main__":
    main() 