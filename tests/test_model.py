import unittest
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys
import os

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.basic_localization import BasicLocalizationModel
from src.utils.metrics import R90Metric

class TestBasicLocalizationModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all tests."""
        cls.model = BasicLocalizationModel()
        cls.batch_size = 32
        cls.input_shape = (8, 16)  # 8 elements per array, 16 frequency bands
        
    def setUp(self):
        """Set up test fixtures that are run before each test."""
        # Generate random test data
        self.test_input = np.random.randn(self.batch_size, *self.input_shape)
        self.test_labels = np.random.randn(self.batch_size, 2)  # x, y coordinates
        
    def test_model_creation(self):
        """Test that the model can be created with correct architecture."""
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(self.model.count_params() > 0, True)
        
    def test_model_output_shape(self):
        """Test that the model output has the correct shape."""
        output = self.model(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, 2))
        
    def test_model_training(self):
        """Test that the model can be trained."""
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', R90Metric()]
        )
        
        # Train for one epoch
        history = self.model.fit(
            self.test_input,
            self.test_labels,
            epochs=1,
            batch_size=self.batch_size,
            validation_split=0.2
        )
        
        self.assertIn('loss', history.history)
        self.assertIn('mae', history.history)
        self.assertIn('r90_metric', history.history)
        
    def test_model_prediction(self):
        """Test that the model can make predictions."""
        predictions = self.model.predict(self.test_input)
        self.assertEqual(predictions.shape, (self.batch_size, 2))
        
    def test_edge_cases(self):
        """Test model behavior with edge cases."""
        # Test with zero input
        zero_input = np.zeros((self.batch_size, *self.input_shape))
        zero_output = self.model.predict(zero_input)
        self.assertEqual(zero_output.shape, (self.batch_size, 2))
        
        # Test with large input
        large_input = np.random.randn(self.batch_size, *self.input_shape) * 1000
        large_output = self.model.predict(large_input)
        self.assertEqual(large_output.shape, (self.batch_size, 2))
        
        # Test with NaN input
        nan_input = np.full((self.batch_size, *self.input_shape), np.nan)
        with self.assertRaises(Exception):
            self.model.predict(nan_input)
            
    def test_model_saving_loading(self):
        """Test that the model can be saved and loaded."""
        # Save model
        save_path = Path('test_model.h5')
        self.model.save(save_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(save_path)
        
        # Compare predictions
        original_pred = self.model.predict(self.test_input)
        loaded_pred = loaded_model.predict(self.test_input)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        
        # Clean up
        save_path.unlink()
        
    def test_performance_metrics(self):
        """Test model performance metrics."""
        # Compile model with metrics
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', R90Metric()]
        )
        
        # Evaluate model
        metrics = self.model.evaluate(
            self.test_input,
            self.test_labels,
            batch_size=self.batch_size
        )
        
        self.assertIsInstance(metrics, list)
        self.assertEqual(len(metrics), 3)  # loss, mae, r90_metric
        
    def test_batch_processing(self):
        """Test model behavior with different batch sizes."""
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            input_data = np.random.randn(batch_size, *self.input_shape)
            output = self.model.predict(input_data)
            self.assertEqual(output.shape, (batch_size, 2))
            
    def test_model_regularization(self):
        """Test that model regularization is working."""
        # Check if model has regularization layers
        has_dropout = any('dropout' in layer.name.lower() 
                         for layer in self.model.layers)
        has_l2 = any('kernel_regularizer' in layer.get_config() 
                    for layer in self.model.layers)
        
        self.assertTrue(has_dropout or has_l2, 
                       "Model should have regularization layers")
        
    def test_input_validation(self):
        """Test model input validation."""
        # Test with wrong input shape
        wrong_shape_input = np.random.randn(self.batch_size, 4, 8)
        with self.assertRaises(Exception):
            self.model.predict(wrong_shape_input)
            
        # Test with wrong batch size
        wrong_batch_input = np.random.randn(1, *self.input_shape)
        with self.assertRaises(Exception):
            self.model.predict(wrong_batch_input)

if __name__ == '__main__':
    unittest.main() 