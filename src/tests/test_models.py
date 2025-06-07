import unittest
import numpy as np
import tensorflow as tf

from models.vanilla_localization import VanillaLocalizationModel
from models.trajectory_aware_localization import TrajectoryAwareLocalizationModel
from models.feature_selection_model import FeatureSelectionModel

class TestModels(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.batch_size = 4
        self.input_shape = (4, 8, 16, 2)  # (antennas, elements, frequencies, complex)
        self.sequence_length = 5
        
        # Create random input data
        self.test_input = np.random.randn(self.batch_size, *self.input_shape)
        self.test_labels = np.random.randn(self.batch_size, 2)  # x, y coordinates
        
        # Create sequence data for trajectory model
        self.sequence_input = np.random.randn(
            self.batch_size, self.sequence_length, *self.input_shape
        )
        self.sequence_labels = np.random.randn(self.batch_size, 2)
    
    def test_vanilla_model(self):
        """Test vanilla localization model."""
        model = VanillaLocalizationModel(
            input_shape=self.input_shape,
            num_classes=2
        )
        
        # Test forward pass
        predictions = model.predict(self.test_input)
        self.assertEqual(predictions.shape, (self.batch_size, 2))
        
        # Test training
        history = model.train(
            self.test_input,
            self.test_labels,
            epochs=2,
            batch_size=2
        )
        self.assertIsNotNone(history)
        
        # Test evaluation
        loss, mae = model.evaluate(self.test_input, self.test_labels)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(mae, float)
    
    def test_trajectory_model(self):
        """Test trajectory-aware localization model."""
        model = TrajectoryAwareLocalizationModel(
            input_shape=self.input_shape,
            sequence_length=self.sequence_length
        )
        
        # Test forward pass
        predictions = model.predict(self.sequence_input)
        self.assertEqual(predictions.shape, (self.batch_size, 2))
        
        # Test training
        history = model.train(
            self.sequence_input,
            self.sequence_labels,
            epochs=2,
            batch_size=2
        )
        self.assertIsNotNone(history)
        
        # Test evaluation
        loss, mae = model.evaluate(self.sequence_input, self.sequence_labels)
        self.assertIsInstance(loss, float)
        self.assertIsInstance(mae, float)
    
    def test_feature_selection_model(self):
        """Test feature selection model with grant-free random access."""
        model = FeatureSelectionModel(
            input_shape=self.input_shape,
            num_features=self.input_shape[-1]
        )
        
        # Test forward pass
        location_pred, priority_pred = model.predict(self.test_input)
        self.assertEqual(location_pred.shape, (self.batch_size, 2))
        self.assertEqual(priority_pred.shape, (self.batch_size, self.input_shape[-1]))
        
        # Test training
        history = model.train(
            self.test_input,
            self.test_labels,
            epochs=2,
            batch_size=2
        )
        self.assertIsNotNone(history)
        
        # Test evaluation
        results = model.evaluate(self.test_input, self.test_labels)
        self.assertIsInstance(results[0], float)  # Total loss
        self.assertIsInstance(results[1], float)  # Location loss
        self.assertIsInstance(results[2], float)  # Priority loss
        
        # Test feature selection
        selected_features = model.select_features(self.test_input)
        self.assertEqual(selected_features.shape, (self.batch_size, self.input_shape[-1]))
        self.assertTrue(np.all(np.isin(selected_features, [0, 1])))  # Binary selection

if __name__ == '__main__':
    unittest.main() 