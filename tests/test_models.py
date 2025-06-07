import unittest
import tensorflow as tf
import numpy as np
import os
from src.models.basic_localization import create_basic_localization_model
from src.models.improved_localization import create_improved_localization_model
from src.utils.exceptions import ModelError

class TestModels(unittest.TestCase):
    def setUp(self):
        self.input_shape = (64, 64, 1)
        self.num_classes = 2
        self.batch_size = 32
        self.test_model_path = 'test_model.keras'

    def tearDown(self):
        # Clean up test files
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)

    def test_basic_model_creation(self):
        """Test basic model creation and output shape"""
        model = create_basic_localization_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Test model output shape
        test_input = tf.random.normal((self.batch_size, *self.input_shape))
        output = model(test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(model, tf.keras.Model)

    def test_improved_model_creation(self):
        """Test improved model creation and output shape"""
        model = create_improved_localization_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Test model output shape
        test_input = tf.random.normal((self.batch_size, *self.input_shape))
        output = model(test_input)
        
        self.assertEqual(output.shape, (self.batch_size, self.num_classes))
        self.assertIsInstance(model, tf.keras.Model)

    def test_model_training_step(self):
        """Test if model can perform a single training step"""
        model = create_basic_localization_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Create dummy data
        x = tf.random.normal((self.batch_size, *self.input_shape))
        y = tf.random.normal((self.batch_size, self.num_classes))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # Test training step
        history = model.fit(x, y, epochs=1, verbose=0)
        self.assertIn('loss', history.history)
        self.assertIn('mae', history.history)

    def test_model_save_load(self):
        """Test model saving and loading"""
        model = create_basic_localization_model(
            input_shape=self.input_shape,
            num_classes=self.num_classes
        )
        
        # Save model
        model.save(self.test_model_path)
        
        # Load model
        loaded_model = tf.keras.models.load_model(self.test_model_path)
        
        # Test loaded model
        test_input = tf.random.normal((self.batch_size, *self.input_shape))
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        
        np.testing.assert_array_almost_equal(
            original_output.numpy(),
            loaded_output.numpy()
        )

    def test_invalid_input_shape(self):
        """Test model creation with invalid input shape"""
        with self.assertRaises(ValueError):
            create_basic_localization_model(
                input_shape=(32, 32),  # Invalid shape
                num_classes=self.num_classes
            )

    def test_invalid_num_classes(self):
        """Test model creation with invalid number of classes"""
        with self.assertRaises(ValueError):
            create_basic_localization_model(
                input_shape=self.input_shape,
                num_classes=0  # Invalid number of classes
            )

if __name__ == '__main__':
    unittest.main() 