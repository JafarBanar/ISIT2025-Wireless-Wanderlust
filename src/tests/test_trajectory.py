import unittest
import numpy as np
import tensorflow as tf
import tempfile
import os
from src.isit2025.models.trajectory_model import (
    TemporalAttention,
    TrajectoryAwareModel
)
from src.isit2025.data_processing.trajectory_data import (
    TrajectoryDataGenerator,
    TrajectoryTFRecordHandler
)

class TestTrajectoryModel(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.batch_size = 4
        self.sequence_length = 5
        self.feature_dim = 32
        self.position_dim = 2
        
        # Create sample inputs
        self.csi_features = tf.random.normal(
            (self.batch_size, self.sequence_length, self.feature_dim)
        )
        self.prev_positions = tf.random.normal(
            (self.batch_size, self.sequence_length - 1, self.position_dim)
        )
    
    def test_temporal_attention(self):
        attention = TemporalAttention(units=64)
        
        # Test attention mechanism
        query = tf.random.normal((self.batch_size, 128))
        values = tf.random.normal((self.batch_size, 10, 128))
        
        context, weights = attention(query, values)
        
        # Check output shapes
        self.assertEqual(context.shape, (self.batch_size, 128))
        self.assertEqual(weights.shape, (self.batch_size, 10, 1))
        
        # Check attention weights sum to 1
        weights_sum = tf.reduce_sum(weights, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(weights_sum - 1.0) < 1e-6))
    
    def test_trajectory_model(self):
        model = TrajectoryAwareModel(
            sequence_length=self.sequence_length,
            lstm_units=64,
            attention_units=32
        )
        
        # Compile model
        model.compile()
        
        # Test forward pass
        inputs = {
            'csi_features': self.csi_features,
            'prev_positions': self.prev_positions
        }
        
        output = model(inputs, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, 2))
        
        # Test training step
        y = tf.random.normal((self.batch_size, 2))
        loss = model.train_on_batch(inputs, y)
        
        # Verify loss is finite
        self.assertTrue(np.isfinite(loss))
    
    def test_model_save_load(self):
        model = TrajectoryAwareModel()
        
        # Save and load model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'model.keras')
            model.save(model_path)
            loaded_model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'TrajectoryAwareModel': TrajectoryAwareModel,
                    'TemporalAttention': TemporalAttention
                }
            )
        
        # Test loaded model
        inputs = {
            'csi_features': self.csi_features,
            'prev_positions': self.prev_positions
        }
        
        original_output = model(inputs)
        loaded_output = loaded_model(inputs)
        
        # Check outputs match
        np.testing.assert_allclose(
            original_output.numpy(),
            loaded_output.numpy(),
            rtol=1e-5
        )

class TestTrajectoryData(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.n_samples = 100
        self.feature_dim = 32
        self.sequence_length = 5
        
        self.csi_features = np.random.normal(
            size=(self.n_samples, self.feature_dim)
        )
        self.positions = np.random.normal(
            size=(self.n_samples, 2)
        )
    
    def test_data_generator(self):
        generator = TrajectoryDataGenerator(
            self.csi_features,
            self.positions,
            sequence_length=self.sequence_length,
            batch_size=16
        )
        
        # Get datasets
        train_ds = generator.get_train_dataset()
        val_ds = generator.get_val_dataset()
        
        # Check dataset structure
        for x, y in train_ds.take(1):
            self.assertIn('csi_features', x)
            self.assertIn('prev_positions', x)
            self.assertEqual(x['csi_features'].shape[1], self.sequence_length)
            self.assertEqual(x['prev_positions'].shape[1], self.sequence_length - 1)
            self.assertEqual(y.shape[1], 2)
    
    def test_tfrecord_handler(self):
        handler = TrajectoryTFRecordHandler(sequence_length=self.sequence_length)
        
        # Test write and read
        with tempfile.TemporaryDirectory() as temp_dir:
            tfrecord_path = os.path.join(temp_dir, 'test.tfrecord')
            
            # Write data
            handler.write_tfrecord(
                self.csi_features,
                self.positions,
                tfrecord_path
            )
            
            # Load dataset
            dataset = handler.load_dataset(tfrecord_path, batch_size=16)
            
            # Check dataset structure
            for x, y in dataset.take(1):
                self.assertIn('csi_features', x)
                self.assertIn('prev_positions', x)
                self.assertEqual(x['csi_features'].shape[1], self.sequence_length)
                self.assertEqual(x['prev_positions'].shape[1], self.sequence_length - 1)
                self.assertEqual(y.shape[1], 2)
    
    def test_invalid_inputs(self):
        # Test mismatched feature and position lengths
        with self.assertRaises(ValueError):
            TrajectoryDataGenerator(
                self.csi_features[:-1],  # One less feature
                self.positions,
                sequence_length=self.sequence_length
            )
        
        # Test invalid sequence length
        with self.assertRaises(ValueError):
            TrajectoryDataGenerator(
                self.csi_features,
                self.positions,
                sequence_length=len(self.csi_features) + 1  # Too long
            )

if __name__ == '__main__':
    unittest.main() 