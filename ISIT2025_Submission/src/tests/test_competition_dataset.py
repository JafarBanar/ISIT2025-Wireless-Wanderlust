import unittest
import tensorflow as tf
import numpy as np
import glob
import os
from src.data_loader import CSIDataLoader

class TestCompetitionDataset(unittest.TestCase):
    def setUp(self):
        # Use a small sample TFRecord file from the competition dataset
        data_dir = os.environ.get('COMPETITION_DATA_DIR', 'data/competition/test_tfrecord')
        tfrecord_files = glob.glob(os.path.join(data_dir, '*.tfrecord'))
        if not tfrecord_files:
            self.skipTest('No TFRecord files found in competition dataset directory.')
        self.tfrecord_file = tfrecord_files[0]
        self.loader = CSIDataLoader(self.tfrecord_file, batch_size=1)

    def test_csi_shape(self):
        labeled_ds, _ = self.loader.load_dataset(is_training=False)
        for batch in labeled_ds.take(1):
            csi = batch['csi'].numpy()
            self.assertEqual(csi.shape, (1, 4, 8, 16, 2),
                f'CSI shape is {csi.shape}, expected (1, 4, 8, 16, 2)')

    def test_position_shape(self):
        labeled_ds, _ = self.loader.load_dataset(is_training=False)
        for batch in labeled_ds.take(1):
            pos = batch['position'].numpy()
            self.assertEqual(pos.shape, (1, 2),
                f'Position shape is {pos.shape}, expected (1, 2)')

    def test_required_fields(self):
        labeled_ds, _ = self.loader.load_dataset(is_training=False)
        for batch in labeled_ds.take(1):
            for field in ['csi', 'position', 'timestamp', 'cfo', 'snr', 'gt_interp_age']:
                self.assertIn(field, batch, f'Missing required field: {field}')

if __name__ == '__main__':
    unittest.main() 