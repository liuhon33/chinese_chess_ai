import os
import unittest

import numpy as np

from cchess_alphazero.agent.api import _prediction_value_to_float
from cchess_alphazero.agent.backends.torch_backend import TorchModelBackend
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


TORCH_AVAILABLE = _torch_available()


class PredictionValueConversionTest(unittest.TestCase):
    def test_prediction_value_to_float_accepts_batch_column_output(self):
        self.assertEqual(_prediction_value_to_float(np.asarray([0.25], dtype=np.float32)), 0.25)
        self.assertEqual(_prediction_value_to_float(np.asarray([[0.5]], dtype=np.float32)), 0.5)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed or importable")
class TorchModelBackendTest(unittest.TestCase):
    def setUp(self):
        self.config = Config("mini")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.artifact_dir = os.path.join(self.config.resource.model_dir, "test_artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def test_predict_batch_shape_and_device(self):
        backend = TorchModelBackend(self.config)
        backend.build_model()

        batch = np.zeros((2, self.config.model.input_depth, 10, 9), dtype=np.float32)
        policy, value = backend.predict_batch(batch)

        self.assertEqual(policy.shape, (2, len(ActionLabelsRed)))
        self.assertEqual(value.shape, (2, 1))
        self.assertEqual(next(backend.model.parameters()).device.type, "cpu")
        self.assertEqual(str(backend.device), "cpu")

    def test_predict_batch_rejects_bad_shape(self):
        backend = TorchModelBackend(self.config)
        backend.build_model()

        with self.assertRaises(ValueError):
            backend.predict_batch(np.zeros((1, 10, 9), dtype=np.float32))

    def test_load_existing_keras_weights_smoke(self):
        backend = TorchModelBackend(self.config)
        config_path = self.config.resource.model_best_config_path
        weight_path = self.config.resource.model_best_weight_path
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            self.skipTest("sample model files are not available")

        self.assertTrue(backend.load_model(config_path, weight_path))
        batch = np.zeros((1, 14, 10, 9), dtype=np.float32)
        policy, value = backend.predict_batch(batch)

        self.assertEqual(policy.shape, (1, len(ActionLabelsRed)))
        self.assertEqual(value.shape, (1, 1))
        self.assertTrue(np.allclose(policy.sum(axis=1), 1.0, atol=1e-4))

    def test_save_and_reload_torch_checkpoint(self):
        backend = TorchModelBackend(self.config)
        backend.build_model()
        batch = np.zeros((1, self.config.model.input_depth, 10, 9), dtype=np.float32)
        policy_before, value_before = backend.predict_batch(batch)

        config_path = os.path.join(self.artifact_dir, "backend_test_config.json")
        weight_path = os.path.join(self.artifact_dir, "backend_test_weight.h5")
        try:
            backend.save_model(config_path, weight_path)

            clone = TorchModelBackend(self.config)
            self.assertTrue(clone.load_model(config_path, weight_path))
            policy_after, value_after = clone.predict_batch(batch)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)
            if os.path.exists(weight_path):
                os.remove(weight_path)

        np.testing.assert_allclose(policy_before, policy_after, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(value_before, value_after, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
