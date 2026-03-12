import os
import unittest

import numpy as np

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


TORCH_AVAILABLE = _torch_available()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed or importable")
class TorchPipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        self.config = Config("mini")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.artifact_dir = os.path.join(self.config.resource.model_dir, "test_artifacts")
        os.makedirs(self.artifact_dir, exist_ok=True)

    def test_model_init_infer_train_save_reload(self):
        model = CChessModel(self.config)
        model.build()

        batch = np.zeros((2, self.config.model.input_depth, 10, 9), dtype=np.float32)
        batch[0, 0, 0, 0] = 1.0
        batch[1, 1, 1, 1] = 1.0
        policy_before, value_before = model.predict_on_batch(batch)
        self.assertEqual(policy_before.shape, (2, len(ActionLabelsRed)))
        self.assertEqual(value_before.shape, (2, 1))

        target_policy = np.zeros((2, len(ActionLabelsRed)), dtype=np.float32)
        target_policy[0, 0] = 1.0
        target_policy[1, 1] = 1.0
        target_value = np.asarray([[1.0], [-1.0]], dtype=np.float32)

        model.configure_training(
            optimizer_name="sgd",
            learning_rate=0.01,
            momentum=self.config.trainer.momentum,
            loss_weights=self.config.trainer.loss_weights,
        )
        metrics = model.train(
            state_ary=batch,
            policy_ary=target_policy,
            value_ary=target_value,
            batch_size=2,
            epochs=1,
            shuffle=False,
            validation_split=0.0,
        )
        self.assertIn("train_loss", metrics)

        config_path = os.path.join(self.artifact_dir, "pipeline_test_config.json")
        weight_path = os.path.join(self.artifact_dir, "pipeline_test_weight.h5")
        try:
            model.save(config_path, weight_path)
            self.assertTrue(os.path.exists(config_path))
            self.assertTrue(os.path.exists(weight_path))

            clone = CChessModel(self.config)
            self.assertTrue(clone.load(config_path, weight_path))
            policy_after, value_after = clone.predict_on_batch(batch)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)
            if os.path.exists(weight_path):
                os.remove(weight_path)

        self.assertEqual(policy_after.shape, (2, len(ActionLabelsRed)))
        self.assertEqual(value_after.shape, (2, 1))
        self.assertTrue(np.allclose(policy_after.sum(axis=1), 1.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
