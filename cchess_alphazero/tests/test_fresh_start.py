import os
import shutil
import time
import unittest

import numpy as np

from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed
from cchess_alphazero.lib.model_helper import is_next_generation_model_fresh, load_best_model_weight, save_as_next_generation_model
from cchess_alphazero.manager import create_parser, setup
from cchess_alphazero.worker import evaluator


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


TORCH_AVAILABLE = _torch_available()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed or importable")
class FreshStartModeTest(unittest.TestCase):
    def setUp(self):
        self.config = Config("mini")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.config.opts.new = True
        artifact_root = os.path.join(self.config.resource.model_dir, "test_artifacts")
        os.makedirs(artifact_root, exist_ok=True)
        self.artifact_dir = os.path.join(artifact_root, f"fresh_start_{os.getpid()}_{time.time_ns()}")
        os.makedirs(self.artifact_dir, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.artifact_dir, ignore_errors=True))
        self._configure_resource_paths(self.artifact_dir)

    def _configure_resource_paths(self, root_dir):
        rc = self.config.resource
        rc.update_paths(data_dir=root_dir)
        os.makedirs(rc.model_dir, exist_ok=True)
        os.makedirs(rc.next_generation_model_dir, exist_ok=True)

    def test_new_mode_initializes_and_saves_best_model(self):
        model = evaluator.load_best_model(self.config)

        self.assertTrue(os.path.exists(self.config.resource.model_best_config_path))
        self.assertTrue(os.path.exists(self.config.resource.model_best_weight_path))

        batch = np.zeros((1, self.config.model.input_depth, 10, 9), dtype=np.float32)
        policy, value = model.predict_on_batch(batch)

        self.assertEqual(policy.shape, (1, len(ActionLabelsRed)))
        self.assertEqual(value.shape, (1, 1))
        self.assertEqual(model.digest, model.fetch_digest(self.config.resource.model_best_weight_path))

    def test_new_mode_rejects_stale_next_generation_checkpoint(self):
        next_generation_model = CChessModel(self.config)
        next_generation_model.build()
        save_as_next_generation_model(next_generation_model)

        stale_time = time.time() - 3600
        os.utime(self.config.resource.next_generation_config_path, (stale_time, stale_time))
        os.utime(self.config.resource.next_generation_weight_path, (stale_time, stale_time))

        evaluator.load_best_model(self.config)

        self.assertFalse(is_next_generation_model_fresh(self.config))
        self.assertIsNone(evaluator.load_next_generation_model(self.config))

    def test_custom_data_dir_can_resume_best_model(self):
        evaluator.load_best_model(self.config)

        resume_config = Config("mini")
        resume_config.opts.backend = "torch"
        resume_config.opts.device_list = "cpu"
        args = create_parser().parse_args(["opt", "--data-dir", self.artifact_dir])
        setup(resume_config, args)

        model = CChessModel(resume_config)
        self.assertTrue(load_best_model_weight(model))
        self.assertEqual(resume_config.resource.data_dir, os.path.abspath(self.artifact_dir))


if __name__ == "__main__":
    unittest.main()







