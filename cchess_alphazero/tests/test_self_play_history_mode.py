import os
import shutil
import time
import unittest

from cchess_alphazero.config import Config
from cchess_alphazero.worker.self_play import _model_uses_history, load_model


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


TORCH_AVAILABLE = _torch_available()


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed or importable")
class SelfPlayHistoryModeTest(unittest.TestCase):
    def setUp(self):
        self.config = Config("local_torch")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.config.opts.new = True
        root = os.path.join(self.config.resource.model_dir, "test_artifacts", f"selfplay_{os.getpid()}_{time.time_ns()}")
        os.makedirs(root, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        self.config.resource.update_paths(data_dir=root)
        self.config.resource.create_directories()

    def test_fresh_start_local_torch_keeps_history_disabled_for_14_plane_model(self):
        model, use_history = load_model(self.config)

        self.assertFalse(use_history)
        self.assertFalse(_model_uses_history(model))


if __name__ == "__main__":
    unittest.main()
