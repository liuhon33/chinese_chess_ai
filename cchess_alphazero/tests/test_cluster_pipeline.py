import json
import os
import shutil
import unittest
from uuid import uuid4

from cchess_alphazero.config import Config
from cchess_alphazero.lib.cluster_helper import build_cluster_play_data_path, publish_model_pair_atomically
from cchess_alphazero.lib.data_helper import get_game_data_filenames
from cchess_alphazero.lib.model_helper import next_generation_model_exists
from cchess_alphazero.manager import create_parser, setup
from cchess_alphazero.worker.optimize import OptimizeWorker


def make_workspace_test_dir(prefix):
    root = os.path.join(os.getcwd(), ".tmp_testdata")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"{prefix}_{uuid4().hex[:8]}")
    os.makedirs(path, exist_ok=True)
    return path


class ClusterCliTest(unittest.TestCase):
    def test_cluster_flags_default_to_current_behavior(self):
        parser = create_parser()
        args = parser.parse_args(["self", "--type", "local_torch", "--data-dir", "mydata"])
        config = Config("local_torch")
        setup(config, args)

        self.assertFalse(config.cluster.enabled)
        self.assertIsNone(config.cluster.worker_id)
        self.assertIsNone(config.cluster.auto_reload_best)
        self.assertIsNone(config.cluster.reload_best_interval)
        self.assertFalse(config.cluster.safe_write_play_data)
        self.assertFalse(config.cluster.archive_consumed_data)
        self.assertIsNone(config.cluster.optimizer_poll_interval)
        self.assertIsNone(config.cluster.evaluator_poll_interval)


class ClusterHelperTest(unittest.TestCase):
    def test_cluster_play_data_path_is_unique(self):
        config = Config("local_torch")
        config.cluster.enabled = True
        config.cluster.worker_id = "7"
        data_dir = make_workspace_test_dir("cluster_paths")

        try:
            config.resource.update_paths(data_dir=data_dir)
            config.resource.create_directories()
            first = build_cluster_play_data_path(config, pid=111)
            second = build_cluster_play_data_path(config, pid=111)
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(".json"))
        self.assertIn("_7_", first)

    def test_cluster_candidate_publish_requires_ready_marker(self):
        config = Config("local_torch")
        config.cluster.enabled = True
        data_dir = make_workspace_test_dir("cluster_candidate")

        try:
            config.resource.update_paths(data_dir=data_dir)
            config.resource.create_directories()

            def save_pair(config_path, weight_path):
                with open(config_path, "wt") as config_file:
                    json.dump({"backend": "torch", "network_spec": {}}, config_file)
                with open(weight_path, "wb") as weight_file:
                    weight_file.write(b"weights")

            publish_model_pair_atomically(
                save_pair,
                config.resource.next_generation_config_path,
                config.resource.next_generation_weight_path,
                ready_path=config.resource.next_generation_ready_path,
            )

            self.assertTrue(next_generation_model_exists(config))
            self.assertTrue(os.path.exists(config.resource.next_generation_ready_path))
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


class ClusterOptimizerClaimTest(unittest.TestCase):
    def test_optimizer_claims_and_archives_cluster_files(self):
        config = Config("local_torch")
        config.cluster.enabled = True
        config.cluster.worker_id = "3"
        config.cluster.archive_consumed_data = True
        data_dir = make_workspace_test_dir("cluster_claims")

        try:
            config.resource.update_paths(data_dir=data_dir)
            config.resource.create_directories()
            play_path = os.path.join(config.resource.play_data_dir, "play_test.json")
            with open(play_path, "wt") as play_file:
                json.dump(["state", ["7273", 1]], play_file)
            stale_time = os.path.getmtime(play_path) - 10
            os.utime(play_path, (stale_time, stale_time))

            worker = OptimizeWorker(config)
            files = worker.available_play_data_files()
            claimed = worker.claim_selected_files(files)
            worker.finalize_claimed_files(claimed)

            self.assertEqual(get_game_data_filenames(config.resource), [])
            archived = os.listdir(config.resource.trained_data_dir)
            self.assertEqual(len(archived), 1)
            self.assertTrue(archived[0].startswith("play_test"))
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
