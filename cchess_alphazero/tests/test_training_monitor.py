import csv
import json
import os
import shutil
import unittest

from cchess_alphazero.config import Config
from cchess_alphazero.lib.training_monitor import count_cumulative_self_play_games, record_eval_metrics, save_training_state

INIT_STATE = "rkemsmekr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RKEMSMEKR"


class TrainingMonitorTest(unittest.TestCase):
    def make_root_dir(self):
        root_dir = os.path.join(os.getcwd(), "tmp_test_artifacts", self.id().split(".")[-1])
        shutil.rmtree(root_dir, ignore_errors=True)
        os.makedirs(root_dir, exist_ok=True)
        self.addCleanup(shutil.rmtree, root_dir, True)
        return root_dir

    def make_config(self, root_dir):
        project_dir = os.path.join(root_dir, "project")
        data_dir = os.path.join(root_dir, "data")
        config = Config("local_torch")
        config.resource.update_paths(project_dir=project_dir, data_dir=data_dir)
        config.resource.create_directories()
        return config

    def write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wt", encoding="utf-8") as output:
            json.dump(payload, output)

    def test_count_cumulative_self_play_games_uses_recorded_games(self):
        config = self.make_config(self.make_root_dir())
        trained_dir = os.path.join(config.resource.data_dir, "trained")
        os.makedirs(trained_dir, exist_ok=True)

        self.write_json(
            os.path.join(config.resource.play_data_dir, "play_a.json"),
            [INIT_STATE, ["0001", 1], ["0002", -1], INIT_STATE, ["0003", 1]],
        )
        self.write_json(
            os.path.join(trained_dir, "play_b.json"),
            [INIT_STATE, ["1111", 0], INIT_STATE, ["2222", 0]],
        )
        self.write_json(
            os.path.join(config.resource.play_data_dir, "play_eval.json"),
            ["abcd1234", "ffff0000", INIT_STATE, ["3333", 1]],
        )

        total_games = count_cumulative_self_play_games(config)
        cached_total_games = count_cumulative_self_play_games(config)

        self.assertEqual(total_games, 4)
        self.assertEqual(cached_total_games, 4)
        self.assertTrue(os.path.exists(config.resource.self_play_game_cache_path))

    def test_record_eval_metrics_writes_csv_and_png(self):
        config = self.make_config(self.make_root_dir())
        self.write_json(
            os.path.join(config.resource.play_data_dir, "play_a.json"),
            [INIT_STATE, ["0001", 1], INIT_STATE, ["0002", -1], INIT_STATE, ["0003", 1]],
        )
        os.makedirs(os.path.dirname(config.resource.model_best_weight_path), exist_ok=True)
        with open(config.resource.model_best_weight_path, "wb") as best_weight:
            best_weight.write(b"best-weight")
        with open(config.resource.next_generation_weight_path, "wb") as candidate_weight:
            candidate_weight.write(b"candidate-weight")

        save_training_state(config, 32)
        row = record_eval_metrics(
            config,
            {"wins": 3, "losses": 1, "draws": 0, "elo": 190.85},
            "promote_candidate",
        )

        self.assertEqual(row["total_self_play_games"], 3)
        self.assertEqual(row["total_step"], 32)
        self.assertTrue(os.path.exists(config.resource.elo_history_path))
        self.assertTrue(os.path.exists(config.resource.elo_plot_path))
        self.assertGreater(os.path.getsize(config.resource.elo_plot_path), 0)

        with open(config.resource.elo_history_path, "rt", newline="", encoding="utf-8") as history_file:
            history_rows = list(csv.DictReader(history_file))

        self.assertEqual(len(history_rows), 1)
        self.assertEqual(history_rows[0]["total_self_play_games"], "3")
        self.assertEqual(history_rows[0]["total_step"], "32")
        self.assertEqual(history_rows[0]["wins"], "3")
        self.assertEqual(history_rows[0]["losses"], "1")
        self.assertEqual(history_rows[0]["draws"], "0")
        self.assertEqual(history_rows[0]["promotion_decision"], "promote_candidate")


if __name__ == "__main__":
    unittest.main()