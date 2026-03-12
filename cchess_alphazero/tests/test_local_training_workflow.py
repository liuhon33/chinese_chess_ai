import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import numpy as np

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.config import Config
from cchess_alphazero.worker import evaluator, optimize


class _StopLoop(Exception):
    pass


class LocalTorchConfigTest(unittest.TestCase):
    def test_local_torch_config_loads_expected_values(self):
        config = Config("local_torch")

        self.assertEqual(config.opts.backend, "torch")
        self.assertEqual(config.play.max_processes, 2)
        self.assertEqual(config.play.simulation_num_per_move, 200)
        self.assertEqual(config.play_data.nb_game_in_file, 5)
        self.assertEqual(config.play_data.max_file_num, 200)
        self.assertEqual(config.trainer.min_games_to_begin_learn, 8)
        self.assertEqual(config.trainer.batch_size, 256)
        self.assertEqual(config.trainer.load_step, 16)
        self.assertEqual(config.trainer.polling_interval, 90)
        self.assertEqual(config.eval.polling_interval, 15)
        self.assertEqual(config.eval.next_generation_replace_rate, 0.55)


class OptimizeWorkerLoopTest(unittest.TestCase):
    def test_optimizer_waits_after_publishing_candidate(self):
        config = Config("local_torch")
        worker = optimize.OptimizeWorker(config)
        worker.model = MagicMock()
        worker.dataset = (deque(), deque(), deque())

        fake_files = [f"play_{idx:02d}.json" for idx in range(config.trainer.min_games_to_begin_learn)]

        def fill_queue():
            worker.dataset[0].extend(range(config.trainer.batch_size))
            worker.dataset[1].extend(range(config.trainer.batch_size))
            worker.dataset[2].extend(range(config.trainer.batch_size))

        with patch.object(worker, "compile_model"), \
             patch.object(worker, "try_reload_model", return_value=False), \
             patch.object(worker, "has_pending_candidate", side_effect=[False, True]), \
             patch("cchess_alphazero.worker.optimize.get_game_data_filenames", return_value=fake_files), \
             patch.object(worker, "fill_queue", side_effect=fill_queue), \
             patch.object(worker, "train_epoch", return_value=32) as train_epoch, \
             patch.object(worker, "publish_candidate_model") as publish_candidate_model, \
             patch("cchess_alphazero.worker.optimize.save_training_state") as save_training_state, \
             patch.object(worker, "backup_play_data") as backup_play_data, \
             patch("cchess_alphazero.worker.optimize.sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                worker.training()

        train_epoch.assert_called_once_with(config.trainer.epoch_to_checkpoint)
        publish_candidate_model.assert_called_once()
        save_training_state.assert_called_once_with(config, 32)
        backup_play_data.assert_called_once()


class OptimizeDataExpansionTest(unittest.TestCase):
    def test_expanding_data_handles_multiple_games_per_file(self):
        data = [
            senv.INIT_STATE,
            ["7273", 1],
            senv.INIT_STATE,
            ["8081", -1],
        ]

        state_ary, policy_ary, value_ary = optimize.expanding_data(data, use_history=False)

        self.assertEqual(state_ary.shape[0], 2)
        self.assertEqual(policy_ary.shape[0], 2)
        np.testing.assert_array_equal(value_ary, np.asarray([1, -1], dtype=np.float32))


class EvaluatorLoopTest(unittest.TestCase):
    def test_evaluator_waits_then_promotes_candidate(self):
        config = Config("local_torch")
        service = evaluator.ContinuousEvaluator(config)
        result = {
            "score_ratio": 0.75,
            "win_rate": 75.0,
            "wins": 6,
            "losses": 2,
            "draws": 0,
            "elo": 190.85,
        }

        with patch("cchess_alphazero.worker.evaluator.is_next_generation_model_fresh", side_effect=[False, True, False]), \
             patch("cchess_alphazero.worker.evaluator.evaluate_next_generation_model", return_value=result) as evaluate_once, \
             patch("cchess_alphazero.worker.evaluator.record_eval_metrics", return_value={"total_self_play_games": 40, "elo": "190.85"}) as record_eval_metrics, \
             patch("cchess_alphazero.worker.evaluator.replace_best_model") as replace_best_model, \
             patch("cchess_alphazero.worker.evaluator.remove_ng_model") as remove_ng_model, \
             patch("cchess_alphazero.worker.evaluator.sleep", side_effect=[None, _StopLoop]):
            with self.assertRaises(_StopLoop):
                service.start()

        evaluate_once.assert_called_once_with(config)
        record_eval_metrics.assert_called_once_with(config, result, "promote_candidate")
        replace_best_model.assert_called_once_with(config)
        remove_ng_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
