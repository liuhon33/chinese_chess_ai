import unittest
from collections import deque
from unittest.mock import MagicMock, patch

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
        self.assertEqual(config.trainer.polling_interval, 15)
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
             patch.object(worker, "backup_play_data") as backup_play_data, \
             patch("cchess_alphazero.worker.optimize.sleep", side_effect=_StopLoop):
            with self.assertRaises(_StopLoop):
                worker.training()

        train_epoch.assert_called_once_with(config.trainer.epoch_to_checkpoint)
        publish_candidate_model.assert_called_once()
        backup_play_data.assert_called_once()


class EvaluatorLoopTest(unittest.TestCase):
    def test_evaluator_waits_then_promotes_candidate(self):
        config = Config("local_torch")
        service = evaluator.ContinuousEvaluator(config)
        result = {"score_ratio": 0.75, "win_rate": 75.0}

        with patch("cchess_alphazero.worker.evaluator.is_next_generation_model_fresh", side_effect=[False, True, False]), \
             patch("cchess_alphazero.worker.evaluator.evaluate_next_generation_model", return_value=result) as evaluate_once, \
             patch("cchess_alphazero.worker.evaluator.replace_best_model") as replace_best_model, \
             patch("cchess_alphazero.worker.evaluator.remove_ng_model") as remove_ng_model, \
             patch("cchess_alphazero.worker.evaluator.sleep", side_effect=[None, _StopLoop]):
            with self.assertRaises(_StopLoop):
                service.start()

        evaluate_once.assert_called_once_with(config)
        replace_best_model.assert_called_once_with(config)
        remove_ng_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()