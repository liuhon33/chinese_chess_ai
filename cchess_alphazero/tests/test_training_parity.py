"""Regressions against the Keras 2.0.8 training contract (no TF dependency)."""
import json
import math
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import torch
from torch import nn

from cchess_alphazero.agent.backends.torch_backend import TorchModelBackend
from cchess_alphazero.config import Config
import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.lib.model_helper import build_fresh_best_model, promote_next_generation_to_best
from cchess_alphazero.lib.training_monitor import save_training_state
from cchess_alphazero.manager import create_parser, setup
from cchess_alphazero.worker import evaluator, optimize
from cchess_alphazero.worker import self_play


class TrainingParityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.config = Config("mini")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.config.model.cnn_filter_num = 8
        self.config.model.res_layer_num = 1
        self.config.model.value_fc_size = 16
        self.root = Path(".tmp_testdata") / ("training_parity_" + uuid4().hex)
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.config.resource.update_paths(data_dir=str(self.root))
        self.config.resource.create_directories()

    def backend(self):
        backend = TorchModelBackend(self.config)
        backend.build_model()
        return backend

    def test_candidate_publication_does_not_reset_training(self):
        for cluster in (False, True):
            with self.subTest(cluster=cluster):
                self.config.cluster.enabled = cluster
                worker = optimize.OptimizeWorker(self.config)
                worker.model = worker.load_model()
                worker.compile_model()
                backend = worker.model.backend
                weights = backend.model.policy_fc.weight
                weights.grad = torch.ones_like(weights)
                backend._optimizer.step()
                trained_weights = weights.detach().clone()
                optimizer = backend._optimizer
                worker.publish_candidate_model()
                self.assertFalse(worker.try_reload_model(), "publishing is not a change to best")
                self.assertIs(backend._optimizer, optimizer)
                torch.testing.assert_close(backend.model.policy_fc.weight, trained_weights)
                self.assertIn(weights, optimizer.state)
                # Rejection must leave training intact; a real promotion must be detected.
                self.assertFalse(worker.try_reload_model())
                promote_next_generation_to_best(self.config)
                self.assertTrue(worker.try_reload_model())

    def test_fresh_weights_match_glorot_uniform_and_zero_bias(self):
        for module in self.backend().model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight)
                expected_variance = 2.0 / (fan_in + fan_out)
                actual = float(module.weight.detach().var(unbiased=False))
                if module.weight.numel() >= 256:
                    self.assertAlmostEqual(actual / expected_variance, 1.0, delta=0.18)
                self.assertLessEqual(float(module.weight.detach().abs().max()), math.sqrt(3 * expected_variance))
                if module.bias is not None:
                    torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))

    def test_batchnorm_running_variance_matches_tf_population_moments(self):
        bn = self.backend().model.input_bn
        x = torch.arange(2 * 8 * 10 * 9, dtype=torch.float32).reshape(2, 8, 10, 9) / 100
        mean = x.mean((0, 2, 3))
        variance = x.var((0, 2, 3), unbiased=False)
        bn.train()
        actual = bn(x)
        expected = (x - mean[None, :, None, None]) / (variance[None, :, None, None] + 1e-3).sqrt()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(bn.running_mean, 0.01 * mean)
        torch.testing.assert_close(bn.running_var, 0.99 + 0.01 * variance)
        before = bn.running_var.clone()
        bn.eval()
        bn(x)
        torch.testing.assert_close(bn.running_var, before)

    def test_validation_split_uses_tail_before_training_shuffle(self):
        backend = self.backend()
        for shuffle in (False, True):
            train, val = backend._split_indices(51, 0.02, shuffle)
            np.testing.assert_array_equal(train, np.arange(49))
            np.testing.assert_array_equal(val, np.arange(49, 51))

    def test_sgd_matches_keras_velocity_across_learning_rate_change(self):
        backend = self.backend()
        backend.configure_training("sgd", 0.1, momentum=0.9)
        param = backend.model.value_fc2.bias
        initial = param.detach().clone()
        velocity = torch.zeros_like(param)
        for lr in (0.1, 0.1, 0.01, 0.01):
            backend.set_learning_rate(lr)
            param.grad = torch.ones_like(param)
            velocity = 0.9 * velocity - lr * param.grad
            initial += velocity
            backend._optimizer.step()
            torch.testing.assert_close(param, initial)

    def test_zero_l2_and_legacy_flatten_survive_checkpoint(self):
        self.config.model.l2_reg = 0.0
        backend = self.backend()
        config_path = str(self.root / "model.json")
        weight_path = str(self.root / "model.h5")
        backend.save_model(config_path, weight_path)
        layers = json.loads(Path(config_path).read_text())["layers"]
        for layer in layers:
            if layer["class_name"] == "Flatten":
                self.assertNotIn("data_format", layer["config"])
        self.config.model.l2_reg = 0.1
        clone = TorchModelBackend(self.config)
        clone.load_model(config_path, weight_path)
        self.assertEqual(clone._spec.l2_reg, 0.0)

    def test_new_selfplay_can_reload_a_later_promoted_best(self):
        self.config.opts.new = True
        model = build_fresh_best_model(CChessModel(self.config))
        initial_digest = model.digest
        worker = optimize.OptimizeWorker(self.config)
        worker.model = worker.load_model()
        with torch.no_grad():
            worker.model.model.value_fc2.bias.add_(0.5)
        worker.publish_candidate_model()
        promote_next_generation_to_best(self.config)
        CChessModelAPI(self.config, model).try_reload_model()
        self.assertNotEqual(model.digest, initial_digest)
        self.assertEqual(model.digest, model.fetch_digest(self.config.resource.model_best_weight_path))

    def test_new_evaluator_initializes_best_only_once(self):
        self.config.opts.new = True
        first = evaluator.load_best_model(self.config)
        second = evaluator.load_best_model(self.config)
        self.assertEqual(first.digest, second.digest)

    def test_resume_uses_saved_steps_but_explicit_override_wins(self):
        save_training_state(self.config, 150010)
        parser = create_parser()
        for extra, expected in (([], 150010), (["--total-step", "0"], 0), (["--new"], 0)):
            config = Config("mini")
            args = parser.parse_args(["opt", "--data-dir", str(self.root)] + extra)
            with patch("cchess_alphazero.manager.setup_logger"):
                setup(config, args)
            self.assertEqual(config.trainer.start_total_steps, expected)

    def test_step_count_includes_partial_batch_excludes_validation(self):
        worker = optimize.OptimizeWorker(self.config)
        worker.model = MagicMock()
        self.config.trainer.batch_size = 32
        worker.dataset = ([0] * 67, [0] * 67, [0] * 67)
        # 65 training rows, three batches per epoch, two epochs.
        self.assertEqual(worker.train_epoch(2), 6)

    def test_evaluation_honors_configured_search_budget(self):
        self.config.eval.update_play_config(self.config.play)
        budget = self.config.eval.simulation_num_per_move
        worker = evaluator.EvaluateWorker(self.config, [object()], [object()], pid=0)
        with patch.object(evaluator, "CChessPlayer") as player:
            player.return_value.action.return_value = (None, [])
            worker.start_game(0)
        self.assertEqual(self.config.play.simulation_num_per_move, budget)

    def test_slow_selfplay_flushes_a_completed_game_before_five_games(self):
        self.config.play_data.nb_game_in_file = 5
        self.config.play_data.max_buffer_seconds = 300
        self.config.cluster.enabled = True
        self.config.cluster.safe_write_play_data = True
        worker = self_play.SelfPlayWorker(self.config)
        worker.last_flush_time = 0.0
        data = [senv.INIT_STATE, ["8081", 1]]
        with patch.object(self_play, "time", return_value=301.0):
            worker.save_play_data(1, data)
        files = list(Path(self.config.resource.play_data_dir).glob("play_*.json"))
        self.assertEqual(len(files), 1)
        self.assertEqual(json.loads(files[0].read_text()), data)
        self.assertEqual(worker.buffer, [])

    def test_evaluation_scores_candidate_correctly_in_both_colors(self):
        self.config.eval.game_num = 2
        worker = evaluator.EvaluateWorker(self.config, pid=0)
        with patch.object(worker, "start_game", side_effect=[(-1, 20), (1, 20)]), patch.object(evaluator, "sleep"):
            score, red_win, red_draw, red_loss, black_win, black_draw, black_loss = worker.start()
        self.assertEqual((score, red_win, black_win), (2, 1, 1))
        self.assertEqual((red_draw, red_loss, black_draw, black_loss), (0, 0, 0, 0))

    def test_one_step_losses_and_head_gradients_match_keras_equations(self):
        backend = self.backend()
        backend.configure_training("sgd", 0.01, momentum=0.9, loss_weights=(1.25, 1.0))
        states = np.random.default_rng(12).normal(size=(3, 14, 10, 9)).astype(np.float32)
        policies = np.zeros((3, backend._spec.n_labels), dtype=np.float32)
        policies[:, 0] = 0.75
        policies[:, 1] = 0.25
        values = np.asarray([[-1], [0], [1]], dtype=np.float32)
        heads = {}

        def capture(name):
            def hook(module, inputs, output):
                heads[name] = (inputs[0].detach().numpy().copy(), output.detach().numpy().copy(),
                               module.weight.detach().numpy().copy(), module.bias.detach().numpy().copy())
            return hook

        hooks = [backend.model.policy_fc.register_forward_hook(capture("policy")),
                 backend.model.value_fc2.register_forward_hook(capture("value"))]
        l2_before = sum(float(m.weight.detach().square().sum()) for m in backend.model.modules()
                        if isinstance(m, (nn.Conv2d, nn.Linear))) * backend._spec.l2_reg
        try:
            metrics = backend.train(states, policies, values, 3, 1, shuffle=False)
        finally:
            for hook in hooks:
                hook.remove()

        x, logits, w, b = heads["policy"]
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        policy_loss = float(-(policies * np.log(p)).sum(axis=1).mean())
        dlogits = 1.25 * (p - policies) / 3
        expected_w = w - 0.01 * (dlogits.T @ x + 2 * backend._spec.l2_reg * w)
        np.testing.assert_allclose(backend.model.policy_fc.weight.detach().numpy(), expected_w, atol=1e-7)
        np.testing.assert_allclose(backend.model.policy_fc.bias.detach().numpy(), b - 0.01 * dlogits.sum(axis=0), atol=1e-7)

        x, logits, w, b = heads["value"]
        v = np.tanh(logits)
        value_loss = float(np.mean((v - values) ** 2))
        dlogits = 2 * (v - values) * (1 - v ** 2) / 3
        expected_w = w - 0.01 * (dlogits.T @ x + 2 * backend._spec.l2_reg * w)
        np.testing.assert_allclose(backend.model.value_fc2.weight.detach().numpy(), expected_w, atol=1e-7)
        np.testing.assert_allclose(backend.model.value_fc2.bias.detach().numpy(), b - 0.01 * dlogits.sum(axis=0), atol=1e-7)
        self.assertAlmostEqual(metrics["train_policy_loss"], policy_loss, places=5)
        self.assertAlmostEqual(metrics["train_value_loss"], value_loss, places=5)
        self.assertAlmostEqual(metrics["train_loss"], 1.25 * policy_loss + value_loss + l2_before, places=5)

    def test_trained_checkpoint_matches_independent_numpy_legacy_forward(self):
        backend = self.backend()
        rng = np.random.default_rng(23)
        states = rng.normal(size=(2, 14, 10, 9)).astype(np.float32)
        policies = np.zeros((2, backend._spec.n_labels), dtype=np.float32)
        policies[:, 1] = 1
        backend.configure_training("sgd", 0.01, momentum=0.9)
        backend.train(states, policies, np.asarray([0.5, -0.5]), 2, 2)
        expected = backend.predict_batch(states)
        config_path, weight_path = str(self.root / "numpy.json"), str(self.root / "numpy.h5")
        backend.save_model(config_path, weight_path)
        config = json.loads(Path(config_path).read_text())
        # Interpret the exported Keras graph and HWIO kernels independently.
        outputs = {}
        with h5py.File(weight_path, "r") as weights:
            for layer in config["layers"]:
                cfg, kind = layer["config"], layer["class_name"]
                name = cfg["name"]
                if kind == "InputLayer":
                    outputs[name] = states.astype(np.float64)
                    continue
                inputs = [outputs[node[0]] for node in layer["inbound_nodes"][0]]
                x = inputs[0]
                group = weights[name]
                w = group[name] if name in group else group
                if kind == "Conv2D":
                    kernel = w["kernel:0"][:]
                    k = kernel.shape[0]
                    padded = np.pad(x, ((0, 0), (0, 0), (k // 2, k // 2), (k // 2, k // 2)))
                    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k), axis=(2, 3))
                    x = np.einsum("nchwkl,klco->nohw", windows, kernel)
                elif kind == "BatchNormalization":
                    mean, variance, gamma, beta = [w[key][:].reshape(1, -1, 1, 1) for key in
                        ("moving_mean:0", "moving_variance:0", "gamma:0", "beta:0")]
                    x = (x - mean) / np.sqrt(variance + cfg["epsilon"]) * gamma + beta
                elif kind == "Activation":
                    x = np.maximum(x, 0)
                elif kind == "Add":
                    x = inputs[0] + inputs[1]
                elif kind == "Flatten":
                    x = x.reshape(len(x), -1)
                elif kind == "Dense":
                    x = x @ w["kernel:0"][:] + w["bias:0"][:]
                    if cfg["activation"] == "relu":
                        x = np.maximum(x, 0)
                    elif cfg["activation"] == "tanh":
                        x = np.tanh(x)
                    elif cfg["activation"] == "softmax":
                        x = np.exp(x - x.max(axis=1, keepdims=True))
                        x /= x.sum(axis=1, keepdims=True)
                else:
                    self.fail(kind)
                outputs[name] = x
        for actual, node in zip(expected, config["output_layers"]):
            np.testing.assert_allclose(actual, outputs[node[0]], atol=1e-6, rtol=1e-5)
        clone = TorchModelBackend(self.config)
        clone.load_model(config_path, weight_path)
        for actual, wanted in zip(clone.predict_batch(states), expected):
            np.testing.assert_allclose(actual, wanted, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
