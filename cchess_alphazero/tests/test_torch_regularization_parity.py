import unittest

import numpy as np

from cchess_alphazero.config import Config


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None, None
    return torch, nn


torch, nn = _torch_modules()
TORCH_AVAILABLE = torch is not None

if TORCH_AVAILABLE:
    from cchess_alphazero.agent.backends.torch_backend import TorchModelBackend


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed or importable")
class TorchRegularizationParityTest(unittest.TestCase):
    def setUp(self):
        self.config = Config("mini")
        self.config.opts.backend = "torch"
        self.config.opts.device_list = "cpu"
        self.backend = TorchModelBackend(self.config)
        self.backend.build_model()

    def _set_deterministic_weights(self):
        with torch.no_grad():
            for module in self.backend.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.fill_(0.05)
                    if module.bias is not None:
                        module.bias.fill_(3.0)
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(5.0)
                    module.bias.fill_(7.0)
                    module.running_mean.zero_()
                    module.running_var.fill_(1.0)
                    module.num_batches_tracked.zero_()

    def test_kernel_only_l2_matches_keras_scope(self):
        self._set_deterministic_weights()

        expected = 0.0
        with torch.no_grad():
            for module in self.backend.model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    expected += self.backend._spec.l2_reg * float(module.weight.pow(2).sum().cpu())

        actual = float(self.backend._keras_regularization_loss().detach().cpu())
        self.assertAlmostEqual(actual, expected, places=6)

        with torch.no_grad():
            for module in self.backend.model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.weight.fill_(123.0)
                    module.bias.fill_(456.0)
                elif isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.fill_(789.0)

        after_non_kernel_changes = float(self.backend._keras_regularization_loss().detach().cpu())
        self.assertAlmostEqual(after_non_kernel_changes, actual, places=6)

    def test_run_epoch_reports_total_loss_with_regularization(self):
        self._set_deterministic_weights()
        self.backend.configure_training(
            optimizer_name="sgd",
            learning_rate=0.0,
            momentum=0.0,
            loss_weights=(1.0, 1.0),
        )

        state_ary = np.zeros((2, self.config.model.input_depth, 10, 9), dtype=np.float32)
        state_ary[0, 0, 0, 0] = 1.0
        state_ary[1, 1, 1, 1] = 1.0
        policy_ary = np.zeros((2, self.backend._spec.n_labels), dtype=np.float32)
        policy_ary[0, 0] = 1.0
        policy_ary[1, 1] = 1.0
        value_ary = np.asarray([[1.0], [-1.0]], dtype=np.float32)

        metrics = self.backend._run_epoch(
            model=self.backend.model,
            state_ary=state_ary,
            policy_ary=policy_ary,
            value_ary=value_ary,
            indices=np.arange(state_ary.shape[0]),
            batch_size=2,
            shuffle=False,
            training=True,
        )

        batch_states = torch.from_numpy(state_ary).to(self.backend.device)
        batch_policies = torch.from_numpy(policy_ary).to(self.backend.device)
        batch_values = torch.from_numpy(value_ary).to(self.backend.device)
        self.backend.model.train()
        with torch.no_grad():
            pred_policy, pred_value = self.backend.model(batch_states)
            policy_loss = -(batch_policies * torch.log(pred_policy.clamp_min(1e-8))).sum(dim=1).mean()
            value_loss = torch.nn.functional.mse_loss(pred_value, batch_values)
            regularization_loss = self.backend._keras_regularization_loss()
            total_loss = policy_loss + value_loss + regularization_loss

        self.assertAlmostEqual(metrics["policy_loss"], float(policy_loss.cpu()), places=6)
        self.assertAlmostEqual(metrics["value_loss"], float(value_loss.cpu()), places=6)
        self.assertAlmostEqual(metrics["regularization_loss"], float(regularization_loss.cpu()), places=6)
        self.assertAlmostEqual(metrics["loss"], float(total_loss.cpu()), places=6)


if __name__ == "__main__":
    unittest.main()
