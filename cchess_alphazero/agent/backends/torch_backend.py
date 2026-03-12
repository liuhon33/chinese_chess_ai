from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from logging import getLogger
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    F = None
    nn = None
    DataLoader = None
    TensorDataset = None

from cchess_alphazero.agent.backends.base import ModelBackend
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed

logger = getLogger(__name__)

BOARD_HEIGHT = 10
BOARD_WIDTH = 9
DEFAULT_POLICY_CHANNELS = 4
DEFAULT_VALUE_CHANNELS = 2
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.01
KERAS_BACKEND_NAME = "tensorflow"
KERAS_VERSION = "2.0.8"
INPUT_LAYER_NAME = "input_1"
POLICY_CONV_LAYER_NAME = "policy_conv-1-2"
VALUE_CONV_LAYER_NAME = "value_conv-1-4"
POLICY_OUT_LAYER_NAME = "policy_out"
VALUE_DENSE_LAYER_NAME = "value_dense"
VALUE_OUT_LAYER_NAME = "value_out"


@dataclass
class NetworkSpec:
    input_depth: int
    filter_num: int
    first_filter_size: int
    filter_size: int
    res_layer_num: int
    value_fc_size: int
    n_labels: int
    l2_reg: float
    policy_channels: int = DEFAULT_POLICY_CHANNELS
    value_channels: int = DEFAULT_VALUE_CHANNELS
    board_height: int = BOARD_HEIGHT
    board_width: int = BOARD_WIDTH


if nn is not None:
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int):
            super().__init__()
            padding = kernel_size // 2
            self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
            self.bn1 = nn.BatchNorm2d(channels, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm2d(channels, eps=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM)

        def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            return F.relu(x + residual)


    class CChessTorchModule(nn.Module):
        def __init__(self, spec: NetworkSpec):
            super().__init__()
            self.spec = spec
            self.input_conv = nn.Conv2d(
                spec.input_depth,
                spec.filter_num,
                spec.first_filter_size,
                padding=spec.first_filter_size // 2,
                bias=False,
            )
            self.input_bn = nn.BatchNorm2d(
                spec.filter_num,
                eps=BATCH_NORM_EPSILON,
                momentum=BATCH_NORM_MOMENTUM,
            )
            self.res_blocks = nn.ModuleList(
                [ResidualBlock(spec.filter_num, spec.filter_size) for _ in range(spec.res_layer_num)]
            )
            self.policy_conv = nn.Conv2d(spec.filter_num, spec.policy_channels, 1, bias=False)
            self.policy_bn = nn.BatchNorm2d(
                spec.policy_channels,
                eps=BATCH_NORM_EPSILON,
                momentum=BATCH_NORM_MOMENTUM,
            )
            self.policy_fc = nn.Linear(spec.policy_channels * spec.board_height * spec.board_width, spec.n_labels)
            self.value_conv = nn.Conv2d(spec.filter_num, spec.value_channels, 1, bias=False)
            self.value_bn = nn.BatchNorm2d(
                spec.value_channels,
                eps=BATCH_NORM_EPSILON,
                momentum=BATCH_NORM_MOMENTUM,
            )
            self.value_fc1 = nn.Linear(spec.value_channels * spec.board_height * spec.board_width, spec.value_fc_size)
            self.value_fc2 = nn.Linear(spec.value_fc_size, 1)

        def forward(self, x):
            x = F.relu(self.input_bn(self.input_conv(x)))
            for block in self.res_blocks:
                x = block(x)

            policy = F.relu(self.policy_bn(self.policy_conv(x)))
            policy = torch.flatten(policy, 1)
            policy = torch.softmax(self.policy_fc(policy), dim=1)

            value = F.relu(self.value_bn(self.value_conv(x)))
            value = torch.flatten(value, 1)
            value = F.relu(self.value_fc1(value))
            value = torch.tanh(self.value_fc2(value))
            return policy, value
else:
    class CChessTorchModule:
        def __init__(self, spec: NetworkSpec):
            raise ImportError("PyTorch is required to use the torch backend.")


def configure_torch_session(per_process_gpu_memory_fraction=None, allow_growth=None, device_list="0"):
    normalized = _normalize_device_list(device_list)
    if normalized.lower() == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif normalized:
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized

    if torch is not None and torch.cuda.is_available():
        if per_process_gpu_memory_fraction is not None and hasattr(torch.cuda, "set_per_process_memory_fraction"):
            try:
                torch.cuda.set_per_process_memory_fraction(float(per_process_gpu_memory_fraction), 0)
            except Exception:
                pass
    return normalized

class TorchModelBackend(ModelBackend):
    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._lock = RLock()
        self._device = None
        self._spec = None
        self._optimizer = None
        self._optimizer_name = None
        self._loss_weights = (1.0, 1.0)
        self._parallel_model = None

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    def build_model(self) -> None:
        self._require_torch()
        mc = self.config.model
        spec = NetworkSpec(
            input_depth=mc.input_depth,
            filter_num=mc.cnn_filter_num,
            first_filter_size=mc.cnn_first_filter_size,
            filter_size=mc.cnn_filter_size,
            res_layer_num=mc.res_layer_num,
            value_fc_size=mc.value_fc_size,
            n_labels=len(ActionLabelsRed),
            l2_reg=mc.l2_reg,
        )
        self._build_from_spec(spec)

    def load_model(self, config_path: str, weight_path: str) -> bool:
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            return False

        self._require_torch()
        with open(config_path, "rt") as config_file:
            config_data = json.load(config_file)

        if config_data.get("backend") == "torch":
            spec = NetworkSpec(**config_data["network_spec"])
            self._build_from_spec(spec)
            self._load_torch_checkpoint(weight_path)
            return True

        spec = self._infer_spec_from_keras_files(config_data, weight_path)
        self._build_from_spec(spec)
        self._load_keras_weights(weight_path)
        return True

    def save_model(self, config_path: str, weight_path: str) -> None:
        self._require_torch()
        with self._lock:
            self._ensure_model_built()
            keras_config = self._build_keras_config(self._spec)
            with open(config_path, "wt") as config_file:
                json.dump(keras_config, config_file)
            self._save_keras_weights(weight_path)

    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._require_torch()
        batch = np.asarray(data, dtype=np.float32)
        with self._lock:
            self._ensure_model_built()
            batch = self._validate_input_array(batch)
            self._model.eval()
            self._assert_model_device()
            with torch.no_grad():
                tensor = torch.from_numpy(batch).to(self._device)
                self._assert_tensor_device(tensor)
                policy, value = self._model(tensor)
            policy_np = policy.detach().cpu().numpy().astype(np.float32, copy=False)
            value_np = value.detach().cpu().numpy().astype(np.float32, copy=False)
            return self._validate_output_arrays(batch.shape[0], policy_np, value_np)

    def configure_training(
        self,
        optimizer_name: str,
        learning_rate: float,
        momentum: float = 0.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        self._require_torch()
        with self._lock:
            self._ensure_model_built()
            self._optimizer_name = optimizer_name.lower()
            self._loss_weights = tuple(loss_weights)
            params = self._model.parameters()
            if self._optimizer_name == "sgd":
                self._optimizer = torch.optim.SGD(params, lr=float(learning_rate), momentum=float(momentum))
            elif self._optimizer_name == "adam":
                self._optimizer = torch.optim.Adam(params, lr=float(learning_rate))
            else:
                raise ValueError(f"Unsupported optimizer for torch backend: {optimizer_name}")
            self._parallel_model = self._build_parallel_model()

    def set_learning_rate(self, learning_rate: float) -> None:
        with self._lock:
            if self._optimizer is None:
                raise RuntimeError("Training optimizer has not been configured")
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = float(learning_rate)

    def train(
        self,
        state_ary: np.ndarray,
        policy_ary: np.ndarray,
        value_ary: np.ndarray,
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
        validation_split: float = 0.0,
    ) -> Dict[str, float]:
        self._require_torch()
        with self._lock:
            self._ensure_model_built()
            if self._optimizer is None:
                raise RuntimeError("Training optimizer has not been configured")

            state_ary = self._validate_input_array(np.asarray(state_ary, dtype=np.float32))
            policy_ary = np.asarray(policy_ary, dtype=np.float32)
            value_ary = np.asarray(value_ary, dtype=np.float32).reshape(-1, 1)
            if policy_ary.shape != (state_ary.shape[0], self._spec.n_labels):
                raise ValueError(
                    f"Expected policy targets shape ({state_ary.shape[0]}, {self._spec.n_labels}), got {policy_ary.shape}"
                )
            if value_ary.shape != (state_ary.shape[0], 1):
                raise ValueError(f"Expected value targets shape ({state_ary.shape[0]}, 1), got {value_ary.shape}")

            train_indices, val_indices = self._split_indices(state_ary.shape[0], validation_split, shuffle)
            model = self._parallel_model if self._parallel_model is not None else self._model
            history = {}

            for epoch in range(epochs):
                train_metrics = self._run_epoch(
                    model=model,
                    state_ary=state_ary,
                    policy_ary=policy_ary,
                    value_ary=value_ary,
                    indices=train_indices,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    training=True,
                )
                history = {f"train_{k}": v for k, v in train_metrics.items()}
                if len(val_indices) > 0:
                    val_metrics = self._run_epoch(
                        model=model,
                        state_ary=state_ary,
                        policy_ary=policy_ary,
                        value_ary=value_ary,
                        indices=val_indices,
                        batch_size=batch_size,
                        shuffle=False,
                        training=False,
                    )
                    history.update({f"val_{k}": v for k, v in val_metrics.items()})
                logger.info(
                    "Torch epoch %s/%s loss=%.5f policy=%.5f value=%.5f",
                    epoch + 1,
                    epochs,
                    history["train_loss"],
                    history["train_policy_loss"],
                    history["train_value_loss"],
                )

            self._model.eval()
            self._assert_model_device()
            return history

    def _build_from_spec(self, spec: NetworkSpec) -> None:
        with self._lock:
            self._spec = spec
            self._device = self._resolve_device()
            self._model = CChessTorchModule(spec).to(self._device)
            self._optimizer = None
            self._optimizer_name = None
            self._parallel_model = None
            self._model.eval()
            self._assert_model_device()

    def _load_torch_checkpoint(self, weight_path: str) -> None:
        with self._lock:
            self._ensure_model_built()
            state_dict = torch.load(weight_path, map_location=self._device)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            self._assert_model_device()

    def _infer_spec_from_keras_files(self, config_data: Dict[str, object], weight_path: str) -> NetworkSpec:
        with h5py.File(weight_path, "r") as weights_file:
            layer_names = _decode_names(weights_file.attrs.get("layer_names", []))
            input_conv_name = next(name for name in layer_names if name.startswith("input_conv-"))
            input_kernel = self._read_weight(weights_file, input_conv_name, "kernel:0")
            res_indices = sorted(
                {int(match.group(1)) for name in layer_names for match in [re.match(r"res(\d+)_conv1", name)] if match}
            )
            filter_size = _extract_filter_size_from_config(config_data)
            if not filter_size and res_indices:
                res_name = next(name for name in layer_names if re.match(rf"res{res_indices[0]}_conv1", name))
                filter_size = int(self._read_weight(weights_file, res_name, "kernel:0").shape[0])
            if not filter_size:
                filter_size = int(input_kernel.shape[0])

            policy_kernel = self._read_weight(weights_file, POLICY_CONV_LAYER_NAME, "kernel:0")
            value_kernel = self._read_weight(weights_file, VALUE_CONV_LAYER_NAME, "kernel:0")
            value_dense_bias = self._read_weight(weights_file, VALUE_DENSE_LAYER_NAME, "bias:0")
            policy_out_bias = self._read_weight(weights_file, POLICY_OUT_LAYER_NAME, "bias:0")

        return NetworkSpec(
            input_depth=int(input_kernel.shape[2]),
            filter_num=int(input_kernel.shape[3]),
            first_filter_size=int(input_kernel.shape[0]),
            filter_size=int(filter_size),
            res_layer_num=res_indices[-1] if res_indices else 0,
            value_fc_size=int(value_dense_bias.shape[0]),
            n_labels=int(policy_out_bias.shape[0]),
            l2_reg=_extract_l2_from_config(config_data) or self.config.model.l2_reg,
            policy_channels=int(policy_kernel.shape[3]),
            value_channels=int(value_kernel.shape[3]),
        )

    def _build_parallel_model(self):
        if not self.config.opts.use_multiple_gpus or self._device.type != "cuda":
            return None
        if torch.cuda.device_count() < 2:
            return None
        return nn.DataParallel(self._model)

    def _run_epoch(
        self,
        model,
        state_ary: np.ndarray,
        policy_ary: np.ndarray,
        value_ary: np.ndarray,
        indices: np.ndarray,
        batch_size: int,
        shuffle: bool,
        training: bool,
    ) -> Dict[str, float]:
        if len(indices) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        states = torch.from_numpy(state_ary[indices])
        policies = torch.from_numpy(policy_ary[indices])
        values = torch.from_numpy(value_ary[indices])
        dataset = TensorDataset(states, policies, values)
        loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle)

        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_count = 0

        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(self._device)
            batch_policies = batch_policies.to(self._device)
            batch_values = batch_values.to(self._device)
            batch_count = batch_states.shape[0]

            if training:
                self._optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                pred_policy, pred_value = model(batch_states)
                policy_loss = -(batch_policies * torch.log(pred_policy.clamp_min(1e-8))).sum(dim=1).mean()
                value_loss = F.mse_loss(pred_value, batch_values)
                loss = self._loss_weights[0] * policy_loss + self._loss_weights[1] * value_loss
                if training:
                    loss.backward()
                    self._optimizer.step()

            total_count += batch_count
            total_loss += float(loss.detach().cpu()) * batch_count
            total_policy_loss += float(policy_loss.detach().cpu()) * batch_count
            total_value_loss += float(value_loss.detach().cpu()) * batch_count

        return {
            "loss": total_loss / total_count,
            "policy_loss": total_policy_loss / total_count,
            "value_loss": total_value_loss / total_count,
        }

    def _split_indices(self, sample_count: int, validation_split: float, shuffle: bool) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.arange(sample_count)
        if shuffle:
            np.random.shuffle(indices)
        if validation_split <= 0 or sample_count < 2:
            return indices, np.asarray([], dtype=np.int64)
        val_size = int(round(sample_count * validation_split))
        val_size = min(max(val_size, 1), sample_count - 1)
        return indices[val_size:], indices[:val_size]

    def _save_keras_weights(self, weight_path: str) -> None:
        layer_names = [layer["config"]["name"] for layer in self._build_keras_config(self._spec)["layers"]]
        weights_by_layer = self._serialize_layer_weights()

        with h5py.File(weight_path, "w") as weights_file:
            weights_file.attrs["backend"] = np.bytes_(KERAS_BACKEND_NAME)
            weights_file.attrs["keras_version"] = np.bytes_(KERAS_VERSION)
            weights_file.attrs["layer_names"] = _encode_names(layer_names)

            for layer_name in layer_names:
                layer_group = weights_file.create_group(layer_name)
                entries = weights_by_layer.get(layer_name, [])
                layer_group.attrs["weight_names"] = _encode_names([name for name, _ in entries])
                if not entries:
                    continue
                nested_group = layer_group.create_group(layer_name)
                for weight_name, array in entries:
                    dataset_name = weight_name.split("/", 1)[1]
                    nested_group.create_dataset(dataset_name, data=array)

    def _load_keras_weights(self, weight_path: str) -> None:
        with h5py.File(weight_path, "r") as weights_file:
            self._copy_conv(self._model.input_conv, self._read_weight(weights_file, self._input_conv_name, "kernel:0"))
            self._copy_batch_norm(self._model.input_bn, self._read_batch_norm(weights_file, "input_batchnorm"))

            for index, block in enumerate(self._model.res_blocks, start=1):
                conv1_name = self._res_conv_name(index, 1)
                conv2_name = self._res_conv_name(index, 2)
                self._copy_conv(block.conv1, self._read_weight(weights_file, conv1_name, "kernel:0"))
                self._copy_batch_norm(block.bn1, self._read_batch_norm(weights_file, f"res{index}_batchnorm1"))
                self._copy_conv(block.conv2, self._read_weight(weights_file, conv2_name, "kernel:0"))
                self._copy_batch_norm(block.bn2, self._read_batch_norm(weights_file, f"res{index}_batchnorm2"))

            self._copy_conv(self._model.policy_conv, self._read_weight(weights_file, POLICY_CONV_LAYER_NAME, "kernel:0"))
            self._copy_batch_norm(self._model.policy_bn, self._read_batch_norm(weights_file, "policy_batchnorm"))
            self._copy_dense(self._model.policy_fc, self._read_dense(weights_file, POLICY_OUT_LAYER_NAME))
            self._copy_conv(self._model.value_conv, self._read_weight(weights_file, VALUE_CONV_LAYER_NAME, "kernel:0"))
            self._copy_batch_norm(self._model.value_bn, self._read_batch_norm(weights_file, "value_batchnorm"))
            self._copy_dense(self._model.value_fc1, self._read_dense(weights_file, VALUE_DENSE_LAYER_NAME))
            self._copy_dense(self._model.value_fc2, self._read_dense(weights_file, VALUE_OUT_LAYER_NAME))

        with self._lock:
            self._model.eval()
            self._assert_model_device()

    def _serialize_layer_weights(self) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        layer_weights: Dict[str, List[Tuple[str, np.ndarray]]] = {
            INPUT_LAYER_NAME: [],
            self._input_conv_name: [(f"{self._input_conv_name}/kernel:0", self._conv_kernel_to_keras(self._model.input_conv.weight))],
            "input_batchnorm": self._batch_norm_to_keras("input_batchnorm", self._model.input_bn),
            "input_relu": [],
            POLICY_CONV_LAYER_NAME: [(f"{POLICY_CONV_LAYER_NAME}/kernel:0", self._conv_kernel_to_keras(self._model.policy_conv.weight))],
            "policy_batchnorm": self._batch_norm_to_keras("policy_batchnorm", self._model.policy_bn),
            "policy_relu": [],
            "policy_flatten": [],
            POLICY_OUT_LAYER_NAME: self._dense_to_keras(POLICY_OUT_LAYER_NAME, self._model.policy_fc),
            VALUE_CONV_LAYER_NAME: [(f"{VALUE_CONV_LAYER_NAME}/kernel:0", self._conv_kernel_to_keras(self._model.value_conv.weight))],
            "value_batchnorm": self._batch_norm_to_keras("value_batchnorm", self._model.value_bn),
            "value_relu": [],
            "value_flatten": [],
            VALUE_DENSE_LAYER_NAME: self._dense_to_keras(VALUE_DENSE_LAYER_NAME, self._model.value_fc1),
            VALUE_OUT_LAYER_NAME: self._dense_to_keras(VALUE_OUT_LAYER_NAME, self._model.value_fc2),
        }

        for index, block in enumerate(self._model.res_blocks, start=1):
            conv1_name = self._res_conv_name(index, 1)
            conv2_name = self._res_conv_name(index, 2)
            layer_weights[conv1_name] = [(f"{conv1_name}/kernel:0", self._conv_kernel_to_keras(block.conv1.weight))]
            layer_weights[f"res{index}_batchnorm1"] = self._batch_norm_to_keras(f"res{index}_batchnorm1", block.bn1)
            layer_weights[f"res{index}_relu1"] = []
            layer_weights[conv2_name] = [(f"{conv2_name}/kernel:0", self._conv_kernel_to_keras(block.conv2.weight))]
            layer_weights[f"res{index}_batchnorm2"] = self._batch_norm_to_keras(f"res{index}_batchnorm2", block.bn2)
            layer_weights[f"res{index}_add"] = []
            layer_weights[f"res{index}_relu2"] = []

        return layer_weights

    def _build_keras_config(self, spec: NetworkSpec) -> Dict[str, object]:
        layers: List[Dict[str, object]] = [
            _make_input_layer(spec.input_depth),
            _make_conv_layer(self._input_conv_name, INPUT_LAYER_NAME, spec.filter_num, spec.first_filter_size, spec.l2_reg),
            _make_batch_norm_layer("input_batchnorm", self._input_conv_name),
            _make_activation_layer("input_relu", "input_batchnorm", "relu"),
        ]

        prev_layer = "input_relu"
        for index in range(1, spec.res_layer_num + 1):
            conv1_name = self._res_conv_name(index, 1)
            bn1_name = f"res{index}_batchnorm1"
            relu1_name = f"res{index}_relu1"
            conv2_name = self._res_conv_name(index, 2)
            bn2_name = f"res{index}_batchnorm2"
            add_name = f"res{index}_add"
            relu2_name = f"res{index}_relu2"
            layers.extend(
                [
                    _make_conv_layer(conv1_name, prev_layer, spec.filter_num, spec.filter_size, spec.l2_reg),
                    _make_batch_norm_layer(bn1_name, conv1_name),
                    _make_activation_layer(relu1_name, bn1_name, "relu"),
                    _make_conv_layer(conv2_name, relu1_name, spec.filter_num, spec.filter_size, spec.l2_reg),
                    _make_batch_norm_layer(bn2_name, conv2_name),
                    _make_add_layer(add_name, [prev_layer, bn2_name]),
                    _make_activation_layer(relu2_name, add_name, "relu"),
                ]
            )
            prev_layer = relu2_name

        layers.extend(
            [
                _make_conv_layer(POLICY_CONV_LAYER_NAME, prev_layer, spec.policy_channels, 1, spec.l2_reg),
                _make_batch_norm_layer("policy_batchnorm", POLICY_CONV_LAYER_NAME),
                _make_activation_layer("policy_relu", "policy_batchnorm", "relu"),
                _make_flatten_layer("policy_flatten", "policy_relu"),
                _make_dense_layer(POLICY_OUT_LAYER_NAME, "policy_flatten", spec.n_labels, "softmax", spec.l2_reg),
                _make_conv_layer(VALUE_CONV_LAYER_NAME, prev_layer, spec.value_channels, 1, spec.l2_reg),
                _make_batch_norm_layer("value_batchnorm", VALUE_CONV_LAYER_NAME),
                _make_activation_layer("value_relu", "value_batchnorm", "relu"),
                _make_flatten_layer("value_flatten", "value_relu"),
                _make_dense_layer(VALUE_DENSE_LAYER_NAME, "value_flatten", spec.value_fc_size, "relu", spec.l2_reg),
                _make_dense_layer(VALUE_OUT_LAYER_NAME, VALUE_DENSE_LAYER_NAME, 1, "tanh", spec.l2_reg),
            ]
        )

        return {
            "name": "cchess_model",
            "layers": layers,
            "input_layers": [[INPUT_LAYER_NAME, 0, 0]],
            "output_layers": [[POLICY_OUT_LAYER_NAME, 0, 0], [VALUE_OUT_LAYER_NAME, 0, 0]],
        }

    @property
    def _input_conv_name(self) -> str:
        return f"input_conv-{self._spec.first_filter_size}-{self._spec.filter_num}"

    def _res_conv_name(self, index: int, part: int) -> str:
        return f"res{index}_conv{part}-{self._spec.filter_size}-{self._spec.filter_num}"

    @staticmethod
    def _read_weight(weights_file: h5py.File, layer_name: str, weight_name: str) -> np.ndarray:
        layer_group = weights_file[layer_name]
        if layer_name in layer_group:
            data = layer_group[layer_name][weight_name]
        else:
            data = layer_group[weight_name]
        return np.asarray(data, dtype=np.float32)

    def _read_batch_norm(self, weights_file: h5py.File, layer_name: str) -> Dict[str, np.ndarray]:
        return {
            "gamma": self._read_weight(weights_file, layer_name, "gamma:0"),
            "beta": self._read_weight(weights_file, layer_name, "beta:0"),
            "moving_mean": self._read_weight(weights_file, layer_name, "moving_mean:0"),
            "moving_variance": self._read_weight(weights_file, layer_name, "moving_variance:0"),
        }

    def _read_dense(self, weights_file: h5py.File, layer_name: str) -> Dict[str, np.ndarray]:
        return {
            "kernel": self._read_weight(weights_file, layer_name, "kernel:0"),
            "bias": self._read_weight(weights_file, layer_name, "bias:0"),
        }

    def _copy_conv(self, module, kernel: np.ndarray) -> None:
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))).to(self._device))

    def _copy_batch_norm(self, module, weights: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(weights["gamma"]).to(self._device))
            module.bias.copy_(torch.from_numpy(weights["beta"]).to(self._device))
            module.running_mean.copy_(torch.from_numpy(weights["moving_mean"]).to(self._device))
            module.running_var.copy_(torch.from_numpy(weights["moving_variance"]).to(self._device))
            module.num_batches_tracked.zero_()

    def _copy_dense(self, module, weights: Dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            module.weight.copy_(torch.from_numpy(weights["kernel"].T).to(self._device))
            module.bias.copy_(torch.from_numpy(weights["bias"]).to(self._device))

    @staticmethod
    def _conv_kernel_to_keras(weight) -> np.ndarray:
        return np.transpose(weight.detach().cpu().numpy(), (2, 3, 1, 0)).astype(np.float32, copy=False)

    def _batch_norm_to_keras(self, layer_name: str, module) -> List[Tuple[str, np.ndarray]]:
        return [
            (f"{layer_name}/gamma:0", module.weight.detach().cpu().numpy().astype(np.float32, copy=False)),
            (f"{layer_name}/beta:0", module.bias.detach().cpu().numpy().astype(np.float32, copy=False)),
            (f"{layer_name}/moving_mean:0", module.running_mean.detach().cpu().numpy().astype(np.float32, copy=False)),
            (f"{layer_name}/moving_variance:0", module.running_var.detach().cpu().numpy().astype(np.float32, copy=False)),
        ]

    def _dense_to_keras(self, layer_name: str, module) -> List[Tuple[str, np.ndarray]]:
        return [
            (f"{layer_name}/kernel:0", module.weight.detach().cpu().numpy().T.astype(np.float32, copy=False)),
            (f"{layer_name}/bias:0", module.bias.detach().cpu().numpy().astype(np.float32, copy=False)),
        ]

    def _validate_input_array(self, data: np.ndarray) -> np.ndarray:
        expected_depth = self._spec.input_depth if self._spec is not None else self.config.model.input_depth
        if data.ndim != 4:
            raise ValueError(f"Expected 4D input batch, got shape {data.shape}")
        expected_shape = (expected_depth, BOARD_HEIGHT, BOARD_WIDTH)
        if tuple(data.shape[1:]) != expected_shape:
            raise ValueError(f"Expected input shape (N, {expected_depth}, {BOARD_HEIGHT}, {BOARD_WIDTH}), got {data.shape}")
        return np.ascontiguousarray(data, dtype=np.float32)

    def _validate_output_arrays(self, batch_size: int, policy: np.ndarray, value: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if policy.shape != (batch_size, self._spec.n_labels):
            raise ValueError(f"Expected policy output shape ({batch_size}, {self._spec.n_labels}), got {policy.shape}")
        if value.shape == (batch_size,):
            value = value.reshape(batch_size, 1)
        if value.shape != (batch_size, 1):
            raise ValueError(f"Expected value output shape ({batch_size}, 1), got {value.shape}")
        return policy, value

    def _resolve_device(self):
        self._require_torch()
        normalized = _normalize_device_list(getattr(self.config.opts, "device_list", "0"))
        if normalized.lower() == "cpu":
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def _assert_model_device(self) -> None:
        param_device = next(self._model.parameters()).device
        if param_device != self._device:
            raise RuntimeError(f"Model parameters are on {param_device}, expected {self._device}")

    def _assert_tensor_device(self, tensor) -> None:
        if tensor.device != self._device:
            raise RuntimeError(f"Input tensor is on {tensor.device}, expected {self._device}")

    def _ensure_model_built(self) -> None:
        if self._model is None or self._spec is None or self._device is None:
            raise RuntimeError("Model backend has not been initialized")

    def _require_torch(self) -> None:
        if torch is None:
            raise ImportError("PyTorch backend selected but torch is not installed. Install the 'torch' package.")


def _normalize_device_list(device_list) -> str:
    if device_list is None:
        return "0"
    return str(device_list).strip() or "0"


def _decode_names(names: Iterable[object]) -> List[str]:
    decoded = []
    for name in names:
        if isinstance(name, bytes):
            decoded.append(name.decode("utf-8"))
        else:
            decoded.append(str(name))
    return decoded


def _encode_names(names: Iterable[str]) -> np.ndarray:
    return np.asarray([name.encode("utf-8") for name in names])


def _extract_l2_from_config(config_data: Dict[str, object]) -> float:
    for layer in config_data.get("layers", []):
        regularizer = layer.get("config", {}).get("kernel_regularizer")
        if regularizer and regularizer.get("config"):
            return float(regularizer["config"].get("l2", 0.0))
    return 0.0


def _extract_filter_size_from_config(config_data: Dict[str, object]) -> int:
    for layer in config_data.get("layers", []):
        name = layer.get("config", {}).get("name", "")
        match = re.match(r"res\d+_conv1-(\d+)-\d+", name)
        if match:
            return int(match.group(1))
        if name.startswith("res") and layer.get("config", {}).get("kernel_size"):
            return int(layer["config"]["kernel_size"][0])
    return 0


def _single_inbound(layer_name: str) -> List[List[List[object]]]:
    return [[[layer_name, 0, 0, {}]]]


def _multi_inbound(layer_names: Iterable[str]) -> List[List[List[object]]]:
    return [[[layer_name, 0, 0, {}] for layer_name in layer_names]]


def _variance_scaling() -> Dict[str, object]:
    return {
        "class_name": "VarianceScaling",
        "config": {
            "scale": 1.0,
            "mode": "fan_avg",
            "distribution": "uniform",
            "seed": None,
        },
    }


def _l2_regularizer(l2_reg: float) -> Dict[str, object]:
    return {
        "class_name": "L1L2",
        "config": {
            "l1": 0.0,
            "l2": float(l2_reg),
        },
    }


def _make_input_layer(input_depth: int) -> Dict[str, object]:
    return {
        "class_name": "InputLayer",
        "config": {
            "batch_input_shape": [None, input_depth, BOARD_HEIGHT, BOARD_WIDTH],
            "dtype": "float32",
            "name": INPUT_LAYER_NAME,
            "sparse": False,
        },
        "inbound_nodes": [],
        "name": INPUT_LAYER_NAME,
    }


def _make_conv_layer(name: str, inbound: str, filters: int, kernel_size: int, l2_reg: float) -> Dict[str, object]:
    return {
        "class_name": "Conv2D",
        "config": {
            "name": name,
            "trainable": True,
            "filters": filters,
            "kernel_size": [kernel_size, kernel_size],
            "strides": [1, 1],
            "padding": "same",
            "data_format": "channels_first",
            "dilation_rate": [1, 1],
            "activation": "linear",
            "use_bias": False,
            "kernel_initializer": _variance_scaling(),
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": _l2_regularizer(l2_reg),
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        },
        "inbound_nodes": _single_inbound(inbound),
        "name": name,
    }


def _make_batch_norm_layer(name: str, inbound: str) -> Dict[str, object]:
    return {
        "class_name": "BatchNormalization",
        "config": {
            "name": name,
            "trainable": True,
            "axis": 1,
            "momentum": 0.99,
            "epsilon": BATCH_NORM_EPSILON,
            "center": True,
            "scale": True,
            "beta_initializer": {"class_name": "Zeros", "config": {}},
            "gamma_initializer": {"class_name": "Ones", "config": {}},
            "moving_mean_initializer": {"class_name": "Zeros", "config": {}},
            "moving_variance_initializer": {"class_name": "Ones", "config": {}},
            "beta_regularizer": None,
            "gamma_regularizer": None,
            "beta_constraint": None,
            "gamma_constraint": None,
        },
        "inbound_nodes": _single_inbound(inbound),
        "name": name,
    }


def _make_activation_layer(name: str, inbound: str, activation: str) -> Dict[str, object]:
    return {
        "class_name": "Activation",
        "config": {
            "name": name,
            "trainable": True,
            "activation": activation,
        },
        "inbound_nodes": _single_inbound(inbound),
        "name": name,
    }


def _make_add_layer(name: str, inbound: Iterable[str]) -> Dict[str, object]:
    return {
        "class_name": "Add",
        "config": {
            "name": name,
            "trainable": True,
        },
        "inbound_nodes": _multi_inbound(inbound),
        "name": name,
    }


def _make_flatten_layer(name: str, inbound: str) -> Dict[str, object]:
    return {
        "class_name": "Flatten",
        "config": {
            "name": name,
            "trainable": True,
            "data_format": "channels_first",
        },
        "inbound_nodes": _single_inbound(inbound),
        "name": name,
    }


def _make_dense_layer(name: str, inbound: str, units: int, activation: str, l2_reg: float) -> Dict[str, object]:
    return {
        "class_name": "Dense",
        "config": {
            "name": name,
            "trainable": True,
            "units": units,
            "activation": activation,
            "use_bias": True,
            "kernel_initializer": _variance_scaling(),
            "bias_initializer": {"class_name": "Zeros", "config": {}},
            "kernel_regularizer": _l2_regularizer(l2_reg),
            "bias_regularizer": None,
            "activity_regularizer": None,
            "kernel_constraint": None,
            "bias_constraint": None,
        },
        "inbound_nodes": _single_inbound(inbound),
        "name": name,
    }
