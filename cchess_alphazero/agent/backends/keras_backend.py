from __future__ import annotations

import json
import os
from threading import RLock
from typing import Dict, Tuple

import numpy as np

from cchess_alphazero.agent.backends.base import ModelBackend
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed


def configure_keras_session(per_process_gpu_memory_fraction=None, allow_growth=None, device_list="0"):
    import tensorflow as tf
    import keras.backend as K

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            per_process_gpu_memory_fraction=per_process_gpu_memory_fraction,
            allow_growth=allow_growth,
            visible_device_list=device_list,
        )
    )
    sess = tf.Session(config=config)
    K.set_session(sess)


class KerasModelBackend(ModelBackend):
    def __init__(self, config):
        super().__init__(config)
        self._model = None
        self._graph = None
        self._lock = RLock()
        self._optimizer = None
        self._loss_weights = (1.0, 1.0)

    @property
    def model(self):
        return self._model

    def build_model(self) -> None:
        modules = self._keras_modules()
        tf = modules["tf"]
        Input = modules["Input"]
        Model = modules["Model"]
        Conv2D = modules["Conv2D"]
        Activation = modules["Activation"]
        Dense = modules["Dense"]
        Flatten = modules["Flatten"]
        BatchNormalization = modules["BatchNormalization"]
        l2 = modules["l2"]

        mc = self.config.model
        n_labels = len(ActionLabelsRed)

        in_x = x = Input((14, 10, 9))

        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_first_filter_size,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name="input_conv-" + str(mc.cnn_first_filter_size) + "-" + str(mc.cnn_filter_num),
        )(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1, modules)

        res_out = x

        x = Conv2D(
            filters=4,
            kernel_size=1,
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name="policy_conv-1-2",
        )(res_out)
        x = BatchNormalization(axis=1, name="policy_batchnorm")(x)
        x = Activation("relu", name="policy_relu")(x)
        x = Flatten(name="policy_flatten")(x)
        policy_out = Dense(
            n_labels,
            kernel_regularizer=l2(mc.l2_reg),
            activation="softmax",
            name="policy_out",
        )(x)

        x = Conv2D(
            filters=2,
            kernel_size=1,
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name="value_conv-1-4",
        )(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(
            mc.value_fc_size,
            kernel_regularizer=l2(mc.l2_reg),
            activation="relu",
            name="value_dense",
        )(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg), activation="tanh", name="value_out")(x)

        with self._lock:
            self._model = Model(in_x, [policy_out, value_out], name="cchess_model")
            self._graph = tf.get_default_graph()

    def load_model(self, config_path: str, weight_path: str) -> bool:
        if not (os.path.exists(config_path) and os.path.exists(weight_path)):
            return False

        modules = self._keras_modules()
        tf = modules["tf"]
        Model = modules["Model"]

        with self._lock:
            with open(config_path, "rt") as f:
                self._model = Model.from_config(json.load(f))
            self._model.load_weights(weight_path)
            self._graph = tf.get_default_graph()
        return True

    def save_model(self, config_path: str, weight_path: str) -> None:
        with self._lock:
            with open(config_path, "wt") as f:
                json.dump(self._model.get_config(), f)
            self._model.save_weights(weight_path)

    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._graph is None:
                return self._model.predict_on_batch(data)
            with self._graph.as_default():
                return self._model.predict_on_batch(data)

    def configure_training(
        self,
        optimizer_name: str,
        learning_rate: float,
        momentum: float = 0.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        modules = self._keras_training_modules()
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "sgd":
            optimizer = modules["SGD"](lr=learning_rate, momentum=momentum)
        elif optimizer_name == "adam":
            optimizer = modules["Adam"](lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer for keras backend: {optimizer_name}")

        with self._lock:
            self._optimizer = optimizer
            self._loss_weights = tuple(loss_weights)
            self._model.compile(
                optimizer=self._optimizer,
                loss=["categorical_crossentropy", "mean_squared_error"],
                loss_weights=list(self._loss_weights),
            )

    def set_learning_rate(self, learning_rate: float) -> None:
        modules = self._keras_training_modules()
        with self._lock:
            if self._optimizer is None:
                raise RuntimeError("Training optimizer has not been configured")
            modules["K"].set_value(self._optimizer.lr, learning_rate)

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
        with self._lock:
            if self._optimizer is None:
                raise RuntimeError("Training optimizer has not been configured")
            history = self._model.fit(
                state_ary,
                [policy_ary, value_ary],
                batch_size=batch_size,
                epochs=epochs,
                shuffle=shuffle,
                validation_split=validation_split,
                verbose=1,
            )
        metrics = {key: float(values[-1]) for key, values in history.history.items() if values}
        return metrics

    def _build_residual_block(self, x, index, modules):
        Conv2D = modules["Conv2D"]
        Activation = modules["Activation"]
        Add = modules["Add"]
        BatchNormalization = modules["BatchNormalization"]
        l2 = modules["l2"]

        mc = self.config.model
        in_x = x
        res_name = "res" + str(index)
        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name=res_name + "_conv1-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num),
        )(x)
        x = BatchNormalization(axis=1, name=res_name + "_batchnorm1")(x)
        x = Activation("relu", name=res_name + "_relu1")(x)
        x = Conv2D(
            filters=mc.cnn_filter_num,
            kernel_size=mc.cnn_filter_size,
            padding="same",
            data_format="channels_first",
            use_bias=False,
            kernel_regularizer=l2(mc.l2_reg),
            name=res_name + "_conv2-" + str(mc.cnn_filter_size) + "-" + str(mc.cnn_filter_num),
        )(x)
        x = BatchNormalization(axis=1, name="res" + str(index) + "_batchnorm2")(x)
        x = Add(name=res_name + "_add")([in_x, x])
        x = Activation("relu", name=res_name + "_relu2")(x)
        return x

    @staticmethod
    def _keras_modules():
        import tensorflow as tf

        from keras.engine.topology import Input
        from keras.engine.training import Model
        from keras.layers.convolutional import Conv2D
        from keras.layers.core import Activation, Dense, Flatten
        from keras.layers.merge import Add
        from keras.layers.normalization import BatchNormalization
        from keras.regularizers import l2

        return {
            "tf": tf,
            "Input": Input,
            "Model": Model,
            "Conv2D": Conv2D,
            "Activation": Activation,
            "Dense": Dense,
            "Flatten": Flatten,
            "Add": Add,
            "BatchNormalization": BatchNormalization,
            "l2": l2,
        }

    @staticmethod
    def _keras_training_modules():
        import keras.backend as K
        from keras.optimizers import Adam, SGD

        return {
            "K": K,
            "Adam": Adam,
            "SGD": SGD,
        }
