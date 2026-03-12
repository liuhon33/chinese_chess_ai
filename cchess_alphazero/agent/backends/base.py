from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class ModelBackend(ABC):
    def __init__(self, config):
        self.config = config

    @property
    @abstractmethod
    def model(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def build_model(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_model(self, config_path: str, weight_path: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def save_model(self, config_path: str, weight_path: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def configure_training(
        self,
        optimizer_name: str,
        learning_rate: float,
        momentum: float = 0.0,
        loss_weights: Tuple[float, float] = (1.0, 1.0),
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_learning_rate(self, learning_rate: float) -> None:
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError
