import hashlib
import os
from logging import getLogger

import numpy as np

from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.agent.backends import create_model_backend
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed

logger = getLogger(__name__)


class CChessModel:
    def __init__(self, config: Config):
        self.config = config
        self.backend = create_model_backend(config)
        self.model = self.backend.model
        self.digest = None
        self.n_labels = len(ActionLabelsRed)
        self.api = None

    def build(self):
        self.backend.build_model()
        self.model = self.backend.model

    @staticmethod
    def fetch_digest(weight_path):
        if os.path.exists(weight_path):
            m = hashlib.sha256()
            with open(weight_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None

    def load(self, config_path, weight_path):
        if self.backend.load_model(config_path, weight_path):
            logger.debug(f"loading model from {config_path}")
            self.model = self.backend.model
            self.digest = self.fetch_digest(weight_path)
            logger.debug(f"loaded model digest = {self.digest}")
            return True
        logger.debug(f"model files does not exist at {config_path} and {weight_path}")
        return False

    def save(self, config_path, weight_path):
        logger.debug(f"save model to {config_path}")
        self.backend.save_model(config_path, weight_path)
        self.model = self.backend.model
        self.digest = self.fetch_digest(weight_path)
        logger.debug(f"saved model digest {self.digest}")

    def predict_on_batch(self, data: np.ndarray):
        return self.backend.predict_batch(data)

    def configure_training(
        self,
        optimizer_name: str,
        learning_rate: float,
        momentum: float = 0.0,
        loss_weights=(1.0, 1.0),
    ):
        self.backend.configure_training(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            loss_weights=tuple(loss_weights),
        )
        self.model = self.backend.model

    def set_learning_rate(self, learning_rate: float):
        self.backend.set_learning_rate(learning_rate)

    def train(
        self,
        state_ary: np.ndarray,
        policy_ary: np.ndarray,
        value_ary: np.ndarray,
        batch_size: int,
        epochs: int,
        shuffle: bool = True,
        validation_split: float = 0.0,
    ):
        metrics = self.backend.train(
            state_ary=state_ary,
            policy_ary=policy_ary,
            value_ary=value_ary,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            validation_split=validation_split,
        )
        self.model = self.backend.model
        return metrics

    def get_pipes(self, num=1, api=None, need_reload=True):
        if self.api is None:
            self.api = CChessModelAPI(self.config, self)
            self.api.start(need_reload)
        return self.api.get_pipe(need_reload)

    def close_pipes(self):
        if self.api is not None:
            self.api.close()
            self.api = None
