import json

import numpy as np

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, Winner, flip_policy
from cchess_alphazero.lib.model_helper import load_sl_best_model_weight, save_as_sl_best_model
from cchess_alphazero.lib.tf_util import set_session_config
from logging import getLogger
from time import time

logger = getLogger(__name__)


def start(config: Config, skip):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return SupervisedWorker(config).start(skip)


class SupervisedWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = [], [], []
        self.buffer = []
        self.games = None

    def start(self, skip=0):
        self.model = self.load_model()
        with open(self.config.resource.sl_onegreen, 'r') as f:
            self.games = json.load(f)
        self.training(skip)

    def training(self, skip=0):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        logger.info(f"Start training, game count = {len(self.games)}, step = {self.config.trainer.sl_game_step} games, skip = {skip}")

        for i in range(skip, len(self.games), self.config.trainer.sl_game_step):
            games = self.games[i:i + self.config.trainer.sl_game_step]
            self.fill_queue(games)
            if len(self.dataset[0]) > self.config.trainer.batch_size:
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                self.save_current_model()
                for bucket in self.dataset:
                    bucket.clear()
                logger.debug(f"total steps = {total_steps}")

    def train_epoch(self, epochs):
        tc = self.config.trainer
        state_ary, policy_ary, value_ary = self.collect_all_loaded_data()
        metrics = self.model.train(
            state_ary=state_ary,
            policy_ary=policy_ary,
            value_ary=value_ary,
            batch_size=tc.batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.02,
        )
        if metrics:
            logger.info(f"Onegreen SL metrics: {metrics}")
        return (state_ary.shape[0] // tc.batch_size) * epochs

    def compile_model(self):
        self.model.configure_training(
            optimizer_name="adam",
            learning_rate=0.003,
            loss_weights=self.config.trainer.loss_weights,
        )

    def fill_queue(self, games):
        _tuple = self.generate_game_data(games)
        if _tuple is not None:
            for x, y in zip(self.dataset, _tuple):
                x.extend(y)

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset
        return (
            np.asarray(state_ary, dtype=np.float32),
            np.asarray(policy_ary, dtype=np.float32),
            np.asarray(value_ary, dtype=np.float32),
        )

    def load_model(self):
        model = CChessModel(self.config)
        if self.config.opts.new or not load_sl_best_model_weight(model):
            model.build()
            save_as_sl_best_model(model)
        return model

    def save_current_model(self):
        logger.debug("Save best sl model")
        save_as_sl_best_model(self.model)

    def generate_game_data(self, games):
        self.buffer = []
        start_time = time()
        cnt = 0
        for idx, game in enumerate(games):
            winner = Winner.draw
            if game['result'] == '红胜' or '胜' in game['title']:
                winner = Winner.red
            elif game['result'] == '黑胜' or '负' in game['title']:
                winner = Winner.black
            v = self.load_game(game['init'], game['move_list'], winner, idx, game['title'], game['url'])
            if v == 1 or v == -1:
                cnt += 1
        end_time = time()
        logger.debug(f"Loading {len(games)} games, time: {end_time - start_time}s, end games = {cnt}")
        return self.convert_to_training_data()

    def load_game(self, init, move_list, winner, idx, title, url):
        turns = 0
        env = CChessEnv(self.config).reset(init)
        red_moves = []
        black_moves = []
        moves = [move_list[i:i + 4] for i in range(len(move_list)) if i % 4 == 0]

        for move in moves:
            action = senv.parse_onegreen_move(move)
            try:
                if turns % 2 == 0:
                    red_moves.append([env.observation, self.build_policy(action, flip=False)])
                else:
                    black_moves.append([env.observation, self.build_policy(action, flip=True)])
                env.step(action)
            except Exception:
                logger.error(
                    f"Invalid Action: idx = {idx}, action = {action}, turns = {turns}, moves = {moves}, "
                    f"winner = {winner}, init = {init}, title: {title}, url: {url}"
                )
                return None
            turns += 1

        if winner == Winner.red:
            red_win = 1
        elif winner == Winner.black:
            red_win = -1
        else:
            red_win = senv.evaluate(env.get_state())
            if not env.red_to_move:
                red_win = -red_win

        for move in red_moves:
            move += [red_win]
        for move in black_moves:
            move += [-red_win]

        data = []
        for i in range(len(red_moves)):
            data.append(red_moves[i])
            if i < len(black_moves):
                data.append(black_moves[i])
        self.buffer += data
        return red_win

    def build_policy(self, action, flip):
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)
        policy[move_lookup[action]] = 1
        if flip:
            policy = flip_policy(policy)
        return policy

    def convert_to_training_data(self):
        state_list = []
        policy_list = []
        value_list = []
        env = CChessEnv()

        for state_fen, policy, value in self.buffer:
            state_list.append(env.fen_to_planes(state_fen))
            policy_list.append(policy)
            value_list.append(value)

        return (
            np.asarray(state_list, dtype=np.float32),
            np.asarray(policy_list, dtype=np.float32),
            np.asarray(value_list, dtype=np.float32),
        )
