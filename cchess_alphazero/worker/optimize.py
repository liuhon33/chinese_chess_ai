import gc
import os
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from logging import getLogger
from random import shuffle
from time import sleep

import numpy as np

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy
from cchess_alphazero.lib.data_helper import get_game_data_filenames, read_game_data_from_file
from cchess_alphazero.lib.model_helper import (
    build_fresh_best_model,
    is_next_generation_model_fresh,
    load_best_model_weight,
    need_to_reload_best_model_weight,
    next_generation_model_exists,
    save_as_next_generation_model,
)
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.training_monitor import save_training_state

logger = getLogger(__name__)


def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = deque(), deque(), deque()
        self.filenames = []
        self.count = 0

    @property
    def polling_interval(self):
        return max(1.0, float(getattr(self.config.trainer, "polling_interval", 300)))

    def start(self):
        self.model = self.load_model()
        self.training()

    def training(self):
        self.compile_model()
        total_steps = self.config.trainer.start_total_steps
        last_file = None
        waiting_for_data = False
        waiting_for_candidate = False

        while True:
            if self.try_reload_model():
                logger.info("Reloaded BestModel before the next optimize cycle.")
                self.compile_model()

            if self.has_pending_candidate():
                if not waiting_for_candidate:
                    logger.info(
                        "Candidate model is awaiting evaluation; optimizer will poll again in %.1f seconds.",
                        self.polling_interval,
                    )
                    waiting_for_candidate = True
                sleep(self.polling_interval)
                continue
            waiting_for_candidate = False

            files = get_game_data_filenames(self.config.resource)
            if not self.has_enough_new_data(files, last_file):
                if not waiting_for_data:
                    last_file_label = os.path.basename(last_file) if last_file else "the start of training"
                    logger.info(
                        "Waiting for more play data; need at least %s total file(s) and %s unseen file(s) after %s. Found %s file(s). Polling again in %.1f seconds.",
                        self.config.trainer.min_games_to_begin_learn,
                        self.config.trainer.min_games_to_begin_learn,
                        last_file_label,
                        len(files),
                        self.polling_interval,
                    )
                    waiting_for_data = True
                sleep(self.polling_interval)
                continue
            waiting_for_data = False

            files = self.select_files_to_train(files, last_file)
            if not files:
                logger.info("No unseen play-data files were selected; polling again in %.1f seconds.", self.polling_interval)
                sleep(self.polling_interval)
                continue

            last_file = files[-1]
            logger.info("Last file = %s", last_file)
            logger.debug("Start training %s files", len(files))
            self.filenames = deque(files)
            shuffle(self.filenames)
            self.fill_queue()
            self.update_learning_rate(total_steps)
            sample_count = len(self.dataset[0])
            if sample_count >= self.config.trainer.batch_size:
                steps = self.train_epoch(self.config.trainer.epoch_to_checkpoint)
                total_steps += steps
                candidate_path = self.publish_candidate_model()
                save_training_state(self.config, total_steps)
                self.update_learning_rate(total_steps)
                self.count += 1
                logger.info(
                    "Optimize cycle complete: trained %s file(s), %s sample(s), %s optimizer step(s), total_steps=%s, candidate=%s",
                    len(files),
                    sample_count,
                    steps,
                    total_steps,
                    candidate_path,
                )
                self.clear_dataset()
                self.backup_play_data(files)
            else:
                logger.warning(
                    "Collected %s sample(s) from %s file(s), below batch size %s; keeping the data in memory for the next pass.",
                    len(self.dataset[0]),
                    len(files),
                    self.config.trainer.batch_size,
                )

    def has_enough_new_data(self, files, last_file):
        required = self.config.trainer.min_games_to_begin_learn
        if len(files) < required:
            return False
        if last_file is None or last_file not in files:
            return True
        next_idx = files.index(last_file) + 1
        return len(files) - next_idx >= required

    def select_files_to_train(self, files, last_file):
        if last_file is not None and last_file in files:
            start_idx = files.index(last_file) + 1
            return files[start_idx:start_idx + self.config.trainer.load_step]
        return files[:self.config.trainer.load_step]

    def has_pending_candidate(self):
        if not next_generation_model_exists(self.config):
            return False
        if not self.config.opts.new:
            return True
        return is_next_generation_model_fresh(self.config)

    def clear_dataset(self):
        state_ary, policy_ary, value_ary = self.dataset
        state_ary.clear()
        policy_ary.clear()
        value_ary.clear()
        del self.dataset, state_ary, policy_ary, value_ary
        gc.collect()
        self.dataset = deque(), deque(), deque()

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
            logger.info(f"Training metrics: {metrics}")
        return (state_ary.shape[0] // tc.batch_size) * epochs

    def compile_model(self):
        self.model.configure_training(
            optimizer_name="sgd",
            learning_rate=0.02,
            momentum=self.config.trainer.momentum,
            loss_weights=self.config.trainer.loss_weights,
        )

    def update_learning_rate(self, total_steps):
        lr = self.decide_learning_rate(total_steps)
        if lr:
            self.model.set_learning_rate(lr)
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def fill_queue(self):
        futures = deque()
        n = len(self.filenames)
        with ProcessPoolExecutor(max_workers=self.config.trainer.cleaning_processes) as executor:
            for _ in range(self.config.trainer.cleaning_processes):
                if len(self.filenames) == 0:
                    break
                filename = self.filenames.pop()
                futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))
            while futures and len(self.dataset[0]) < self.config.trainer.dataset_size:
                loaded_tuple = futures.popleft().result()
                if loaded_tuple is not None:
                    for x, y in zip(self.dataset, loaded_tuple):
                        x.extend(y)
                m = len(self.filenames)
                if m > 0:
                    if (n - m) % 1000 == 0:
                        logger.info(f"Reading {n - m} files")
                    filename = self.filenames.pop()
                    futures.append(executor.submit(load_data_from_file, filename, self.config.opts.has_history))

    def collect_all_loaded_data(self):
        state_ary, policy_ary, value_ary = self.dataset
        return (
            np.asarray(state_ary, dtype=np.float32),
            np.asarray(policy_ary, dtype=np.float32),
            np.asarray(value_ary, dtype=np.float32),
        )

    def load_model(self):
        model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(model):
            build_fresh_best_model(model)
        return model

    def publish_candidate_model(self):
        logger.info("Publishing candidate model for evaluation.")
        save_as_next_generation_model(self.model)
        return self.config.resource.next_generation_weight_path

    def decide_learning_rate(self, total_steps):
        ret = None
        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def try_reload_model(self):
        logger.debug("check model")
        if need_to_reload_best_model_weight(self.model):
            load_best_model_weight(self.model)
            return True
        return False

    def backup_play_data(self, files):
        backup_folder = os.path.join(self.config.resource.data_dir, "trained")
        cnt = 0
        if not os.path.exists(backup_folder):
            os.makedirs(backup_folder)
        for path in files:
            try:
                shutil.move(path, backup_folder)
            except Exception:
                cnt += 1
        logger.info(f"backup {len(files)} files, {cnt} empty files")


def load_data_from_file(filename, use_history=False):
    try:
        data = read_game_data_from_file(filename)
    except Exception as e:
        logger.error(f"Error when loading data {e}")
        os.remove(filename)
        return None
    if data is None:
        return None
    return expanding_data(data, use_history)


def expanding_data(data, use_history=False):
    chunks = split_data_by_game(data)
    if not chunks:
        return None

    expanded_chunks = []
    for chunk in chunks:
        expanded = expand_game_data(chunk, use_history)
        if expanded is not None and len(expanded[0]) > 0:
            expanded_chunks.append(expanded)

    if not expanded_chunks:
        return None
    if len(expanded_chunks) == 1:
        return expanded_chunks[0]

    return tuple(np.concatenate(items, axis=0) for items in zip(*expanded_chunks))


def split_data_by_game(data):
    if not data:
        logger.error("Expand data error: empty play data file")
        return []

    chunks = []
    current = None
    for item in data:
        if isinstance(item, str):
            if current is not None and len(current) > 1:
                chunks.append(current)
            elif current is not None:
                logger.warning("Skipping empty game chunk with state only: %s", current[0])
            current = [item]
        else:
            if current is None:
                logger.error("Expand data error: move record encountered before initial state, data = %s", data)
                return []
            current.append(item)

    if current is not None and len(current) > 1:
        chunks.append(current)
    elif current is not None:
        logger.warning("Skipping empty trailing game chunk with state only: %s", current[0])

    return chunks


def expand_game_data(data, use_history=False):
    state = data[0]
    real_data = []
    if use_history:
        history = [state]
    else:
        history = None
    for item in data[1:]:
        action = item[0]
        value = item[1]
        try:
            policy = build_policy(action, flip=False)
        except Exception as e:
            logger.error(f"Expand data error {e}, item = {item}, data = {data}, state = {state}")
            return None
        real_data.append([state, policy, value])
        state = senv.step(state, action)
        if use_history:
            history.append(action)
            history.append(state)
    return convert_to_training_data(real_data, history)


def convert_to_training_data(data, history):
    state_list = []
    policy_list = []
    value_list = []

    for index, (state, policy, value) in enumerate(data):
        if history is None:
            state_planes = senv.state_to_planes(state)
        else:
            state_planes = senv.state_history_to_planes(state, history[0:index * 2 + 1])
        state_list.append(state_planes)
        policy_list.append(policy)
        value_list.append(value)

    return (
        np.asarray(state_list, dtype=np.float32),
        np.asarray(policy_list, dtype=np.float32),
        np.asarray(value_list, dtype=np.float32),
    )


def build_policy(action, flip):
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)
    policy[move_lookup[action]] = 1
    if flip:
        policy = flip_policy(policy)
    return list(policy)
