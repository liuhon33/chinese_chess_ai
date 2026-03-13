import gc
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from logging import getLogger
from multiprocessing import Manager
from random import random
from threading import Thread
from time import sleep, time

import numpy as np

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.config import Config
from cchess_alphazero.environment.lookup_tables import ActionLabelsRed, flip_policy
from cchess_alphazero.lib.cluster_helper import (
    build_cluster_play_data_path,
    cluster_enabled,
    safe_write_play_data_enabled,
    write_json_atomic,
)
from cchess_alphazero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from cchess_alphazero.lib.model_helper import build_fresh_best_model, load_model_weight
from cchess_alphazero.lib.terminal_logger import (
    emit_terminal_log,
    should_log_buffer_flush,
    should_log_game_summary,
    should_log_moves,
)
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import upload_file

logger = getLogger(__name__)


def load_model(config, config_file=None):
    use_history = False
    model = CChessModel(config)
    weight_path = config.resource.model_best_weight_path
    if not config_file:
        config_path = config.resource.model_best_config_path
        use_history = False
    else:
        config_path = os.path.join(config.resource.model_dir, config_file)
    try:
        if config.opts.new or not load_model_weight(model, config_path, weight_path):
            build_fresh_best_model(model)
            use_history = True
    except Exception as e:
        logger.info(f"Exception {e}, building a fresh BestModel instead")
        build_fresh_best_model(model)
        use_history = True
    return model, use_history


def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    current_model, use_history = load_model(config)
    manager = Manager()
    cur_pipes = manager.list([current_model.get_pipes() for _ in range(config.play.max_processes)])
    with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
        futures = []
        for i in range(config.play.max_processes):
            play_worker = SelfPlayWorker(config, cur_pipes, i, use_history)
            logger.debug("Initialize selfplay worker")
            futures.append(executor.submit(play_worker.start))


class SelfPlayWorker:
    def __init__(self, config: Config, pipes=None, pid=None, use_history=False):
        self.config = config
        self.player = None
        self.cur_pipes = pipes
        self.id = pid
        self.buffer = []
        self.pid = os.getpid()
        self.use_history = use_history
        self.worker_label = self.config.cluster.worker_id or self.id

    def start(self):
        self.pid = os.getpid()
        ran = self.config.play.max_processes if self.config.play.max_processes > 5 else self.config.play.max_processes * 2
        sleep((self.pid % ran) * 10)
        logger.debug(f"Selfplay#Start Process index = {self.id}, pid = {self.pid}")
        emit_terminal_log(self.config, "self", "worker started", worker_id=self.worker_label, pid=self.pid)

        idx = 1
        self.buffer = []

        while True:
            start_time = time()
            value, turns, state, store = self.start_game(idx)
            end_time = time()
            logger.debug(
                f"Process {self.pid}-{self.id} play game {idx} time={(end_time - start_time):.1f} sec, "
                f"turn={turns / 2}, winner = {value:.2f} (1 = red, -1 = black, 0 draw)"
            )
            if should_log_game_summary(self.config):
                emit_terminal_log(
                    self.config,
                    "self",
                    f"game={idx} duration={end_time - start_time:.1f}s turns={turns / 2:.1f} winner={value:.2f} stored={store}",
                    worker_id=self.worker_label,
                    pid=self.pid,
                )
            if turns <= 10:
                senv.render(state)
            if store:
                idx += 1
            sleep(random())

    def start_game(self, idx):
        pipes = self.cur_pipes.pop()
        search_tree = defaultdict(VisitState)

        if not self.config.play.share_mtcs_info_in_self_play or idx % self.config.play.reset_mtcs_info_per_game == 0:
            search_tree = defaultdict(VisitState)

        enable_resign = random() > self.config.play.enable_resign_rate
        self.player = CChessPlayer(
            self.config,
            search_tree=search_tree,
            pipes=pipes,
            enable_resign=enable_resign,
            debugging=False,
            use_history=self.use_history,
        )

        state = senv.INIT_STATE
        history = [state]
        value = 0
        turns = 0
        game_over = False
        final_move = None
        no_eat_count = 0
        check = False
        no_act = []
        increase_temp = False

        while not game_over:
            start_time = time()
            action, policy = self.player.action(state, turns, no_act, increase_temp=increase_temp)
            end_time = time()
            if action is None:
                logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned!")
                if should_log_moves(self.config):
                    emit_terminal_log(
                        self.config,
                        "self",
                        f"turn={turns} side={turns % 2} resigned",
                        worker_id=self.worker_label,
                        pid=self.pid,
                    )
                value = -1
                break
            if should_log_moves(self.config):
                emit_terminal_log(
                    self.config,
                    "self",
                    f"turn={turns} side={turns % 2} action={action} move_time={end_time - start_time:.1f}s",
                    worker_id=self.worker_label,
                    pid=self.pid,
                )
            history.append(action)
            try:
                state, no_eat = senv.new_step(state, action)
            except Exception as e:
                logger.error(f"{e}, no_act = {no_act}, policy = {policy}")
                game_over = True
                value = 0
                break
            turns += 1
            if no_eat:
                no_eat_count += 1
            else:
                no_eat_count = 0
            history.append(state)

            if no_eat_count >= 120 or turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = 0
            else:
                game_over, value, final_move, check = senv.done(state, need_check=True)
                if not game_over and not senv.has_attack_chessman(state):
                    logger.info(f"双方无进攻子力，作和。state = {state}")
                    game_over = True
                    value = 0
                increase_temp = False
                no_act = []
                if not game_over and not check and state in history[:-1]:
                    free_move = defaultdict(int)
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            if senv.will_check_or_catch(state, history[i + 1]):
                                no_act.append(history[i + 1])
                            elif not senv.be_catched(state, history[i + 1]):
                                increase_temp = True
                                free_move[state] += 1
                                if free_move[state] >= 3:
                                    game_over = True
                                    value = 0
                                    logger.info("闲着循环三次，作和棋处理")
                                    break

        if final_move:
            history.append(final_move)
            state = senv.step(state, final_move)
            turns += 1
            value = -value
            history.append(state)

        self.player.close()
        del search_tree
        del self.player
        gc.collect()
        if turns % 2 == 1:
            value = -value

        original_value = value
        if turns < 10:
            store = random() > 0.9
        else:
            store = True

        if store:
            data = [history[0]]
            for i in range(turns):
                k = i * 2
                data.append([history[k + 1], value])
                value = -value
            self.save_play_data(idx, data)

        self.cur_pipes.append(pipes)
        self.remove_play_data()
        return original_value, turns, state, store

    def save_play_data(self, idx, data):
        self.buffer += data
        if not idx % self.config.play_data.nb_game_in_file == 0:
            return

        rc = self.config.resource
        if cluster_enabled(self.config):
            path = build_cluster_play_data_path(self.config, pid=self.pid)
            filename = os.path.basename(path)
        else:
            utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
            game_id = bj_dt.strftime("%Y%m%d-%H%M%S.%f")
            filename = rc.play_data_filename_tmpl % game_id
            path = os.path.join(rc.play_data_dir, filename)
        logger.info(f"Process {self.pid} save play data to {path}")
        if should_log_buffer_flush(self.config):
            emit_terminal_log(
                self.config,
                "self",
                f"flush path={path} records={len(self.buffer)}",
                worker_id=self.worker_label,
                pid=self.pid,
            )
        if safe_write_play_data_enabled(self.config):
            write_json_atomic(path, self.buffer)
        else:
            write_game_data_to_file(path, self.buffer)
        if self.config.internet.distributed:
            upload_worker = Thread(target=self.upload_play_data, args=(path, filename), name="upload_worker")
            upload_worker.daemon = True
            upload_worker.start()
        self.buffer = []

    def upload_play_data(self, path, filename):
        digest = CChessModel.fetch_digest(self.config.resource.model_best_weight_path)
        data = {'digest': digest, 'username': self.config.internet.username, 'version': '2.4'}
        response = upload_file(self.config.internet.upload_url, path, filename, data, rm=False)
        if response is not None and response['status'] == 0:
            logger.info(f"Upload play data {filename} finished.")
        else:
            logger.error(f'Upload play data {filename} failed. {response.msg if response is not None else None}')

    def remove_play_data(self):
        if cluster_enabled(self.config):
            return
        files = get_game_data_filenames(self.config.resource)
        if len(files) < self.config.play_data.max_file_num:
            return
        try:
            for i in range(len(files) - self.config.play_data.max_file_num):
                os.remove(files[i])
        except Exception:
            pass

    def build_policy(self, action, flip):
        labels_n = len(ActionLabelsRed)
        move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
        policy = np.zeros(labels_n)
        policy[move_lookup[action]] = 1
        if flip:
            policy = flip_policy(policy)
        return list(policy)
