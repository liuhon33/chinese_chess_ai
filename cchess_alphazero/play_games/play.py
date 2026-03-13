import math
import os
import sys
import pygame
import random
import time
import copy
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from threading import Lock, Thread
from time import sleep
from datetime import datetime

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)
PLAY_GAMES_DIR = Path(__file__).resolve().parent
IMAGES_DIR = PLAY_GAMES_DIR / "images"
DEFAULT_FONT_PATH = PLAY_GAMES_DIR / "PingFang.ttc"
CHINESE_FONT_SAMPLE = "着法记录当前局势评估搜索次数动作价值先验概率"
CHINESE_FONT_FALLBACKS = [
    "Microsoft YaHei",
    "SimHei",
    "PingFang SC",
    "PingFang TC",
    "Noto Sans CJK SC",
    "Noto Sans CJK TC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]
PIECE_STYLE = 'WOOD'
DEFAULT_BOARD_WIDTH = 521
DEFAULT_BOARD_HEIGHT = 577
DEFAULT_WINDOW_WIDTH = 720
DEFAULT_WINDOW_HEIGHT = 577
MIN_PANEL_WIDTH = DEFAULT_WINDOW_WIDTH - DEFAULT_BOARD_WIDTH


@dataclass(frozen=True)
class ModelOption:
    key: str
    name: str
    label: str
    config_path: str
    weight_path: str
    allow_background_reload: bool = False
    build_if_missing: bool = False


@dataclass
class ModelBinding:
    option: ModelOption
    model: CChessModel
    pipe: object
    ai: CChessPlayer


def orient_board_coordinate(col_num, row_num, invert=False):
    if invert:
        return 8 - col_num, 9 - row_num
    return col_num, row_num


def board_to_sprite_rect(col_num, row_num, w, h, invert=False):
    display_col_num, display_row_num = orient_board_coordinate(col_num, row_num, invert=invert)
    return Rect(display_col_num * w, (9 - display_row_num) * h, w, h)


def start(config: Config, human_move_first=True):
    global PIECE_STYLE
    PIECE_STYLE = config.opts.piece_style
    play = PlayWithHuman(config)
    play.start(human_move_first)


class PlayWithHuman:
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.winstyle = getattr(pygame, "RESIZABLE", 0)
        self.chessmans = None
        self.human_move_first = True
        self.width = DEFAULT_BOARD_WIDTH
        self.height = DEFAULT_BOARD_HEIGHT
        self.min_panel_width = MIN_PANEL_WIDTH
        self.screen_width = DEFAULT_WINDOW_WIDTH
        self.window_height = DEFAULT_WINDOW_HEIGHT
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 15
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0
        self.mcts_moves = {}
        self.analysis_arrows = []
        self.history = []
        if self.config.opts.bg_style == 'WOOD':
            self.chessman_w += 1
            self.chessman_h += 1
        self.board_rect = Rect(0, 0, self.width, self.height)
        self.board_click_tolerance = int(round(min(self.chessman_w, self.chessman_h) * 0.65))
        self.piece_click_tolerance = int(round(min(self.chessman_w, self.chessman_h) * 0.60))
        self.gui_debug = bool(getattr(self.config.opts, "debug_gui", False))
        self.analysis_only = bool(getattr(self.config.opts, "analysis_only", False))
        self.invert_board = bool(getattr(self.config.opts, "invert", False))
        self.preferred_font_path = self.resolve_font_path()
        self.font_cache = {}
        self.font_warning_cache = set()
        self.analysis_lock = Lock()
        self.analysis_request_id = 0
        self.binding_lock = Lock()
        self.binding_generation = 0
        self.worker_binding = None
        self.model_binding = None
        self.retired_bindings = []
        self.model_options = []
        self.active_model_option = None
        self.model_dropdown_open = False
        self.model_dropdown_rect = Rect(10, 350, self.screen_width - self.width - 20, 26)
        self.model_dropdown_item_height = 26
        self.undo_button_rect = Rect(10, 382, 84, 24)
        self.redo_button_rect = Rect(105, 382, 84, 24)
        self.redo_stack = []
        self.shutdown_requested = False
        requested_width = getattr(self.config.opts, "window_width", None) or DEFAULT_WINDOW_WIDTH
        requested_height = getattr(self.config.opts, "window_height", None) or DEFAULT_WINDOW_HEIGHT
        self.update_window_layout(requested_width, requested_height)

    def update_window_layout(self, screen_width=None, window_height=None):
        if screen_width is None:
            screen_width = self.screen_width
        if window_height is None:
            window_height = self.window_height
        self.screen_width = max(int(screen_width), self.width + self.min_panel_width)
        self.window_height = max(int(window_height), self.height)
        panel_width = self.screen_width - self.width
        button_width = 60
        button_height = 24
        button_spacing = 6
        right_margin = 10
        top_margin = 8
        redo_x = max(10, panel_width - button_width - right_margin)
        undo_x = max(10, redo_x - button_width - button_spacing)
        self.model_dropdown_rect = Rect(10, 350, max(panel_width - 20, 20), 26)
        self.undo_button_rect = Rect(undo_x, top_margin, button_width, button_height)
        self.redo_button_rect = Rect(redo_x, top_margin, button_width, button_height)
        return self.screen_width, self.window_height

    def create_board_background(self):
        bgdtile = load_image(f'{self.config.opts.bg_style}.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.window_height])
        board_background.fill((255, 255, 255))
        board_background.blit(bgdtile, (0, 0))
        return board_background

    def create_widget_background(self):
        widget_background = pygame.Surface([self.screen_width - self.width, self.window_height])
        widget_background.fill((255, 255, 255))
        return widget_background

    def recreate_display(self):
        bestdepth = pygame.display.mode_ok([self.screen_width, self.window_height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.window_height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋Zero")
        return screen, self.create_board_background(), self.create_widget_background()

    def resolve_font_path(self):
        configured_font = getattr(self.config.resource, "font_path", None)
        candidate_paths = []
        if configured_font:
            candidate_paths.append(Path(configured_font))
        candidate_paths.append(DEFAULT_FONT_PATH)

        seen = set()
        for candidate in candidate_paths:
            candidate = candidate.expanduser()
            if not candidate.is_absolute():
                candidate = PLAY_GAMES_DIR / candidate
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

        checked = ", ".join(str(path) for path in seen)
        logger.warning(
            "Chinese font asset missing; checked %s. Falling back to system fonts.",
            checked,
        )
        return None

    def model_location_label(self, weight_path):
        parent = Path(weight_path).resolve().parent
        project_dir = Path(self.config.resource.project_dir).resolve()
        try:
            return str(parent.relative_to(project_dir))
        except ValueError:
            return str(parent)

    def build_model_option(
        self,
        key,
        name,
        config_path,
        weight_path,
        allow_background_reload=False,
        build_if_missing=False,
    ):
        if not build_if_missing and not (os.path.exists(config_path) and os.path.exists(weight_path)):
            return None
        label = f"{name} [{self.model_location_label(weight_path)}]"
        return ModelOption(
            key=key,
            name=name,
            label=label,
            config_path=config_path,
            weight_path=weight_path,
            allow_background_reload=allow_background_reload,
            build_if_missing=build_if_missing,
        )

    def iter_model_roots(self):
        project_dir = Path(self.config.resource.project_dir).resolve()
        current_model_dir = Path(self.config.resource.model_dir).resolve()
        candidate_roots = [
            current_model_dir,
            project_dir / "data" / "model",
            project_dir / "mydata" / "model",
            project_dir / "testdata" / "model",
            project_dir / "validation_local_torch" / "model",
        ]
        roots = []
        seen = set()
        for root in candidate_roots:
            root = root.resolve()
            if root in seen:
                continue
            seen.add(root)
            if root == current_model_dir or root.exists():
                roots.append(root)
        return roots

    def discover_model_options(self):
        current_root = Path(self.config.resource.model_dir).resolve()
        options = []
        seen = set()
        for model_root in self.iter_model_roots():
            root_name = model_root.parent.name
            root_is_current = model_root == current_root
            candidates = [
                self.build_model_option(
                    f"{root_name}:best",
                    f"Best model ({root_name})",
                    str(model_root / "model_best_config.json"),
                    str(model_root / "model_best_weight.h5"),
                    allow_background_reload=root_is_current,
                    build_if_missing=root_is_current,
                ),
                self.build_model_option(
                    f"{root_name}:next_generation",
                    f"Next generation ({root_name})",
                    str(model_root / "next_generation" / "next_generation_config.json"),
                    str(model_root / "next_generation" / "next_generation_weight.h5"),
                ),
                self.build_model_option(
                    f"{root_name}:sl_best",
                    f"SL best ({root_name})",
                    str(model_root / "sl_best_config.json"),
                    str(model_root / "sl_best_weight.h5"),
                ),
                self.build_model_option(
                    f"{root_name}:rival",
                    f"Rival ({root_name})",
                    str(model_root / "rival_config.json"),
                    str(model_root / "rival_weight.h5"),
                ),
            ]
            for option in candidates:
                if option is None:
                    continue
                pair = (option.config_path, option.weight_path)
                if pair in seen:
                    continue
                seen.add(pair)
                options.append(option)
        return options

    def create_model_binding(self, option: ModelOption):
        model = CChessModel(self.config)
        try:
            if option.build_if_missing:
                loaded = False
                if not self.config.opts.new:
                    loaded = model.load(option.config_path, option.weight_path)
                if not loaded:
                    model.build()
            elif not model.load(option.config_path, option.weight_path):
                raise FileNotFoundError(
                    f"Model files not found for {option.label}: {option.config_path}, {option.weight_path}"
                )
            pipe = model.get_pipes(need_reload=option.allow_background_reload)
            ai = CChessPlayer(
                self.config,
                search_tree=defaultdict(VisitState),
                pipes=pipe,
                enable_resign=True,
                debugging=True,
            )
            return ModelBinding(option=option, model=model, pipe=pipe, ai=ai)
        except Exception:
            model.close_pipes()
            raise

    def close_binding(self, binding: ModelBinding):
        if binding is None:
            return
        try:
            binding.ai.close(wait=False)
        except Exception as exc:
            logger.warning("Failed to close AI player for %s: %s", binding.option.label, exc)
        try:
            binding.model.close_pipes()
        except Exception as exc:
            logger.warning("Failed to close model pipes for %s: %s", binding.option.label, exc)

    def cleanup_retired_bindings(self):
        to_close = []
        with self.binding_lock:
            active_worker_binding = self.worker_binding
            remaining = []
            for binding in self.retired_bindings:
                if binding is active_worker_binding:
                    remaining.append(binding)
                else:
                    to_close.append(binding)
            self.retired_bindings = remaining
        for binding in to_close:
            self.close_binding(binding)

    def get_active_binding(self):
        with self.binding_lock:
            return self.binding_generation, self.model_binding

    def mark_worker_binding(self, binding):
        with self.binding_lock:
            self.worker_binding = binding

    def clear_worker_binding(self, binding):
        with self.binding_lock:
            if self.worker_binding is binding:
                self.worker_binding = None

    def activate_model_option(self, option: ModelOption):
        if option is None:
            return False
        try:
            new_binding = self.create_model_binding(option)
        except Exception:
            logger.exception("Failed to activate GUI model option %s", option.label)
            return False
        with self.binding_lock:
            previous_binding = self.model_binding
            self.model_binding = new_binding
            self.model = new_binding.model
            self.pipe = new_binding.pipe
            self.ai = new_binding.ai
            self.active_model_option = option
            self.binding_generation += 1
            if previous_binding is not None:
                self.retired_bindings.append(previous_binding)
        self.model_dropdown_open = False
        logger.info("Active GUI model switched to %s", option.label)
        self.request_analysis()
        self.cleanup_retired_bindings()
        return True

    def shutdown_model_bindings(self):
        with self.binding_lock:
            bindings = []
            seen = set()
            for binding in [self.model_binding, *self.retired_bindings]:
                if binding is None:
                    continue
                binding_id = id(binding)
                if binding_id in seen:
                    continue
                seen.add(binding_id)
                bindings.append(binding)
            self.model_binding = None
            self.retired_bindings = []
            self.worker_binding = None
            self.model = None
            self.pipe = None
            self.ai = None
            self.active_model_option = None
        for binding in bindings:
            self.close_binding(binding)

    def font_supports_text(self, font, text=CHINESE_FONT_SAMPLE):
        try:
            metrics = font.metrics(text)
        except pygame.error:
            return False
        return bool(metrics) and all(metric is not None for metric in metrics)

    def log_font_warning_once(self, key, message, *args):
        if key in self.font_warning_cache:
            return
        self.font_warning_cache.add(key)
        logger.warning(message, *args)

    def load_font_with_fallback(self, size, prefer_chinese=True):
        cache_key = (size, prefer_chinese)
        if cache_key in self.font_cache:
            return self.font_cache[cache_key]

        if self.preferred_font_path is not None:
            try:
                font = pygame.font.Font(str(self.preferred_font_path), size)
                if not prefer_chinese or self.font_supports_text(font):
                    self.font_cache[cache_key] = font
                    return font
                self.log_font_warning_once(
                    ("font-glyphs", str(self.preferred_font_path)),
                    "Bundled font %s does not render required Chinese glyphs. Falling back to system fonts.",
                    self.preferred_font_path,
                )
            except (FileNotFoundError, OSError, pygame.error) as exc:
                self.log_font_warning_once(
                    ("font-file", str(self.preferred_font_path)),
                    "Failed to load bundled font %s (%s). Falling back to system fonts.",
                    self.preferred_font_path,
                    exc,
                )
                self.preferred_font_path = None

        if prefer_chinese:
            for font_name in CHINESE_FONT_FALLBACKS:
                try:
                    font = pygame.font.SysFont(font_name, size)
                except pygame.error:
                    continue
                if self.font_supports_text(font):
                    self.log_font_warning_once(
                        ("font-system", font_name),
                        "Using system font fallback '%s' for Chinese text.",
                        font_name,
                    )
                    self.font_cache[cache_key] = font
                    return font

            self.log_font_warning_once(
                ("font-default", size),
                "No Chinese-capable bundled/system font found; using pygame default font.",
            )

        font = pygame.font.Font(None, size)
        self.font_cache[cache_key] = font
        return font

    def board_to_screen(self, col_num, row_num):
        display_col_num, display_row_num = orient_board_coordinate(
            col_num,
            row_num,
            invert=self.invert_board,
        )
        return (
            display_col_num * self.chessman_w + self.chessman_w / 2,
            (9 - display_row_num) * self.chessman_h + self.chessman_h / 2,
        )

    def move_to_board_points(self, move, red_to_move):
        display_move = move if red_to_move else flip_move(move)
        x0, y0, x1, y1 = (int(display_move[i]) for i in range(4))
        return (x0, y0), (x1, y1)

    def distance_to_piece_center(self, sprite, screen_x, screen_y):
        center_x, center_y = sprite.rect.center
        return math.hypot(screen_x - center_x, screen_y - center_y)

    def find_sprite_near_point(self, screen_x, screen_y):
        nearest_sprite = None
        nearest_distance = None
        for sprite in self.chessmans:
            distance = self.distance_to_piece_center(sprite, screen_x, screen_y)
            if distance <= self.piece_click_tolerance and (
                nearest_distance is None or distance < nearest_distance
            ):
                nearest_sprite = sprite
                nearest_distance = distance
        return nearest_sprite

    def screen_to_board(self, screen_x, screen_y):
        tolerance = self.board_click_tolerance
        if (
            screen_x < -tolerance
            or screen_y < -tolerance
            or screen_x > self.width + tolerance
            or screen_y > self.height + tolerance
        ):
            self.log_gui_event(
                "Click (%s, %s) rejected: outside board bounds.",
                screen_x,
                screen_y,
            )
            return None

        display_col_num = round((screen_x - self.chessman_w / 2) / self.chessman_w)
        display_row_num = round(9 - ((screen_y - self.chessman_h / 2) / self.chessman_h))
        col_num, row_num = orient_board_coordinate(
            display_col_num,
            display_row_num,
            invert=self.invert_board,
        )
        if not (0 <= col_num <= 8 and 0 <= row_num <= 9):
            self.log_gui_event(
                "Click (%s, %s) rejected: mapped to invalid board coordinate (%s, %s).",
                screen_x,
                screen_y,
                col_num,
                row_num,
            )
            return None

        center_x, center_y = self.board_to_screen(col_num, row_num)
        distance = math.hypot(screen_x - center_x, screen_y - center_y)
        if distance > tolerance:
            self.log_gui_event(
                "Click (%s, %s) rejected: nearest board coordinate (%s, %s) is %.2f px away (tolerance=%s).",
                screen_x,
                screen_y,
                col_num,
                row_num,
                distance,
                tolerance,
            )
            return None

        self.log_gui_event(
            "Click (%s, %s) mapped to board coordinate (%s, %s) with distance %.2f.",
            screen_x,
            screen_y,
            col_num,
            row_num,
            distance,
        )
        return col_num, row_num

    def resolve_click_target(self, screen_x, screen_y):
        sprite = self.find_sprite_near_point(screen_x, screen_y)
        if sprite is not None:
            board_pos = (sprite.chessman.col_num, sprite.chessman.row_num)
            self.log_gui_event(
                "Click (%s, %s) snapped to piece at %s with center distance %.2f.",
                screen_x,
                screen_y,
                board_pos,
                self.distance_to_piece_center(sprite, screen_x, screen_y),
            )
            return board_pos, sprite

        board_pos = self.screen_to_board(screen_x, screen_y)
        if board_pos is None:
            return None, None

        sprite = select_sprite_from_group(self.chessmans, board_pos[0], board_pos[1])
        return board_pos, sprite

    def log_gui_event(self, message, *args):
        if self.gui_debug:
            logger.info("GUI: " + message, *args)


    def can_human_move(self):
        return self.analysis_only or self.human_move_first == self.env.red_to_move

    def build_no_act(self, state, history):
        no_act = None
        _, _, _, check = senv.done(state, need_check=True)
        if not check and state in history[:-1]:
            no_act = []
            free_move = defaultdict(int)
            for i in range(len(history) - 1):
                if history[i] == state:
                    if senv.will_check_or_catch(state, history[i + 1]):
                        no_act.append(history[i + 1])
                    else:
                        free_move[state] += 1
                        if free_move[state] >= 2:
                            self.env.winner = Winner.draw
                            self.env.board.winner = Winner.draw
                            break
            if no_act:
                logger.debug(f"no_act = {no_act}")
        return no_act, check

    def update_analysis_panel(self, state, check):
        binding = self.model_binding
        if binding is None:
            return
        self.update_analysis_panel_for_binding(binding, state, check)

    def update_analysis_panel_for_binding(self, binding, state, check):
        debug_info = binding.ai.debug.get(state)
        self.nn_value = debug_info[1] if debug_info else 0
        logger.info(f"check = {check}, NN value = {self.nn_value:.3f}")
        logger.info("MCTS results:")
        self.mcts_moves = {}
        top_moves = sorted(
            binding.ai.search_results.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:3]
        self.analysis_arrows = []
        for move, action_state in binding.ai.search_results.items():
            move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
            logger.info(
                f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}"
            )
            self.mcts_moves[move_cn] = action_state
        for index, (move, action_state) in enumerate(top_moves):
            start_pos, end_pos = self.move_to_board_points(move, self.env.red_to_move)
            self.analysis_arrows.append(
                {
                    "rank": index,
                    "move": move,
                    "start": start_pos,
                    "end": end_pos,
                    "visits": action_state[0],
                }
            )

    def clear_analysis_display(self):
        self.nn_value = 0
        self.mcts_moves = {}
        self.analysis_arrows = []

    def request_analysis(self):
        self.clear_analysis_display()
        if not self.analysis_only:
            return
        with self.analysis_lock:
            self.analysis_request_id += 1

    def can_undo(self):
        return len(self.history) >= 3

    def can_redo(self):
        return bool(self.redo_stack)

    def clear_selected_piece(self, current_chessman):
        if current_chessman is not None:
            current_chessman.is_selected = False
        return None

    def record_completed_move(self, move, clear_redo=True):
        if clear_redo:
            self.redo_stack.clear()
        self.history.append(move)
        self.history.append(self.env.get_state())
        self.request_analysis()

    def build_env_from_history(self):
        env = CChessEnv()
        env.reset()
        for move in self.history[1::2]:
            env.step(move)
        return env

    def rebuild_sprites_from_env(self):
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(
            self.chessmans,
            self.env.board.chessmans_hash,
            self.chessman_w,
            self.chessman_h,
            invert=self.invert_board,
        )

    def refresh_after_history_change(self):
        active_option = self.active_model_option
        self.env = self.build_env_from_history()
        self.env.board.calc_chessmans_moving_list()
        current_state = self.env.get_state()
        if not self.history:
            self.history = [current_state]
        elif len(self.history) % 2 == 0:
            self.history.append(current_state)
        else:
            self.history[-1] = current_state
        self.rebuild_sprites_from_env()
        if active_option is not None:
            self.activate_model_option(active_option)
        else:
            self.request_analysis()

    def undo_move(self):
        if not self.can_undo():
            return False
        resulting_state = self.history.pop()
        move = self.history.pop()
        self.redo_stack.append((move, resulting_state))
        self.refresh_after_history_change()
        return True

    def redo_move(self):
        if not self.can_redo():
            return False
        move, resulting_state = self.redo_stack.pop()
        self.history.append(move)
        self.history.append(resulting_state)
        self.refresh_after_history_change()
        return True

    def draw_move_arrow(self, screen, start_pos, end_pos, color, width=6):
        start_x, start_y = self.board_to_screen(*start_pos)
        end_x, end_y = self.board_to_screen(*end_pos)
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        if length < 1:
            return

        ux = dx / length
        uy = dy / length
        piece_radius = min(self.chessman_w, self.chessman_h) * 0.32
        start = (
            start_x + ux * piece_radius,
            start_y + uy * piece_radius,
        )
        end = (
            end_x - ux * piece_radius,
            end_y - uy * piece_radius,
        )
        head_length = min(18, max(10, int(min(self.chessman_w, self.chessman_h) * 0.28)))
        head_width = max(8, int(width * 1.8))
        perp_x = -uy
        perp_y = ux
        head_base = (
            end[0] - ux * head_length,
            end[1] - uy * head_length,
        )

        pygame.draw.line(screen, color, start, end, width)
        pygame.draw.polygon(
            screen,
            color,
            [
                end,
                (
                    head_base[0] + perp_x * head_width / 2,
                    head_base[1] + perp_y * head_width / 2,
                ),
                (
                    head_base[0] - perp_x * head_width / 2,
                    head_base[1] - perp_y * head_width / 2,
                ),
            ],
        )

    def draw_analysis_arrows(self, screen):
        if not self.analysis_only or not self.analysis_arrows:
            return

        arrow_colors = [
            (255, 90, 90, 200),
            (90, 220, 120, 200),
            (80, 170, 255, 200),
        ]
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for arrow in self.analysis_arrows[:3]:
            color = arrow_colors[min(arrow["rank"], len(arrow_colors) - 1)]
            width = 7 if arrow["rank"] == 0 else 5
            self.draw_move_arrow(overlay, arrow["start"], arrow["end"], color, width=width)
        screen.blit(overlay, (0, 0))

    def analysis_worker(self):
        last_request_id = 0
        while not self.shutdown_requested and not self.env.board.is_end():
            with self.analysis_lock:
                request_id = self.analysis_request_id
            if request_id == 0 or request_id == last_request_id:
                self.cleanup_retired_bindings()
                sleep(0.05)
                continue

            last_request_id = request_id
            generation, binding = self.get_active_binding()
            if binding is None:
                sleep(0.05)
                continue
            self.mark_worker_binding(binding)
            state = self.env.get_state()
            history = list(self.history)
            turns = self.env.num_halfmoves
            no_act, check = self.build_no_act(state, history)
            try:
                binding.ai.search_results = {}
                binding.ai.action(state, turns, no_act)
            except Exception:
                latest_generation, latest_binding = self.get_active_binding()
                if latest_generation != generation or latest_binding is not binding:
                    logger.info("Discarded stale analysis for previous model: %s", binding.option.label)
                else:
                    logger.exception("Analysis worker failed for %s", binding.option.label)
                continue
            finally:
                self.clear_worker_binding(binding)
                self.cleanup_retired_bindings()

            latest_generation, latest_binding = self.get_active_binding()
            with self.analysis_lock:
                if request_id != self.analysis_request_id:
                    continue
            if latest_generation != generation or latest_binding is not binding:
                logger.info("Discarded analysis output for stale model %s", binding.option.label)
                continue
            if self.env.get_state() != state or self.env.num_halfmoves != turns:
                continue
            self.update_analysis_panel_for_binding(binding, state, check)

    def init_screen(self):
        screen, board_background, widget_background = self.recreate_display()
        screen.blit(board_background, (0, 0))
        screen.blit(widget_background, (self.width, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(
            self.chessmans,
            self.env.board.chessmans_hash,
            self.chessman_w,
            self.chessman_h,
            invert=self.invert_board,
        )
        return screen, board_background, widget_background

    def start(self, human_first=True):
        self.shutdown_requested = False
        self.env.reset()
        self.model_options = self.discover_model_options()
        if not self.activate_model_option(self.model_options[0]):
            raise RuntimeError(f"Unable to initialize GUI model {self.model_options[0].label}")
        self.human_move_first = human_first

        pygame.init()
        screen, board_background, widget_background = self.init_screen()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        current_chessman = None
        if human_first or self.analysis_only:
            self.env.board.calc_chessmans_moving_list()

        self.history = [self.env.get_state()]
        self.redo_stack = []
        self.request_analysis()
        worker_target = self.analysis_worker if self.analysis_only else self.ai_move
        worker_name = "analysis_worker" if self.analysis_only else "ai_worker"
        worker = Thread(target=worker_target, name=worker_name)
        worker.daemon = True
        worker.start()

        while not self.env.board.is_end():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.shutdown_requested = True
                    self.shutdown_model_bindings()
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    self.model_dropdown_open = False
                    self.update_window_layout(event.w, event.h)
                    screen, board_background, widget_background = self.recreate_display()
                elif event.type == KEYDOWN:
                    if getattr(event, "mod", 0) & KMOD_CTRL:
                        if event.key == K_z:
                            current_chessman = self.clear_selected_piece(current_chessman)
                            self.undo_move()
                        elif event.key == K_y:
                            current_chessman = self.clear_selected_piece(current_chessman)
                            self.redo_move()
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    mouse_x, mouse_y = event.pos
                    widget_handled, current_chessman = self.handle_widget_click(mouse_x, mouse_y, current_chessman)
                    if widget_handled:
                        continue
                    if self.can_human_move():
                        board_pos, chessman_sprite = self.resolve_click_target(mouse_x, mouse_y)
                        if board_pos is None:
                            continue

                        col_num, row_num = board_pos
                        if current_chessman is None and chessman_sprite is not None:
                            if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                current_chessman = chessman_sprite
                                chessman_sprite.is_selected = True
                                self.log_gui_event(
                                    "Selected piece at (%s, %s).",
                                    chessman_sprite.chessman.col_num,
                                    chessman_sprite.chessman.row_num,
                                )
                        elif current_chessman is not None and chessman_sprite is not None:
                            if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                current_chessman.is_selected = False
                                current_chessman = chessman_sprite
                                chessman_sprite.is_selected = True
                                self.log_gui_event(
                                    "Reselected piece at (%s, %s).",
                                    chessman_sprite.chessman.col_num,
                                    chessman_sprite.chessman.row_num,
                                )
                            else:
                                from_pos = (
                                    current_chessman.chessman.col_num,
                                    current_chessman.chessman.row_num,
                                )
                                move = str(from_pos[0]) + str(from_pos[1]) + str(col_num) + str(row_num)
                                success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                self.log_gui_event(
                                    "Tried capture move %s -> %s success=%s.",
                                    from_pos,
                                    (col_num, row_num),
                                    success,
                                )
                                if success:
                                    self.chessmans.remove(chessman_sprite)
                                    chessman_sprite.kill()
                                    current_chessman.is_selected = False
                                    current_chessman = None
                                    self.record_completed_move(move)
                        elif current_chessman is not None and chessman_sprite is None:
                            from_pos = (
                                current_chessman.chessman.col_num,
                                current_chessman.chessman.row_num,
                            )
                            move = str(from_pos[0]) + str(from_pos[1]) + str(col_num) + str(row_num)
                            success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                            self.log_gui_event(
                                "Tried move %s -> %s success=%s.",
                                from_pos,
                                (col_num, row_num),
                                success,
                            )
                            if success:
                                current_chessman.is_selected = False
                                current_chessman = None
                                self.record_completed_move(move)

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            screen.blit(board_background, (0, 0))

            # update all the sprites
            self.chessmans.update()
            self.chessmans.draw(screen)
            self.draw_analysis_arrows(screen)
            pygame.display.update()

        self.shutdown_requested = True
        self.shutdown_model_bindings()
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(3)

    def ai_move(self):
        ai_move_first = not self.human_move_first
        while not self.shutdown_requested and not self.env.done:
            if ai_move_first == self.env.red_to_move:
                generation, binding = self.get_active_binding()
                if binding is None:
                    sleep(0.05)
                    continue
                self.mark_worker_binding(binding)
                state = self.env.get_state()
                turns = self.env.num_halfmoves
                logger.info(f"state = {state}")
                no_act, check = self.build_no_act(state, self.history)
                try:
                    binding.ai.search_results = {}
                    action, _ = binding.ai.action(state, turns, no_act)
                except Exception:
                    latest_generation, latest_binding = self.get_active_binding()
                    if latest_generation != generation or latest_binding is not binding:
                        logger.info("Discarded in-flight search for stale model %s", binding.option.label)
                    else:
                        logger.exception("AI move failed for %s", binding.option.label)
                    continue
                finally:
                    self.clear_worker_binding(binding)
                    self.cleanup_retired_bindings()
                latest_generation, latest_binding = self.get_active_binding()
                if latest_generation != generation or latest_binding is not binding:
                    logger.info("Discarded completed search result for stale model %s", binding.option.label)
                    continue
                if self.env.get_state() != state or self.env.num_halfmoves != turns:
                    logger.info("Discarded completed search result for stale position under %s", binding.option.label)
                    continue
                if action is None:
                    logger.info("AI has resigned!")
                    return
                recorded_action = action
                if not self.env.red_to_move:
                    action = flip_move(action)
                self.update_analysis_panel_for_binding(binding, state, check)
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                self.record_completed_move(recorded_action)
            else:
                self.cleanup_retired_bindings()
                sleep(0.05)

    def draw_widget(self, screen, widget_background):
        widget_background.fill((255, 255, 255))
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        pygame.draw.line(widget_background, (255, 0, 0), (10, 412), (self.screen_width - self.width - 10, 412))
        screen.blit(widget_background, (self.width, 0))
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background)
        self.draw_model_selector(screen, widget_background)

    def draw_records(self, screen, widget_background):
        text = '着法记录'
        self.draw_label(screen, widget_background, text, 10, 16, 10)
        records = self.env.board.record.split('\n')
        font = self.load_font_with_fallback(12, prefer_chinese=True)
        i = 0
        for record in records[-self.disp_record_num:]:
            self.rec_labels[i] = font.render(record, True, (0, 0, 0), (255, 255, 255))
            t_rect = self.rec_labels[i].get_rect()
            t_rect.y = 35 + i * 15
            t_rect.x = 10
            t_rect.width = self.screen_width - self.width
            widget_background.blit(self.rec_labels[i], t_rect)
            i += 1
        screen.blit(widget_background, (self.width, 0))

    def draw_evaluation(self, screen, widget_background):
        title_label = 'CC-Zero信息'
        self.draw_label(screen, widget_background, title_label, 420, 16, 10)
        info_label = f'MCTS搜索次数：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 448, 14, 10)
        eval_label = f"当前局势评估: {self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 473, 14, 10)
        label = f"MCTS搜索结果:"
        self.draw_label(screen, widget_background, label, 498, 14, 10)
        label = f"着法 访问计数 动作价值 先验概率"
        self.draw_label(screen, widget_background, label, 518, 12, 10)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 538 + i * 16, 12, 10)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 538 + i * 16, 12, 70)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 538 + i * 16, 12, 100)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 538 + i * 16, 12, 150)
            i += 1

    def fit_text_to_width(self, text, font, max_width):
        if font.size(text)[0] <= max_width:
            return text
        ellipsis = "..."
        clipped = text
        while clipped and font.size(clipped + ellipsis)[0] > max_width:
            clipped = clipped[:-1]
        return clipped + ellipsis if clipped else ellipsis

    def get_model_option_rect(self, index):
        return Rect(
            self.model_dropdown_rect.x,
            self.model_dropdown_rect.bottom + index * self.model_dropdown_item_height,
            self.model_dropdown_rect.w,
            self.model_dropdown_item_height,
        )

    def draw_action_button(self, widget_background, rect, label, enabled, font):
        fill_color = (245, 245, 245) if enabled else (235, 235, 235)
        border_color = (0, 0, 0) if enabled else (160, 160, 160)
        text_color = (0, 0, 0) if enabled else (120, 120, 120)
        pygame.draw.rect(widget_background, fill_color, rect)
        pygame.draw.rect(widget_background, border_color, rect, 1)
        button_label = font.render(label, True, text_color, fill_color)
        button_rect = button_label.get_rect()
        button_rect.centerx = rect.centerx
        button_rect.centery = rect.centery
        widget_background.blit(button_label, button_rect)

    def draw_model_selector(self, screen, widget_background):
        body_font = self.load_font_with_fallback(12, prefer_chinese=True)
        self.draw_label(screen, widget_background, "模型选择", 300, 16, 10)
        current_label = self.active_model_option.label if self.active_model_option else "未加载"
        self.draw_label(
            screen,
            widget_background,
            self.fit_text_to_width(f"当前模型: {current_label}", body_font, self.model_dropdown_rect.w),
            325,
            12,
            10,
        )

        pygame.draw.rect(widget_background, (245, 245, 245), self.model_dropdown_rect)
        pygame.draw.rect(widget_background, (0, 0, 0), self.model_dropdown_rect, 1)
        selector_text = self.fit_text_to_width(current_label, body_font, self.model_dropdown_rect.w - 24)
        selector_label = body_font.render(selector_text, True, (0, 0, 0), (245, 245, 245))
        selector_rect = selector_label.get_rect()
        selector_rect.x = self.model_dropdown_rect.x + 6
        selector_rect.centery = self.model_dropdown_rect.centery
        widget_background.blit(selector_label, selector_rect)

        arrow_x = self.model_dropdown_rect.right - 12
        arrow_y = self.model_dropdown_rect.centery
        if self.model_dropdown_open:
            pygame.draw.polygon(widget_background, (0, 0, 0), [(arrow_x - 5, arrow_y + 3), (arrow_x + 5, arrow_y + 3), (arrow_x, arrow_y - 3)])
            for index, option in enumerate(self.model_options):
                option_rect = self.get_model_option_rect(index)
                is_active = self.active_model_option is not None and option.key == self.active_model_option.key
                fill_color = (230, 238, 255) if is_active else (250, 250, 250)
                pygame.draw.rect(widget_background, fill_color, option_rect)
                pygame.draw.rect(widget_background, (0, 0, 0), option_rect, 1)
                option_text = self.fit_text_to_width(option.label, body_font, option_rect.w - 12)
                option_label = body_font.render(option_text, True, (0, 0, 0), fill_color)
                option_text_rect = option_label.get_rect()
                option_text_rect.x = option_rect.x + 6
                option_text_rect.centery = option_rect.centery
                widget_background.blit(option_label, option_text_rect)
        else:
            pygame.draw.polygon(widget_background, (0, 0, 0), [(arrow_x - 5, arrow_y - 3), (arrow_x + 5, arrow_y - 3), (arrow_x, arrow_y + 3)])

        self.draw_action_button(widget_background, self.undo_button_rect, "撤销", self.can_undo(), body_font)
        self.draw_action_button(widget_background, self.redo_button_rect, "重做", self.can_redo(), body_font)
        screen.blit(widget_background, (self.width, 0))

    def handle_widget_click(self, screen_x, screen_y, current_chessman):
        widget_x = screen_x - self.width
        if widget_x < 0:
            if self.model_dropdown_open:
                self.model_dropdown_open = False
            return False, current_chessman

        if self.model_dropdown_rect.collidepoint(widget_x, screen_y):
            self.model_dropdown_open = not self.model_dropdown_open
            return True, current_chessman

        if self.model_dropdown_open:
            for index, option in enumerate(self.model_options):
                if self.get_model_option_rect(index).collidepoint(widget_x, screen_y):
                    self.model_dropdown_open = False
                    if self.active_model_option is None or option.key != self.active_model_option.key:
                        self.activate_model_option(option)
                    return True, current_chessman
            self.model_dropdown_open = False
            return True, current_chessman

        if self.undo_button_rect.collidepoint(widget_x, screen_y):
            current_chessman = self.clear_selected_piece(current_chessman)
            self.undo_move()
            return True, current_chessman

        if self.redo_button_rect.collidepoint(widget_x, screen_y):
            current_chessman = self.clear_selected_piece(current_chessman)
            self.redo_move()
            return True, current_chessman

        return True, current_chessman

    def handle_model_selector_click(self, screen_x, screen_y):
        handled, _ = self.handle_widget_click(screen_x, screen_y, None)
        return handled

    def draw_label(self, screen, widget_background, text, y, font_size, x=None):
        font = self.load_font_with_fallback(font_size, prefer_chinese=True)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        t_rect = label.get_rect()
        t_rect.y = y
        if x is not None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2
        widget_background.blit(label, t_rect)
        screen.blit(widget_background, (self.width, 0))


class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=80, h=80, invert=False):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = [pygame.transform.scale(image, (w, h)) for image in images]
        self.image = self.images[0]
        self.invert = invert
        self.rect = board_to_sprite_rect(chessman.col_num, chessman.row_num, w, h, invert=self.invert)

    def move(self, col_num, row_num, w=80, h=80):
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect = board_to_sprite_rect(self.chessman.col_num, self.chessman.row_num, w, h, invert=self.invert)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def load_image(file, sub_dir=None):
    if sub_dir:
        file = IMAGES_DIR / sub_dir / file
    else:
        file = IMAGES_DIR / file
    try:
        surface = pygame.image.load(str(file))
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' % (file, pygame.get_error()))
    return surface.convert()


def load_images(*files):
    global PIECE_STYLE
    imgs = []
    for file in files:
        imgs.append(load_image(file, PIECE_STYLE))
    return imgs


def creat_sprite_group(sprite_group, chessmans_hash, w, h, invert=False):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h, invert=invert)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None


def translate_hit_area(screen_x, screen_y, w=80, h=80):
    col_num = round((screen_x - w / 2) / w)
    row_num = round(9 - ((screen_y - h / 2) / h))
    return col_num, row_num
