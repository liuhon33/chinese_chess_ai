import math
import os
import sys
import pygame
import random
import time
import copy
import numpy as np
from pathlib import Path

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
from cchess_alphazero.lib.model_helper import load_best_model_weight
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
        self.winstyle = 0
        self.chessmans = None
        self.human_move_first = True
        self.screen_width = 720
        self.height = 577
        self.width = 521
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

    def load_model(self):
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

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
        debug_info = self.ai.debug.get(state)
        self.nn_value = debug_info[1] if debug_info else 0
        logger.info(f"check = {check}, NN value = {self.nn_value:.3f}")
        logger.info("MCTS results:")
        self.mcts_moves = {}
        top_moves = sorted(
            self.ai.search_results.items(),
            key=lambda item: item[1][0],
            reverse=True,
        )[:3]
        self.analysis_arrows = []
        for move, action_state in self.ai.search_results.items():
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

    def request_analysis(self):
        if not self.analysis_only:
            return
        with self.analysis_lock:
            self.analysis_request_id += 1
        self.nn_value = 0
        self.mcts_moves = {}
        self.analysis_arrows = []

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
        while not self.env.board.is_end():
            with self.analysis_lock:
                request_id = self.analysis_request_id
            if request_id == 0 or request_id == last_request_id:
                sleep(0.05)
                continue

            last_request_id = request_id
            state = self.env.get_state()
            history = list(self.history)
            turns = self.env.num_halfmoves
            no_act, check = self.build_no_act(state, history)
            self.ai.search_results = {}
            self.ai.action(state, turns, no_act)

            with self.analysis_lock:
                if request_id != self.analysis_request_id:
                    continue
            if self.env.get_state() != state:
                continue
            self.update_analysis_panel(state, check)

    def init_screen(self):
        bestdepth = pygame.display.mode_ok([self.screen_width, self.height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋Zero")
        # create the background, tile the bgd image
        bgdtile = load_image(f'{self.config.opts.bg_style}.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))
        widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)

        #create text label
        font = self.load_font_with_fallback(16, prefer_chinese=True)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        t = font.render("着法记录", True, font_color, font_background)
        t_rect = t.get_rect()
        t_rect.x = 10
        t_rect.y = 10
        widget_background.blit(t, t_rect)

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
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=True)
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
                    self.ai.close(wait=False)
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    if self.can_human_move():
                        mouse_x, mouse_y = event.pos
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
                                    self.history.append(move)
                                    self.chessmans.remove(chessman_sprite)
                                    chessman_sprite.kill()
                                    current_chessman.is_selected = False
                                    current_chessman = None
                                    self.history.append(self.env.get_state())
                                    self.request_analysis()
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
                                self.history.append(move)
                                current_chessman.is_selected = False
                                current_chessman = None
                                self.history.append(self.env.get_state())
                                self.request_analysis()

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            screen.blit(board_background, (0, 0))

            # update all the sprites
            self.chessmans.update()
            self.chessmans.draw(screen)
            self.draw_analysis_arrows(screen)
            pygame.display.update()

        self.ai.close(wait=False)
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir, self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(3)

    def ai_move(self):
        ai_move_first = not self.human_move_first
        while not self.env.done:
            if ai_move_first == self.env.red_to_move:
                self.ai.search_results = {}
                state = self.env.get_state()
                logger.info(f"state = {state}")
                no_act, check = self.build_no_act(state, self.history)
                action, _ = self.ai.action(state, self.env.num_halfmoves, no_act)
                if action is None:
                    logger.info("AI has resigned!")
                    return
                self.history.append(action)
                if not self.env.red_to_move:
                    action = flip_move(action)
                self.update_analysis_panel(state, check)
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                self.history.append(self.env.get_state())
            else:
                sleep(0.05)

    def draw_widget(self, screen, widget_background):
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        screen.blit(widget_background, (self.width, 0))
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background)

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
        self.draw_label(screen, widget_background, title_label, 300, 16, 10)
        info_label = f'MCTS搜索次数：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 335, 14, 10)
        eval_label = f"当前局势评估: {self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 360, 14, 10)
        label = f"MCTS搜索结果:"
        self.draw_label(screen, widget_background, label, 395, 14, 10)
        label = f"着法 访问计数 动作价值 先验概率"
        self.draw_label(screen, widget_background, label, 415, 12, 10)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 10)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 70)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 100)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 150)
            i += 1

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
