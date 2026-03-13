import os
import sys
import shutil
import types
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

from cchess_alphazero.config import Config


class _FakeGroup:
    def __init__(self):
        self._sprites = []

    def add(self, sprite):
        self._sprites.append(sprite)

    def remove(self, sprite):
        if sprite in self._sprites:
            self._sprites.remove(sprite)

    def draw(self, *args, **kwargs):
        return []

    def update(self):
        return None

    def __iter__(self):
        return iter(list(self._sprites))


class _FakeRect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def width(self):
        return self.w

    @property
    def height(self):
        return self.h

    @property
    def right(self):
        return self.x + self.w

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def centery(self):
        return self.y + self.h / 2

    @property
    def centerx(self):
        return self.x + self.w / 2

    @property
    def topleft(self):
        return (self.x, self.y)

    def collidepoint(self, x, y):
        return self.x <= x <= self.right and self.y <= y <= self.bottom


_fake_pygame = types.ModuleType("pygame")
_fake_pygame.error = Exception
_fake_pygame.sprite = types.SimpleNamespace(Sprite=type("Sprite", (), {}), Group=_FakeGroup)
_fake_pygame.transform = types.SimpleNamespace(scale=lambda image, size: image)
_fake_pygame.draw = types.SimpleNamespace(rect=lambda *args, **kwargs: None, polygon=lambda *args, **kwargs: None, line=lambda *args, **kwargs: None)
_fake_locals = types.ModuleType("pygame.locals")
_fake_locals.Rect = _FakeRect
_fake_locals.VIDEORESIZE = 1
_fake_locals.MOUSEBUTTONDOWN = 2
_fake_locals.KEYDOWN = 3
_fake_locals.RESIZABLE = 4
_fake_locals.K_z = ord('z')
_fake_locals.K_y = ord('y')
_fake_locals.KMOD_CTRL = 0x40
sys.modules.setdefault("pygame", _fake_pygame)
sys.modules.setdefault("pygame.locals", _fake_locals)

from cchess_alphazero.play_games import play as play_module


class _FakeModel:
    def __init__(self, config):
        self.config = config
        self.loaded = []
        self.build_calls = 0
        self.closed = False
        self.need_reload_calls = []

    def load(self, config_path, weight_path):
        self.loaded.append((config_path, weight_path))
        return os.path.exists(config_path) and os.path.exists(weight_path)

    def build(self):
        self.build_calls += 1

    def get_pipes(self, need_reload=True):
        self.need_reload_calls.append(need_reload)
        return {"need_reload": need_reload}

    def close_pipes(self):
        self.closed = True


class _FakePlayer:
    instances = []

    def __init__(self, config, search_tree=None, pipes=None, enable_resign=False, debugging=False, **kwargs):
        self.config = config
        self.tree = search_tree
        self.pipes = pipes
        self.enable_resign = enable_resign
        self.debugging = debugging
        self.closed_waits = []
        self.search_results = {}
        self.debug = {}
        self.action_calls = []
        type(self).instances.append(self)

    def action(self, state, turns, no_act=None):
        self.action_calls.append((state, turns, no_act))
        return "0001", []

    def close(self, wait=True):
        self.closed_waits.append(wait)


class PlayModelSelectorTest(unittest.TestCase):
    def setUp(self):
        _FakePlayer.instances = []
        self.rect_patch = patch.object(play_module, "Rect", _FakeRect)
        self.rect_patch.start()
        self.addCleanup(self.rect_patch.stop)
        self.group_patch = patch.object(play_module.pygame.sprite, "Group", _FakeGroup, create=True)
        self.group_patch.start()
        self.addCleanup(self.group_patch.stop)
        self.sprite_patch = patch.object(play_module, "creat_sprite_group", lambda *args, **kwargs: None)
        self.sprite_patch.start()
        self.addCleanup(self.sprite_patch.stop)
        self.project_root = Path(__file__).resolve().parents[2]
        scratch_root = self.project_root / ".tmp_test_play_models"
        scratch_root.mkdir(parents=True, exist_ok=True)
        self.data_dir = scratch_root / f"case_{uuid4().hex}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(self.data_dir, ignore_errors=True))
        self.model_dir = self.data_dir / "model"
        self.next_generation_dir = self.model_dir / "next_generation"
        self.next_generation_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / "model_best_config.json").write_text("{}", encoding="utf-8")
        (self.model_dir / "model_best_weight.h5").write_text("best", encoding="utf-8")
        (self.next_generation_dir / "next_generation_config.json").write_text("{}", encoding="utf-8")
        (self.next_generation_dir / "next_generation_weight.h5").write_text("next", encoding="utf-8")

        self.testdata_model_dir = self.project_root / "testdata" / "model"
        self.testdata_model_dir.mkdir(parents=True, exist_ok=True)
        self.testdata_best_config = self.testdata_model_dir / "model_best_config.json"
        self.testdata_best_weight = self.testdata_model_dir / "model_best_weight.h5"
        self._created_testdata_files = []
        for target, contents in [
            (self.testdata_best_config, "{}"),
            (self.testdata_best_weight, "testdata-best"),
        ]:
            if not target.exists():
                target.write_text(contents, encoding="utf-8")
                self._created_testdata_files.append(target)
        self.addCleanup(self.cleanup_testdata_model_root)

    def create_play(self, window_width=None, window_height=None):
        config = Config("local_torch")
        config.resource.update_paths(data_dir=str(self.data_dir))
        config.resource.create_directories()
        config.opts.window_width = window_width
        config.opts.window_height = window_height
        return play_module.PlayWithHuman(config)

    def cleanup_testdata_model_root(self):
        for target in self._created_testdata_files:
            if target.exists():
                target.unlink()
        try:
            self.testdata_model_dir.rmdir()
        except OSError:
            pass

    def test_discover_model_options_uses_current_and_explicit_repo_roots(self):
        play = self.create_play()

        options = play.discover_model_options()

        self.assertEqual(options[0].config_path, str(self.model_dir / "model_best_config.json"))
        self.assertEqual(options[1].config_path, str(self.next_generation_dir / "next_generation_config.json"))
        self.assertTrue(options[0].allow_background_reload)
        self.assertFalse(options[1].allow_background_reload)
        self.assertTrue(any(option.weight_path == str(self.testdata_best_weight) for option in options))
        self.assertTrue(any("testdata" in option.label for option in options))

    def test_window_layout_uses_configured_size_and_updates_panel_width(self):
        play = self.create_play(window_width=900, window_height=700)

        self.assertEqual(play.screen_width, 900)
        self.assertEqual(play.window_height, 700)
        self.assertEqual(play.model_dropdown_rect.w, 900 - play.width - 20)
        self.assertEqual(play.redo_button_rect.y, 8)
        self.assertEqual(play.redo_button_rect.right, 900 - play.width - 10)
        self.assertLess(play.undo_button_rect.right, play.redo_button_rect.x)

        play.update_window_layout(980, 760)

        self.assertEqual(play.screen_width, 980)
        self.assertEqual(play.window_height, 760)
        self.assertEqual(play.model_dropdown_rect.w, 980 - play.width - 20)
        self.assertEqual(play.redo_button_rect.right, 980 - play.width - 10)
        self.assertLess(play.undo_button_rect.right, play.redo_button_rect.x)
        self.assertEqual(play.screen_to_board(*play.board_to_screen(0, 0)), (0, 0))
        self.assertEqual(play.screen_to_board(*play.board_to_screen(8, 9)), (8, 9))

    def test_selector_switch_preserves_board_state_and_history(self):
        play = self.create_play()
        play.env.reset()
        play.env.board.calc_chessmans_moving_list()
        first_move = play.env.board.legal_moves()[0]
        initial_state = play.env.get_state()
        play.env.step(first_move)
        state_before = play.env.get_state()
        red_to_move_before = play.env.red_to_move
        record_before = play.env.board.record
        halfmoves_before = play.env.num_halfmoves
        play.history = [initial_state, first_move, state_before]

        with patch.object(play_module, "CChessModel", _FakeModel), patch.object(play_module, "CChessPlayer", _FakePlayer):
            play.model_options = play.discover_model_options()
            play.activate_model_option(play.model_options[0])
            original_binding = play.model_binding
            self.assertTrue(original_binding.model.need_reload_calls[-1])

            toggle_x = play.width + play.model_dropdown_rect.x + 1
            toggle_y = play.model_dropdown_rect.y + 1
            self.assertTrue(play.handle_model_selector_click(toggle_x, toggle_y))
            self.assertTrue(play.model_dropdown_open)

            option_rect = play.get_model_option_rect(1)
            self.assertTrue(play.handle_model_selector_click(play.width + option_rect.x + 1, option_rect.y + 1))

            self.assertTrue(play.active_model_option.key.endswith(":next_generation"))
            self.assertIsNot(play.model_binding, original_binding)
            self.assertEqual(play.env.get_state(), state_before)
            self.assertEqual(play.env.red_to_move, red_to_move_before)
            self.assertEqual(play.env.board.record, record_before)
            self.assertEqual(play.env.num_halfmoves, halfmoves_before)
            self.assertEqual(play.history, [initial_state, first_move, state_before])
            self.assertEqual(original_binding.ai.closed_waits, [False])
            self.assertTrue(original_binding.model.closed)
            self.assertFalse(play.model_binding.model.need_reload_calls[-1])
            source_square = (int(first_move[0]), int(first_move[1]))
            dest_square = (int(first_move[2]), int(first_move[3]))
            self.assertEqual(play.screen_to_board(*play.board_to_screen(*source_square)), source_square)
            self.assertEqual(play.screen_to_board(*play.board_to_screen(*dest_square)), dest_square)

            play.model_binding.ai.action(play.env.get_state(), play.env.num_halfmoves, None)
            self.assertEqual(play.model_binding.ai.action_calls[-1], (state_before, halfmoves_before, None))

    def test_undo_restores_previous_position_and_preserves_selected_model(self):
        play = self.create_play()
        play.env.reset()
        play.env.board.calc_chessmans_moving_list()
        initial_state = play.env.get_state()
        initial_turn = play.env.red_to_move
        initial_record = play.env.board.record
        play.history = [initial_state]

        with patch.object(play_module, "CChessModel", _FakeModel), patch.object(play_module, "CChessPlayer", _FakePlayer):
            play.model_options = play.discover_model_options()
            play.activate_model_option(play.model_options[1])
            original_binding = play.model_binding

            first_move = play.env.board.legal_moves()[0]
            play.env.step(first_move)
            state_after_move = play.env.get_state()
            turn_after_move = play.env.red_to_move
            record_after_move = play.env.board.record
            halfmoves_after_move = play.env.num_halfmoves
            play.record_completed_move(first_move)

            self.assertTrue(play.undo_move())

            self.assertEqual(play.env.get_state(), initial_state)
            self.assertEqual(play.env.red_to_move, initial_turn)
            self.assertEqual(play.env.board.record, initial_record)
            self.assertEqual(play.env.num_halfmoves, 0)
            self.assertEqual(play.history, [initial_state])
            self.assertTrue(play.can_redo())
            self.assertEqual(play.redo_stack[-1], (first_move, state_after_move))
            self.assertEqual(play.active_model_option.key, play.model_options[1].key)
            self.assertIsNot(play.model_binding, original_binding)
            self.assertEqual(original_binding.ai.closed_waits, [False])
            self.assertTrue(original_binding.model.closed)
            source_square = (int(first_move[0]), int(first_move[1]))
            self.assertEqual(play.screen_to_board(*play.board_to_screen(*source_square)), source_square)

            play.model_binding.ai.action(play.env.get_state(), play.env.num_halfmoves, None)
            self.assertEqual(play.model_binding.ai.action_calls[-1], (initial_state, 0, None))
            self.assertTrue(record_after_move)
            self.assertFalse(turn_after_move)
            self.assertEqual(halfmoves_after_move, 1)

    def test_redo_reapplies_last_undone_move_and_clears_after_new_move(self):
        play = self.create_play()
        play.env.reset()
        play.env.board.calc_chessmans_moving_list()
        initial_state = play.env.get_state()
        play.history = [initial_state]

        with patch.object(play_module, "CChessModel", _FakeModel), patch.object(play_module, "CChessPlayer", _FakePlayer):
            play.model_options = play.discover_model_options()
            play.activate_model_option(play.model_options[0])

            first_move = play.env.board.legal_moves()[0]
            play.env.step(first_move)
            state_after_first_move = play.env.get_state()
            turn_after_first_move = play.env.red_to_move
            record_after_first_move = play.env.board.record
            halfmoves_after_first_move = play.env.num_halfmoves
            play.record_completed_move(first_move)

            self.assertTrue(play.undo_move())
            self.assertTrue(play.redo_move())

            self.assertEqual(play.env.get_state(), state_after_first_move)
            self.assertEqual(play.env.red_to_move, turn_after_first_move)
            self.assertEqual(play.env.board.record, record_after_first_move)
            self.assertEqual(play.env.num_halfmoves, halfmoves_after_first_move)
            self.assertEqual(play.history, [initial_state, first_move, state_after_first_move])
            self.assertFalse(play.can_redo())
            self.assertTrue(play.active_model_option.key.endswith(":best"))

            self.assertTrue(play.undo_move())
            alternative_move = next(move for move in play.env.board.legal_moves() if move != first_move)
            play.env.step(alternative_move)
            play.record_completed_move(alternative_move)

            self.assertFalse(play.can_redo())
            self.assertEqual(play.history, [initial_state, alternative_move, play.env.get_state()])


if __name__ == "__main__":
    unittest.main()
