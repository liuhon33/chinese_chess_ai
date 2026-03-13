import sys
import types
import unittest

from cchess_alphazero.config import Config
from cchess_alphazero.manager import create_parser, setup


class _FakeRect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def topleft(self):
        return (self.x, self.y)


_fake_pygame = types.ModuleType("pygame")
_fake_pygame.error = Exception
_fake_pygame.sprite = types.SimpleNamespace(Sprite=type("Sprite", (), {}))
_fake_pygame.transform = types.SimpleNamespace(scale=lambda image, size: image)
_fake_locals = types.ModuleType("pygame.locals")
_fake_locals.Rect = _FakeRect
sys.modules.setdefault("pygame", _fake_pygame)
sys.modules.setdefault("pygame.locals", _fake_locals)

from cchess_alphazero.play_games.play import PlayWithHuman, board_to_sprite_rect


class GuiInvertConfigTest(unittest.TestCase):
    def test_invert_flag_is_opt_in(self):
        parser = create_parser()
        args = parser.parse_args(["play", "--type", "local_torch", "--invert"])
        config = Config("local_torch")
        setup(config, args)

        self.assertTrue(config.opts.invert)


class GuiInvertMappingTest(unittest.TestCase):
    def create_play(self, invert):
        config = Config("local_torch")
        config.opts.invert = invert
        return PlayWithHuman(config)

    def test_normal_board_mapping_is_unchanged(self):
        play = self.create_play(invert=False)

        self.assertEqual(play.board_to_screen(0, 0), (play.chessman_w / 2, 9 * play.chessman_h + play.chessman_h / 2))
        self.assertEqual(play.board_to_screen(8, 9), (8 * play.chessman_w + play.chessman_w / 2, play.chessman_h / 2))

        for col_num in (0, 4, 8):
            for row_num in (0, 5, 9):
                with self.subTest(invert=False, col_num=col_num, row_num=row_num):
                    self.assertEqual(play.screen_to_board(*play.board_to_screen(col_num, row_num)), (col_num, row_num))

    def test_inverted_board_mapping_flips_both_axes(self):
        play = self.create_play(invert=True)

        self.assertEqual(play.board_to_screen(0, 0), (8 * play.chessman_w + play.chessman_w / 2, play.chessman_h / 2))
        self.assertEqual(play.board_to_screen(8, 9), (play.chessman_w / 2, 9 * play.chessman_h + play.chessman_h / 2))

        for col_num in (0, 4, 8):
            for row_num in (0, 5, 9):
                with self.subTest(invert=True, col_num=col_num, row_num=row_num):
                    self.assertEqual(play.screen_to_board(*play.board_to_screen(col_num, row_num)), (col_num, row_num))

    def test_sprite_rect_mapping_matches_visual_orientation(self):
        self.assertEqual(board_to_sprite_rect(0, 0, 10, 20).topleft, (0, 180))
        self.assertEqual(board_to_sprite_rect(0, 0, 10, 20, invert=True).topleft, (80, 0))
        self.assertEqual(board_to_sprite_rect(8, 9, 10, 20, invert=True).topleft, (0, 180))


if __name__ == "__main__":
    unittest.main()
