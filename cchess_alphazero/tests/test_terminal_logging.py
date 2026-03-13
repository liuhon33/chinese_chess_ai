import io
import unittest
from contextlib import redirect_stdout

from cchess_alphazero.config import Config
from cchess_alphazero.lib.terminal_logger import emit_terminal_log
from cchess_alphazero.manager import create_parser, setup


class TerminalLoggingConfigTest(unittest.TestCase):
    def test_terminal_logging_defaults_disabled(self):
        parser = create_parser()
        args = parser.parse_args(["self", "--type", "local_torch"])
        config = Config("local_torch")
        setup(config, args)

        self.assertIsNone(config.terminal_log.style)
        self.assertFalse(config.terminal_log.log_moves)
        self.assertFalse(config.terminal_log.log_game_summary)
        self.assertFalse(config.terminal_log.log_buffer_flush)
        self.assertFalse(config.terminal_log.log_model_reload)
        self.assertFalse(config.terminal_log.log_worker_prefix)
        self.assertFalse(config.terminal_log.log_pid)
        self.assertFalse(config.terminal_log.log_node_info)

    def test_terminal_logging_flags_are_opt_in(self):
        parser = create_parser()
        args = parser.parse_args(
            [
                "self",
                "--type",
                "local_torch",
                "--terminal-log-style",
                "linux",
                "--log-moves",
                "--log-game-summary",
                "--log-buffer-flush",
                "--log-model-reload",
                "--log-worker-prefix",
                "--log-pid",
                "--log-node-info",
            ]
        )
        config = Config("local_torch")
        setup(config, args)

        self.assertEqual(config.terminal_log.style, "linux")
        self.assertTrue(config.terminal_log.log_moves)
        self.assertTrue(config.terminal_log.log_game_summary)
        self.assertTrue(config.terminal_log.log_buffer_flush)
        self.assertTrue(config.terminal_log.log_model_reload)
        self.assertTrue(config.terminal_log.log_worker_prefix)
        self.assertTrue(config.terminal_log.log_pid)
        self.assertTrue(config.terminal_log.log_node_info)

    def test_emit_terminal_log_uses_prefix_flags(self):
        config = Config("local_torch")
        config.terminal_log.style = "linux"
        config.terminal_log.log_worker_prefix = True
        config.terminal_log.log_pid = True
        config.terminal_log.log_node_info = True

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            emit_terminal_log(config, "self", "hello", worker_id="7", pid=123)

        output = buffer.getvalue().strip()
        self.assertIn("role=self", output)
        self.assertIn("worker=7", output)
        self.assertIn("pid=123", output)
        self.assertIn("host=", output)
        self.assertTrue(output.endswith("hello"))


if __name__ == "__main__":
    unittest.main()
