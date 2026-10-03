import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cchess_alphazero.config import Config
from cchess_alphazero.lib.logger import ClusterFileHandler, log_model_event


class ModelEventLoggingTest(unittest.TestCase):
    def setUp(self):
        os.makedirs(".tmp_testdata", exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=".tmp_testdata")
        self.addCleanup(self.temp.cleanup)
        self.config = Config("mini")
        self.config.resource.update_paths(project_dir=self.temp.name, data_dir=self.temp.name)
        self.config.resource.create_directories()

    def test_role_event_is_copied_to_main_with_both_model_ids(self):
        log_model_event(self.config, "BEST_MODEL_PROMOTED", previous_best="old", current_best="new")
        text = Path(self.config.resource.main_log_path).read_text(encoding="utf-8")
        self.assertIn("MODEL_EVENT BEST_MODEL_PROMOTED previous_best=old current_best=new", text)
        self.assertIn("pid=", text)

    def test_selfplay_event_is_not_duplicated_in_main(self):
        root = logging.getLogger()
        level = root.level
        handler = ClusterFileHandler(self.config.resource.main_log_path)
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        try:
            log_model_event(self.config, "SELFPLAY_MODEL_RELOADED", previous_best="old", current_best="new")
        finally:
            root.removeHandler(handler)
            root.setLevel(level)
            handler.close()
        text = Path(self.config.resource.main_log_path).read_text(encoding="utf-8")
        self.assertEqual(text.count("MODEL_EVENT SELFPLAY_MODEL_RELOADED"), 1)

    @unittest.skipUnless(os.name == "posix", "Cluster NFS locking uses POSIX flock")
    def test_forked_workers_reopen_inherited_handler_and_keep_records_intact(self):
        script = r'''
import logging, multiprocessing, os, sys
from cchess_alphazero.lib.logger import ClusterFileHandler
handler = ClusterFileHandler(sys.argv[1])
def write(label):
    for index in range(40):
        text = f"{label}:{index}:" + label * 8192
        handler.handle(logging.LogRecord("test", logging.INFO, "", 0, text, (), None))
    assert handler._owner_pid == os.getpid()
write("p")  # Open the parent's descriptor before forking.
ctx = multiprocessing.get_context("fork")
workers = [ctx.Process(target=write, args=(label,)) for label in "abcd"]
for worker in workers: worker.start()
for worker in workers:
    worker.join(15)
    assert worker.exitcode == 0, worker.exitcode
handler.close()
'''
        path = str(Path(self.temp.name) / "concurrent.log")
        subprocess.run([sys.executable, "-c", script, path], check=True, timeout=30)
        lines = Path(path).read_text().splitlines()
        expected = {f"{label}:{index}:" + label * 8192 for label in "pabcd" for index in range(40)}
        self.assertEqual(len(lines), len(expected))
        self.assertEqual(set(lines), expected)


if __name__ == "__main__":
    unittest.main()
