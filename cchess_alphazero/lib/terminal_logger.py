import os
import socket
from datetime import datetime


def terminal_logging_enabled(config) -> bool:
    return getattr(getattr(config, "terminal_log", None), "style", None) == "linux"


def should_log_moves(config) -> bool:
    return terminal_logging_enabled(config) and bool(getattr(config.terminal_log, "log_moves", False))


def should_log_game_summary(config) -> bool:
    return terminal_logging_enabled(config) and bool(getattr(config.terminal_log, "log_game_summary", False))


def should_log_buffer_flush(config) -> bool:
    return terminal_logging_enabled(config) and bool(getattr(config.terminal_log, "log_buffer_flush", False))


def should_log_model_reload(config) -> bool:
    return terminal_logging_enabled(config) and bool(getattr(config.terminal_log, "log_model_reload", False))


def emit_terminal_log(config, role, message, worker_id=None, pid=None):
    if not terminal_logging_enabled(config):
        return
    parts = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    prefix = []
    if getattr(config.terminal_log, "log_worker_prefix", False):
        prefix.append(f"role={role}")
        if worker_id is not None:
            prefix.append(f"worker={worker_id}")
    if getattr(config.terminal_log, "log_pid", False):
        prefix.append(f"pid={os.getpid() if pid is None else pid}")
    if getattr(config.terminal_log, "log_node_info", False):
        prefix.append(f"host={socket.gethostname()}")
    if prefix:
        parts.append("[" + " ".join(prefix) + "]")
    parts.append(str(message))
    print(" ".join(parts), flush=True)
