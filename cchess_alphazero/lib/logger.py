import os
import socket
from logging import StreamHandler, basicConfig, DEBUG, INFO, getLogger, Formatter, FileHandler, LogRecord

try:
    import fcntl
except ImportError:  # Windows uses the existing local file handler behavior.
    fcntl = None


LOG_FORMAT = '%(asctime)s@%(name)s %(levelname)s # %(message)s'


class ClusterFileHandler(FileHandler):
    """Serialize NFS appends across jobs, including forked game workers."""

    def __init__(self, filename):
        super().__init__(filename, encoding="utf-8", delay=True)
        self._owner_pid = os.getpid()

    def emit(self, record):
        if fcntl is None:
            return super().emit(record)
        try:
            # flock ownership follows the open file description across fork.
            # Each child therefore needs its own descriptor, not the parent's.
            if self._owner_pid != os.getpid():
                if self.stream is not None:
                    self.stream.close()
                self.stream = None
                self._owner_pid = os.getpid()
            if self.stream is None:
                self.stream = self._open()
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX)
            try:
                # NFS does not provide atomic O_APPEND between clients.
                self.stream.seek(0, os.SEEK_END)
                super().emit(record)
            finally:
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        except Exception:
            self.handleError(record)


def setup_logger(log_filename, cluster=False):
    format_str = LOG_FORMAT
    if cluster:
        format_str += f' [host={socket.gethostname()} pid=%(process)d]'
    if not getLogger().handlers:
        handler = ClusterFileHandler(log_filename) if cluster else FileHandler(log_filename, encoding="utf-8")
        basicConfig(handlers=[handler], level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)


def log_model_event(config, event, **fields):
    """Put model transitions in the role log and in the shared main timeline."""
    message = "MODEL_EVENT %s %s" % (event, " ".join(f"{key}={value}" for key, value in fields.items()))
    event_logger = getLogger("cchess_alphazero.model_events")
    event_logger.info(message)
    main_path = os.path.abspath(config.resource.main_log_path)
    # Self-play already writes to main.log; avoid duplicating its events.
    if any(isinstance(h, FileHandler) and h.baseFilename == main_path for h in getLogger().handlers):
        return
    os.makedirs(os.path.dirname(main_path), exist_ok=True)
    handler = ClusterFileHandler(main_path)
    handler.setFormatter(Formatter(LOG_FORMAT + f' [host={socket.gethostname()} pid=%(process)d]'))
    try:
        handler.handle(LogRecord(event_logger.name, INFO, __file__, 0, message, (), None))
    finally:
        handler.close()

def setup_file_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)

if __name__ == '__main__':
    setup_logger("test.log")
    logger = getLogger("test")
    logger.info("OK")
