import json
import os
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Callable, Optional
from uuid import uuid4

from logging import getLogger

logger = getLogger(__name__)

DEFAULT_MODEL_RELOAD_INTERVAL = 600.0
DEFAULT_FILE_STABILITY_SECONDS = 2.0


def cluster_enabled(config) -> bool:
    return bool(getattr(getattr(config, "cluster", None), "enabled", False))


def safe_write_play_data_enabled(config) -> bool:
    return cluster_enabled(config) and bool(getattr(config.cluster, "safe_write_play_data", False))


def archive_consumed_data_enabled(config) -> bool:
    return cluster_enabled(config) and bool(getattr(config.cluster, "archive_consumed_data", False))


def auto_reload_best_enabled(config) -> bool:
    setting = getattr(getattr(config, "cluster", None), "auto_reload_best", None)
    if setting is None:
        return True
    return bool(setting)


def best_model_reload_interval(config, default: float = DEFAULT_MODEL_RELOAD_INTERVAL) -> float:
    override = getattr(getattr(config, "cluster", None), "reload_best_interval", None)
    if override is None:
        return float(default)
    return max(1.0, float(override))


def optimizer_poll_interval(config) -> float:
    override = getattr(getattr(config, "cluster", None), "optimizer_poll_interval", None)
    if override is None:
        return max(1.0, float(getattr(config.trainer, "polling_interval", 300)))
    return max(1.0, float(override))


def evaluator_poll_interval(config) -> float:
    override = getattr(getattr(config, "cluster", None), "evaluator_poll_interval", None)
    if override is None:
        return max(1.0, float(getattr(config.eval, "polling_interval", 300)))
    return max(1.0, float(override))


def build_cluster_play_data_path(config, pid: Optional[int] = None) -> str:
    rc = config.resource
    worker_id = getattr(config.cluster, "worker_id", None)
    worker_label = sanitize_label(worker_id or "worker")
    host_label = sanitize_label(socket.gethostname() or "host")
    pid_value = int(os.getpid() if pid is None else pid)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    token = uuid4().hex[:8]
    filename = rc.play_data_filename_tmpl % f"{stamp}_{host_label}_{worker_label}_p{pid_value}_{token}"
    return str(Path(rc.play_data_dir) / filename)


def sanitize_label(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))


def write_json_atomic(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path_for(target)
    try:
        with temp_path.open("wt") as handle:
            json.dump(data, handle)
        os.replace(str(temp_path), str(target))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def publish_model_pair_atomically(
    save_fn: Callable[[str, str], None],
    config_path: str,
    weight_path: str,
    ready_path: Optional[str] = None,
):
    config_target = Path(config_path)
    weight_target = Path(weight_path)
    config_target.parent.mkdir(parents=True, exist_ok=True)
    weight_target.parent.mkdir(parents=True, exist_ok=True)
    temp_config = _temp_path_for(config_target)
    temp_weight = _temp_path_for(weight_target)
    ready_target = Path(ready_path) if ready_path else None

    if ready_target is not None and ready_target.exists():
        ready_target.unlink()

    try:
        save_fn(str(temp_config), str(temp_weight))
        os.replace(str(temp_config), str(config_target))
        os.replace(str(temp_weight), str(weight_target))
        if ready_target is not None:
            write_json_atomic(
                ready_target,
                {
                    "published_at": time.time(),
                    "config": config_target.name,
                    "weight": weight_target.name,
                },
            )
    finally:
        for temp_path in (temp_config, temp_weight):
            if temp_path.exists():
                temp_path.unlink()


def copy_file_atomically(src_path: str, dst_path: str):
    src = Path(src_path)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_path_for(dst)
    try:
        shutil.copyfile(str(src), str(temp_path))
        os.replace(str(temp_path), str(dst))
    finally:
        if temp_path.exists():
            temp_path.unlink()


def next_generation_model_ready(config) -> bool:
    rc = config.resource
    config_path = Path(rc.next_generation_config_path)
    weight_path = Path(rc.next_generation_weight_path)
    ready_path = Path(rc.next_generation_ready_path)
    if not (config_path.exists() and weight_path.exists()):
        return False
    if not cluster_enabled(config):
        return True
    if not ready_path.exists():
        return False
    ready_mtime = ready_path.stat().st_mtime
    return ready_mtime >= max(config_path.stat().st_mtime, weight_path.stat().st_mtime)


def remove_file_if_exists(path: str):
    target = Path(path)
    if target.exists():
        target.unlink()


def is_file_stable(path: str, min_age_seconds: float = DEFAULT_FILE_STABILITY_SECONDS) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    try:
        file_stat = target.stat()
    except FileNotFoundError:
        return False
    if file_stat.st_size <= 0:
        return False
    return time.time() - file_stat.st_mtime >= float(min_age_seconds)


def claim_play_data_file(config, path: str) -> Optional[str]:
    source = Path(path)
    if not source.exists():
        return None
    inflight_dir = Path(config.resource.play_data_inflight_dir)
    inflight_dir.mkdir(parents=True, exist_ok=True)
    claimed_name = f"{source.name}.{sanitize_label(getattr(config.cluster, 'worker_id', 'worker'))}.{os.getpid()}.{uuid4().hex[:8]}.claim"
    claimed_path = inflight_dir / claimed_name
    try:
        os.replace(str(source), str(claimed_path))
        return str(claimed_path)
    except FileNotFoundError:
        return None
    except PermissionError:
        logger.debug("Unable to claim play data file %s due to a permission error.", source)
        return None
    except OSError:
        logger.debug("Unable to claim play data file %s.", source)
        return None


def finalize_claimed_play_data(config, claimed_path: str, original_name: str):
    claimed = Path(claimed_path)
    if not claimed.exists():
        return None
    if archive_consumed_data_enabled(config):
        archive_dir = Path(config.resource.trained_data_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / original_name
        if destination.exists():
            destination = archive_dir / f"{Path(original_name).stem}_{uuid4().hex[:8]}{Path(original_name).suffix}"
        shutil.move(str(claimed), str(destination))
        return str(destination)
    claimed.unlink()
    return None


def restore_claimed_play_data(config, claimed_path: str, original_name: str):
    claimed = Path(claimed_path)
    if not claimed.exists():
        return None
    original_path = Path(config.resource.play_data_dir) / original_name
    if original_path.exists():
        original_path = Path(config.resource.play_data_dir) / f"{Path(original_name).stem}_{uuid4().hex[:8]}{Path(original_name).suffix}"
    os.replace(str(claimed), str(original_path))
    return str(original_path)


def _temp_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.{uuid4().hex}.tmp")
