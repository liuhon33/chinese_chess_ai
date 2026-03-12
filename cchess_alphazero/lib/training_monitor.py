import csv
import json
import math
import os
from datetime import datetime, timezone
from glob import glob
from logging import getLogger

from cchess_alphazero.agent.model import CChessModel

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - Pillow is available in the local Torch env
    Image = None
    ImageDraw = None
    ImageFont = None

logger = getLogger(__name__)

ELO_HISTORY_FIELDS = [
    "timestamp",
    "data_dir",
    "total_self_play_games",
    "total_step",
    "candidate_model",
    "candidate_digest",
    "best_model",
    "best_digest",
    "wins",
    "losses",
    "draws",
    "elo",
    "promotion_decision",
]

PNG_BACKGROUND = (255, 255, 255)
PNG_AXIS = (40, 40, 40)
PNG_GRID = (224, 224, 224)
PNG_LINE = (36, 88, 168)
PNG_POINT = (20, 55, 118)
PNG_TEXT = (32, 32, 32)


def save_training_state(config, total_steps):
    os.makedirs(os.path.dirname(config.resource.training_state_path), exist_ok=True)
    payload = {
        "updated_at": utc_timestamp(),
        "total_steps": int(total_steps),
        "candidate_digest": CChessModel.fetch_digest(config.resource.next_generation_weight_path),
    }
    with open(config.resource.training_state_path, "wt", encoding="utf-8") as state_file:
        json.dump(payload, state_file, indent=2, sort_keys=True)
    return payload


def load_training_state(config):
    path = config.resource.training_state_path
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rt", encoding="utf-8") as state_file:
            return json.load(state_file)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read training state from %s: %s", path, exc)
        return {}


def record_eval_metrics(config, result, promotion_decision):
    rc = config.resource
    os.makedirs(rc.log_dir, exist_ok=True)

    total_self_play_games = count_cumulative_self_play_games(config)
    training_state = load_training_state(config)
    row = {
        "timestamp": utc_timestamp(),
        "data_dir": rc.data_dir,
        "total_self_play_games": total_self_play_games,
        "total_step": training_state.get("total_steps", ""),
        "candidate_model": os.path.basename(rc.next_generation_weight_path),
        "candidate_digest": CChessModel.fetch_digest(rc.next_generation_weight_path) or "",
        "best_model": os.path.basename(rc.model_best_weight_path),
        "best_digest": CChessModel.fetch_digest(rc.model_best_weight_path) or "",
        "wins": int(result["wins"]),
        "losses": int(result["losses"]),
        "draws": int(result["draws"]),
        "elo": format_float(result["elo"]),
        "promotion_decision": promotion_decision,
    }
    append_csv_row(rc.elo_history_path, row)
    plot_rows = load_plot_rows(rc.elo_history_path)
    write_elo_plot(rc.elo_plot_path, plot_rows)
    return row


def count_cumulative_self_play_games(config):
    rc = config.resource
    cache = load_count_cache(rc.self_play_game_cache_path)
    updated_files = {}
    total_games = 0

    for path in iter_self_play_data_files(rc):
        try:
            stat_result = os.stat(path)
        except OSError:
            continue
        cache_key = os.path.abspath(path)
        signature = {
            "size": int(stat_result.st_size),
            "mtime_ns": int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1e9))),
        }
        cached = cache.get(cache_key)
        if cached and cached.get("size") == signature["size"] and cached.get("mtime_ns") == signature["mtime_ns"]:
            game_count = int(cached.get("games", 0))
        else:
            game_count = count_games_in_file(path)
        signature["games"] = game_count
        updated_files[cache_key] = signature
        total_games += game_count

    write_count_cache(rc.self_play_game_cache_path, updated_files)
    return total_games


def iter_self_play_data_files(resource):
    trained_dir = os.path.join(resource.data_dir, "trained")
    patterns = [
        os.path.join(resource.play_data_dir, resource.play_data_filename_tmpl % "*"),
        os.path.join(trained_dir, resource.play_data_filename_tmpl % "*"),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob(pattern))
    return sorted(set(files))


def count_games_in_file(path):
    try:
        with open(path, "rt", encoding="utf-8") as game_file:
            data = json.load(game_file)
    except (OSError, ValueError) as exc:
        logger.warning("Failed to read play data from %s: %s", path, exc)
        return 0

    if not isinstance(data, list):
        return 0
    if len(data) >= 3 and is_eval_record(data[0], data[1], data[2]):
        return 0
    return sum(1 for item in data if isinstance(item, str) and "/" in item)



def is_eval_record(first_item, second_item, third_item):
    return (
        isinstance(first_item, str)
        and isinstance(second_item, str)
        and isinstance(third_item, str)
        and "/" not in first_item
        and "/" not in second_item
        and "/" in third_item
    )
def estimate_match_elo(wins, losses, draws):
    total_games = int(wins) + int(losses) + int(draws)
    if total_games <= 0:
        return 0.0

    score = float(wins) + 0.5 * float(draws)
    expected_score = score / total_games
    if expected_score <= 0.0:
        expected_score = 0.5 / total_games
    elif expected_score >= 1.0:
        expected_score = 1.0 - (0.5 / total_games)

    relative_elo = -400.0 * math.log10((1.0 / expected_score) - 1.0)
    return round(relative_elo, 2)


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ELO_HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_plot_rows(path):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "rt", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                rows.append(
                    {
                        "timestamp": row.get("timestamp", ""),
                        "total_self_play_games": int(row["total_self_play_games"]),
                        "elo": float(row["elo"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def write_elo_plot(path, rows):
    if Image is None or ImageDraw is None or ImageFont is None:
        raise RuntimeError("Pillow is required to write elo_vs_games.png")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    width, height = 960, 540
    margin_left, margin_right = 95, 30
    margin_top, margin_bottom = 45, 70
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    image = Image.new("RGB", (width, height), PNG_BACKGROUND)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((plot_left, 12), "Elo vs cumulative training games", fill=PNG_TEXT, font=font)

    if rows:
        x_values = [row["total_self_play_games"] for row in rows]
        y_values = [row["elo"] for row in rows]
        x_min, x_max = expand_bounds(min(x_values), max(x_values), integer_axis=True)
        y_min, y_max = expand_bounds(min(y_values), max(y_values), integer_axis=False)

        draw_grid(draw, plot_left, plot_top, plot_right, plot_bottom, x_min, x_max, y_min, y_max, font)
        points = [
            (
                project(value=row["total_self_play_games"], low=x_min, high=x_max, start=plot_left, span=plot_width),
                project(value=row["elo"], low=y_min, high=y_max, start=plot_bottom, span=-plot_height),
            )
            for row in rows
        ]
        if len(points) == 1:
            x_pos, y_pos = points[0]
            draw.ellipse((x_pos - 4, y_pos - 4, x_pos + 4, y_pos + 4), fill=PNG_POINT, outline=PNG_POINT)
        else:
            draw.line(points, fill=PNG_LINE, width=3)
            for x_pos, y_pos in points:
                draw.ellipse((x_pos - 3, y_pos - 3, x_pos + 3, y_pos + 3), fill=PNG_POINT, outline=PNG_POINT)
    else:
        draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=PNG_AXIS, width=2)
        draw.text((plot_left + 20, plot_top + 20), "No Elo history recorded yet", fill=PNG_TEXT, font=font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=PNG_AXIS, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=PNG_AXIS, width=2)
    draw.text((plot_left + plot_width // 2 - 70, height - 28), "Cumulative self-play games", fill=PNG_TEXT, font=font)
    paste_vertical_text(image, "Elo", 14, plot_top + plot_height // 2 - 10, font)
    image.save(path, format="PNG")


def draw_grid(draw, left, top, right, bottom, x_min, x_max, y_min, y_max, font):
    for tick_index in range(5):
        tick_ratio = tick_index / 4
        x_pos = left + int((right - left) * tick_ratio)
        y_pos = bottom - int((bottom - top) * tick_ratio)
        x_value = x_min + (x_max - x_min) * tick_ratio
        y_value = y_min + (y_max - y_min) * tick_ratio

        draw.line((x_pos, top, x_pos, bottom), fill=PNG_GRID)
        draw.line((left, y_pos, right, y_pos), fill=PNG_GRID)
        draw.line((x_pos, bottom, x_pos, bottom + 5), fill=PNG_AXIS, width=1)
        draw.line((left - 5, y_pos, left, y_pos), fill=PNG_AXIS, width=1)
        draw.text((x_pos - 18, bottom + 10), format_tick(x_value), fill=PNG_TEXT, font=font)
        draw.text((left - 56, y_pos - 6), format_tick(y_value), fill=PNG_TEXT, font=font)


def paste_vertical_text(image, text, x_pos, y_pos, font):
    text_bbox = font.getbbox(text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    label = Image.new("RGBA", (text_width + 4, text_height + 4), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((2, 2), text, fill=PNG_TEXT, font=font)
    rotated = label.rotate(90, expand=True)
    image.paste(rotated, (x_pos, y_pos), rotated)


def project(value, low, high, start, span):
    if high == low:
        return start
    return int(round(start + ((value - low) / (high - low)) * span))


def expand_bounds(low, high, integer_axis):
    if low == high:
        if integer_axis:
            return low, low + 1
        return low - 1.0, high + 1.0

    padding = (high - low) * 0.05
    low -= padding
    high += padding
    if integer_axis:
        return int(math.floor(low)), int(math.ceil(high))
    return float(low), float(high)


def format_tick(value):
    if abs(value) >= 100 or value == int(value):
        return str(int(round(value)))
    return f"{value:.1f}"


def format_float(value):
    return f"{float(value):.2f}"


def utc_timestamp():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_count_cache(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rt", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, ValueError):
        return {}
    return payload.get("files", {})


def write_count_cache(path, files):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "updated_at": utc_timestamp(),
        "files": files,
    }
    with open(path, "wt", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2, sort_keys=True)