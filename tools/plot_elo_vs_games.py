import argparse
import os
import sys

_PATH_ = os.path.dirname(os.path.dirname(__file__))

if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

from cchess_alphazero.config import Config
from cchess_alphazero.lib.training_monitor import load_plot_rows, write_elo_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="store checkpoints and play data under this directory")
    parser.add_argument("--type", default="local_torch", help="config type (mini, normal, local_torch, distribute)")
    args = parser.parse_args()

    config = Config(config_type=args.type)
    if args.data_dir:
        config.resource.update_paths(data_dir=os.path.abspath(args.data_dir))
    config.resource.create_directories()

    rows = load_plot_rows(config.resource.elo_history_path)
    write_elo_plot(config.resource.elo_plot_path, rows)
    print(f"Wrote {config.resource.elo_plot_path} from {len(rows)} row(s) in {config.resource.elo_history_path}")


if __name__ == "__main__":
    main()