"""Launch gate for training.

This command loads a training config and refuses to start a run. The shipped
smoke config stays blocked while dataset download is disabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from codeslm.config.errors import ConfigError
from codeslm.config.schema import find_project_root, launch_blockers, load_training_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether a training config is allowed to run.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        root = find_project_root(args.config if args.config.is_absolute() else Path.cwd())
        config_path = args.config if args.config.is_absolute() else root / args.config
        training = load_training_config(config_path, root)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    blockers = launch_blockers(training)
    if blockers:
        for blocker in blockers:
            print(blocker, file=sys.stderr)
        print("training was not started", file=sys.stderr)
        return 2
    print(
        "the config has no launch blocker; this command still does not start a training job",
        file=sys.stderr,
    )
    return 2
