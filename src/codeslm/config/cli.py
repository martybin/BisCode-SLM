"""Validate the shipped YAML configs."""

from __future__ import annotations

import sys

from codeslm.config.errors import ConfigError
from codeslm.config.schema import find_project_root, launch_blockers, load_model_config, load_training_config


def main(argv: list[str] | None = None) -> int:
    """Validate the baseline, smoke, and data configs. Return a process status."""
    del argv
    try:
        root = find_project_root(__file__)
        base = load_model_config(root / "configs" / "model" / "base.yaml")
        smoke = load_model_config(root / "configs" / "model" / "smoke.yaml")
        training = load_training_config(root / "configs" / "training" / "smoke_test.yaml", root)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    blockers = launch_blockers(training)
    print(f"base {base.name} parameters {base.parameter_count:,}")
    print(f"smoke {smoke.name} parameters {smoke.parameter_count:,}")
    print(
        f"training {training.experiment_name} seq_len {training.batch.seq_len} "
        f"max_tokens {training.budget.max_tokens}"
    )
    print(
        f"data {training.data.name} download_enabled {str(training.data.download_enabled).lower()} "
        f"train_sources {len(training.data.enabled_sources('train'))}"
    )
    if blockers:
        print("launch blockers:")
        for blocker in blockers:
            print(f"- {blocker}")
    print("configs ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
