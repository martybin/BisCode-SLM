#!/usr/bin/env python3
"""Check a training config. This command does not start a run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeslm.training.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
