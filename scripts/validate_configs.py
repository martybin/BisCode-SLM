#!/usr/bin/env python3
"""Validate the shipped configs without installing the package."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeslm.config.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
