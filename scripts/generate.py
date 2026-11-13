#!/usr/bin/env python3
"""Generate a completion from a local checkpoint and tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeslm.inference.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
