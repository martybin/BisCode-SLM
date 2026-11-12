#!/usr/bin/env python3
"""Train a byte-level BPE tokenizer from local UTF-8 files."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from codeslm.tokenizer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
