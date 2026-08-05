#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atom_gate.run_gate import run  # noqa: E402


if __name__ == "__main__":
    profile = os.environ.get("ATOM_GATE_PROFILE", "standard")
    raise SystemExit(run(profile))
