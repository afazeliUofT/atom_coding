from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def decision_contract() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "config" / "decision_contract.json"
    return json.loads(path.read_text(encoding="utf-8"))
