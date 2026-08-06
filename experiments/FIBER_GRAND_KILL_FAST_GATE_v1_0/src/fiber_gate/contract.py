from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def decision_contract() -> dict[str, Any]:
    return json.loads((package_root() / "config" / "decision_contract.json").read_text(encoding="utf-8"))


def profiles() -> dict[str, Any]:
    return json.loads((package_root() / "config" / "profiles.json").read_text(encoding="utf-8"))
