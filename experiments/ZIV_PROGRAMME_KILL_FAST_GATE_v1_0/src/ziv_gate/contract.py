from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def decision_contract() -> dict[str, Any]:
    return read_json(package_root() / "config" / "decision_contract.json")


def profiles() -> dict[str, Any]:
    return read_json(package_root() / "config" / "profiles.json")
