"""Config persistence: ~/.omegafm/config.yaml (devices, preset, trims, RDS)."""

from __future__ import annotations

from pathlib import Path
import yaml

CONFIG_DIR = Path.home() / ".omegafm"
CONFIG_FILE = CONFIG_DIR / "config.yaml"


def load() -> dict:
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save(data: dict):
    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception:
        pass
