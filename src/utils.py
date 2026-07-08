"""Shared helpers: config loading, global seeding, path resolution.

Kept tiny on purpose. Everything reads the single config.yaml so run.py is the
one source of truth for reproducibility.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Project root = parent of this src/ directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml (defaults to the one at the project root)."""
    cfg_path = Path(path) if path is not None else PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = SEED) -> None:
    """Seed every source of randomness we touch (python, numpy)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
