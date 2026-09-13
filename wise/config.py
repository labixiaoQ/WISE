"""Dependency-free inspection of paper configuration manifests.

Parsing a manifest does not instantiate an agent or validate runtime readiness.
These manifests are independent of the inherited baseline's Hydra configuration.
"""

from pathlib import Path
from typing import Any
import tomllib


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a TOML manifest without importing simulator/model dependencies."""
    with Path(path).open("rb") as stream:
        return tomllib.load(stream)
