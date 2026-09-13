"""WISE: Which-Why Informed Semantic Explorer.

This release exposes the paper's architecture and integration contracts. The
trained controllers and WISE runtime implementation are awaiting migration.
Importing this namespace never imports or initializes the MrSteve baseline.
"""

from .agent import WISEAgent
from .errors import RuntimeNotAvailableError

__version__ = "0.1.0"
__all__ = ["WISEAgent", "RuntimeNotAvailableError", "__version__"]
