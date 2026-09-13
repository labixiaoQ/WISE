"""Short-term geometric memory boundary (paper §3.1.1).

The intended integration uses MineCLIP's 16-frame encoding window, geometric
place clusters, and DP-Means event clusters following MrSteve's PEM design.
This interface does not wrap baseline retrieval as if it implemented WISE.
"""

from .errors import RuntimeNotAvailableError
from .types import Embedding, MemoryFrame, Observation


class ShortTermGeometricMemory:
    """Own recent embeddings, spatial/event indexing, and memory eviction."""

    def update(self, observation: Observation) -> None:
        """Encode/store the observation and apply the configured eviction rule."""
        raise RuntimeNotAvailableError("ShortTermGeometricMemory.update")

    def retrieve(self, task_embedding: Embedding, *, top_k: int) -> tuple[MemoryFrame, ...]:
        """Return visually matched recent memories using the geometric index."""
        raise RuntimeNotAvailableError("ShortTermGeometricMemory.retrieve")

    def snapshot(self) -> tuple[MemoryFrame, ...]:
        """Expose recent memory cues for hybrid keyframe selection."""
        raise RuntimeNotAvailableError("ShortTermGeometricMemory.snapshot")
