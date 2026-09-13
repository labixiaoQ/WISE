"""Hybrid cluster/entropy keyframe selection boundary (paper §3.1.2)."""

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

from .errors import RuntimeNotAvailableError
from .types import MemoryFrame, Observation


class SelectionReason(str, Enum):
    CLUSTER_REPRESENTATIVE = "cluster_representative"
    IMAGE_ENTROPY = "image_entropy"


@dataclass(frozen=True)
class Keyframe:
    observation: Observation
    reasons: tuple[SelectionReason, ...]


class HybridKeyframeSelector:
    def select(
        self, observation: Observation, recent: Sequence[MemoryFrame]
    ) -> tuple[Keyframe, ...]:
        """Merge representative/entropy cues and remove near-duplicate frames."""
        raise RuntimeNotAvailableError("HybridKeyframeSelector.select")
