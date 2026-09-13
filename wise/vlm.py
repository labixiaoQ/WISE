"""Asynchronous VLM integration boundary (paper §3.1.2, Appendix A.2/A.9).

The online path submits keyframes and drains completed results without waiting.
The future worker owns batching (16 frames), timed flush (25 seconds), transport,
and output parsing. No network client or background worker starts in this release.
"""

from dataclasses import dataclass

from .errors import RuntimeNotAvailableError
from .keyframes import Keyframe
from .saeg import AffordanceRelation
from .types import Pose


@dataclass(frozen=True)
class ProposedAffordance:
    source: str
    relation: AffordanceRelation
    target: str


@dataclass(frozen=True)
class FrameSemantics:
    """Parsed, ungrounded output linked to its originating input keyframe."""

    observation_id: str
    timestep: int
    pose: Pose
    entities: tuple[str, ...]
    affordances: tuple[ProposedAffordance, ...]
    co_occurrences: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class VLMResult:
    request_id: str
    frames: tuple[FrameSemantics, ...]


@dataclass(frozen=True)
class SubmissionReceipt:
    """Explicit admission feedback; acceptance does not imply inference finished."""

    accepted_observation_ids: tuple[str, ...]
    deferred_observation_ids: tuple[str, ...]


class AsyncVLMQueue:
    """Non-blocking online API; a worker implementation is still to be migrated."""

    def submit(self, keyframes: tuple[Keyframe, ...]) -> SubmissionReceipt:
        """Enqueue only; must never perform or wait for inference on this path."""
        raise RuntimeNotAvailableError("AsyncVLMQueue.submit")

    def drain_ready(self) -> tuple[VLMResult, ...]:
        """Return completed outputs immediately, or an empty tuple when none exist."""
        raise RuntimeNotAvailableError("AsyncVLMQueue.drain_ready")
