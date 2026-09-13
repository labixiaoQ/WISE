"""Vocabulary and Minecraft-mechanics grounding boundary (Appendix A.8/A.9)."""

from typing import Protocol

from .errors import RuntimeNotAvailableError
from .saeg import AffordanceRelation, GroundedGraphUpdate
from .vlm import VLMResult


class TaskResourceVocabulary(Protocol):
    """Adapter for the paper's task-resource database, which is not bundled yet."""

    def normalize(self, name: str) -> str | None:
        """Return a known normalized concept, or None for an unsupported name."""
        ...

    def permits(self, source: str, relation: AffordanceRelation, target: str) -> bool:
        """Validate a proposed relation against Minecraft resource mechanics."""
        ...


class SemanticGrounder:
    def ground(
        self, result: VLMResult, vocabulary: TaskResourceVocabulary
    ) -> GroundedGraphUpdate:
        """Normalize names, reject unknown concepts, and validate observed sources.

        The migrated implementation must preserve input observation provenance,
        merge nearby same-type instances, and retain only mechanics-valid edges.
        """
        raise RuntimeNotAvailableError("SemanticGrounder.ground")
