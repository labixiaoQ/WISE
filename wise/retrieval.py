"""Two-level visual/affordance retrieval boundary (paper §3.1.3, Eq. 6)."""

from dataclasses import dataclass

from .errors import RuntimeNotAvailableError
from .memory import ShortTermGeometricMemory
from .saeg import AffordanceEdge, GraphSnapshot
from .types import Embedding, MemoryFrame, Subtask


@dataclass(frozen=True)
class RetrievalCandidate:
    """A memory cue and explicit evidence explaining task relevance."""

    memory: MemoryFrame
    visual_similarity: float
    affordance_match: bool
    affordance_evidence: tuple[AffordanceEdge, ...]
    combined_score: float


class TwoLevelRetriever:
    def retrieve(
        self,
        task: Subtask,
        task_embedding: Embedding,
        short_term: ShortTermGeometricMemory,
        graph: GraphSnapshot,
    ) -> tuple[RetrievalCandidate, ...]:
        """Merge both candidate sets and rank by Eq. 6; empty means explore.

        Visual retrieval supplies top-k geometric cues; graph traversal follows
        incoming affordance edges from task outcomes to observed instances.
        The returned order is descending combined score.
        """
        raise RuntimeNotAvailableError("TwoLevelRetriever.retrieve")
