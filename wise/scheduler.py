"""Opportunistic task scheduling boundary (paper §3.2, Eqs. 7–10; Appendix C)."""

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import RuntimeNotAvailableError
from .retrieval import RetrievalCandidate
from .types import Observation, Subtask, TaskQueue


@dataclass(frozen=True)
class PriorityTerms:
    """Normalized urgency, affordance relevance, and navigation cost."""

    urgency: float
    affordance_relevance: float
    navigation_cost: float


@dataclass(frozen=True)
class ScheduleDecision:
    selected_task: Subtask
    priority: float
    terms: PriorityTerms
    preempt_active_task: bool


class OpportunisticTaskScheduler:
    def select(
        self,
        observation: Observation,
        queue: TaskQueue,
        candidates: Mapping[str, tuple[RetrievalCandidate, ...]],
        *,
        active_task_id: str | None = None,
    ) -> ScheduleDecision | None:
        """Choose a pending occurrence under the queue's explicit order policy.

        FIXED (ABA-Sparse): only the first pending task is eligible, regardless
        of other opportunities. REORDERABLE (ABC-Sparse): score all pending
        tasks by Eqs. 7–10 and permit preemption. Candidate mappings use task
        IDs rather than instruction strings so repeated goals stay distinct.
        Return None only for an empty queue. Validate the chosen occurrence
        using TaskQueue.validate_selection before dispatching it.
        """
        raise RuntimeNotAvailableError("OpportunisticTaskScheduler.select")
