"""WISE's orchestration boundary (paper Algorithm 1).

The intended decision step updates geometric memory, submits eligible keyframes,
drains and grounds ready semantic outputs, retrieves task cues, schedules an
eligible task, then navigates/executes or explores. VLM inference never belongs
to the blocking action path. This release declares that composition only.
"""

from dataclasses import dataclass

from .controllers import EnvironmentAction, InstructionController, NavigationController
from .errors import RuntimeNotAvailableError
from .exploration import MultiScaleExplorer
from .grounding import SemanticGrounder, TaskResourceVocabulary
from .keyframes import HybridKeyframeSelector
from .memory import ShortTermGeometricMemory
from .retrieval import TwoLevelRetriever
from .saeg import SemanticAffordanceEventGraph
from .scheduler import OpportunisticTaskScheduler
from .types import Observation, TaskQueue
from .vlm import AsyncVLMQueue


@dataclass
class WISEAgent:
    """Explicit component composition for integration after runtime migration."""

    short_term: ShortTermGeometricMemory
    saeg: SemanticAffordanceEventGraph
    keyframes: HybridKeyframeSelector
    vlm_queue: AsyncVLMQueue
    grounder: SemanticGrounder
    vocabulary: TaskResourceVocabulary
    retriever: TwoLevelRetriever
    scheduler: OpportunisticTaskScheduler
    explorer: MultiScaleExplorer
    navigator: NavigationController
    executor: InstructionController

    def step(self, observation: Observation, tasks: TaskQueue) -> EnvironmentAction:
        """Execute one Algorithm 1 decision step once the runtime is migrated."""
        raise RuntimeNotAvailableError("WISEAgent.step")
