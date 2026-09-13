"""Semantic Affordance Event Graph contracts (paper §3.1.2, Appendix A.8).

Semantic nodes describe normalized concepts; spatial instances retain where and
when observations occurred. Only grounded relations may enter a graph update.
"""

from dataclasses import dataclass
from enum import Enum

from .errors import RuntimeNotAvailableError
from .types import Pose


class NodeKind(str, Enum):
    ENTITY = "entity"
    ACTION = "action"
    ENVIRONMENT = "environment"


class AffordanceRelation(str, Enum):
    CAN_OBTAIN = "CAN_OBTAIN"
    CAN_CRAFT = "CAN_CRAFT"
    CAN_SMELT = "CAN_SMELT"


@dataclass(frozen=True)
class EntityNode:
    node_id: str
    normalized_name: str
    kind: NodeKind = NodeKind.ENTITY


@dataclass(frozen=True)
class SpatialInstance:
    instance_id: str
    entity_node_id: str
    observation_id: str
    pose: Pose
    timestep: int


@dataclass(frozen=True)
class AffordanceEdge:
    """For example: observed cow instance → CAN_OBTAIN → beef resource node."""

    source_instance_id: str
    relation: AffordanceRelation
    target_node_id: str


@dataclass(frozen=True)
class CoOccurrenceEdge:
    """CO_OCCURS_WITH between observed spatial instances."""

    source_instance_id: str
    target_instance_id: str
    timestep: int


@dataclass(frozen=True)
class GroundedGraphUpdate:
    """Output of vocabulary/mechanics filtering, ready for synchronized commit."""

    entities: tuple[EntityNode, ...]
    instances: tuple[SpatialInstance, ...]
    affordances: tuple[AffordanceEdge, ...]
    co_occurrences: tuple[CoOccurrenceEdge, ...]


@dataclass(frozen=True)
class GraphSnapshot:
    """Latest committed graph version; readers do not wait for pending inference."""

    revision: int
    entities: tuple[EntityNode, ...]
    instances: tuple[SpatialInstance, ...]
    affordances: tuple[AffordanceEdge, ...]
    co_occurrences: tuple[CoOccurrenceEdge, ...]


class SemanticAffordanceEventGraph:
    def apply(self, update: GroundedGraphUpdate) -> None:
        """Commit a grounded update with duplicate-instance merging (§3.1.2)."""
        raise RuntimeNotAvailableError("SemanticAffordanceEventGraph.apply")

    def snapshot(self) -> GraphSnapshot:
        """Return the latest committed graph, independent of pending VLM work."""
        raise RuntimeNotAvailableError("SemanticAffordanceEventGraph.snapshot")
