"""Coarse-to-fine exploration interfaces (paper §3.3, Appendix A.1).

Quadtree, frontier, and Voronoi algorithms are migration boundaries. No fallback
random walk or baseline goal selector is presented as the WISE explorer.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .errors import RuntimeNotAvailableError
from .types import Observation, Pose


@dataclass(frozen=True)
class MapBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float


@dataclass(frozen=True)
class Region:
    region_id: str
    bounds: MapBounds
    depth: int


@dataclass(frozen=True)
class Frontier:
    frontier_id: str
    pose: Pose
    uncovered_area: float
    novelty: float


@dataclass(frozen=True)
class ResidualGap:
    gap_id: str
    target: Pose
    area: float


class QuadtreeIndex(Protocol):
    """Global macro-region lookup and target selection (Eq. 11)."""

    def update(self, observation: Observation) -> None: ...

    def select_region(self, current_pose: Pose) -> Region | None: ...


class FrontierRefiner(Protocol):
    """Regional boundary expansion after macro-region arrival (Eq. 12)."""

    def select_frontier(self, region: Region, current_pose: Pose) -> Frontier | None: ...


class VoronoiCompleter(Protocol):
    """Local completion targeting residual unobserved interior gaps (Eq. 13)."""

    def select_gap(self, region: Region, current_pose: Pose) -> ResidualGap | None: ...


class ExplorationTier(str, Enum):
    GLOBAL = "global_quadtree"
    REGIONAL = "regional_frontier"
    LOCAL = "local_voronoi"


@dataclass(frozen=True)
class ExplorationTarget:
    pose: Pose
    tier: ExplorationTier
    region_id: str


class MultiScaleExplorer:
    def select_target(self, observation: Observation) -> ExplorationTarget | None:
        """Update coverage and select a target across the three spatial tiers.

        The migrated implementation also owns tier transitions and stagnation
        replanning. Coverage is derived from observed/visited map cells, not
        privileged entity or terrain metadata.
        """
        raise RuntimeNotAvailableError("MultiScaleExplorer.select_target")
