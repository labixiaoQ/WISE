"""Shared observation, task, and memory contracts (paper §3, Eqs. 1–5)."""

from dataclasses import dataclass
from enum import Enum

Embedding = tuple[float, ...]


@dataclass(frozen=True)
class Pose:
    """Position in blocks and camera yaw/pitch in degrees, relative to origin."""

    x: float
    y: float
    z: float
    yaw: float
    pitch: float


@dataclass(frozen=True)
class RGBFrame:
    """Packed RGB bytes; environment adapters own conversion from image tensors."""

    pixels: bytes
    width: int
    height: int


@dataclass(frozen=True)
class Observation:
    observation_id: str
    rgb: RGBFrame
    pose: Pose
    timestep: int


@dataclass(frozen=True)
class MemoryFrame:
    """A stored visual embedding, pose, and timestamp (Eq. 5)."""

    observation_id: str
    embedding: Embedding
    pose: Pose
    timestep: int


@dataclass(frozen=True)
class Subtask:
    """An occurrence of a goal; repeated ABA goals require distinct task IDs."""

    task_id: str
    instruction: str
    target_resource: str | None = None


class TaskOrder(str, Enum):
    FIXED = "fixed_order"
    REORDERABLE = "reorderable"


@dataclass(frozen=True)
class TaskQueue:
    """Pending occurrences in original order; policy follows Appendix C.

    ABA-Sparse uses FIXED and requires A1 → B → A2. ABC-Sparse uses
    REORDERABLE and allows any pending occurrence to be selected.
    Completion detection and queue updates belong to the migrated runtime.
    """

    tasks: tuple[Subtask, ...]
    order: TaskOrder

    def __post_init__(self) -> None:
        ids = tuple(task.task_id for task in self.tasks)
        if len(set(ids)) != len(ids):
            raise ValueError("Every task occurrence must have a unique task_id.")
        if not isinstance(self.order, TaskOrder):
            raise TypeError("order must be a TaskOrder value.")

    def validate_selection(self, task_id: str) -> None:
        """Enforce task-order contracts without implementing task selection."""
        if task_id not in {task.task_id for task in self.tasks}:
            raise ValueError(f"Task {task_id!r} is not pending.")
        if self.order is TaskOrder.FIXED and task_id != self.tasks[0].task_id:
            raise ValueError("A fixed-order queue can select only its first task.")


class AgentMode(str, Enum):
    EXPLORE = "explore"
    EXECUTE = "execute"
