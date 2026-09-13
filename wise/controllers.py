"""Steve-1 and VPT-Nav adapter boundaries (paper Fig. 2, Algorithm 1).

SAEG provides task selection and navigation targets; it does not directly alter
the low-level policies' inputs. Checkpoint loading belongs to migrated adapters.
The inherited mrsteve namespace remains a separate baseline implementation.
"""

from collections.abc import Mapping
from typing import Protocol, TypeAlias

from .errors import RuntimeNotAvailableError
from .types import Observation, Pose

ActionValue: TypeAlias = bool | int | float | tuple[float, ...]
EnvironmentAction: TypeAlias = Mapping[str, ActionValue]


class NavigationController(Protocol):
    def act(self, observation: Observation, target: Pose) -> EnvironmentAction: ...


class InstructionController(Protocol):
    def act(self, observation: Observation, instruction: str) -> EnvironmentAction: ...


class VPTNavAdapter:
    def act(self, observation: Observation, target: Pose) -> EnvironmentAction:
        """Produce a goal-directed navigation action using the VPT-Nav policy."""
        raise RuntimeNotAvailableError("VPTNavAdapter.act")


class Steve1Adapter:
    def act(self, observation: Observation, instruction: str) -> EnvironmentAction:
        """Produce an instruction-conditioned action using the Steve-1 policy."""
        raise RuntimeNotAvailableError("Steve1Adapter.act")
