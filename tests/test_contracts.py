"""Structural release checks; these do not evaluate the paper's algorithms."""

import subprocess
import sys
import unittest

from wise.errors import RuntimeNotAvailableError
from wise.exploration import MultiScaleExplorer
from wise.types import Observation, Pose, RGBFrame, Subtask, TaskOrder, TaskQueue


class TaskOrderContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = (
            Subtask("a1", "find water"),
            Subtask("b", "collect logs"),
            Subtask("a2", "find water"),
        )

    def test_fixed_aba_preserves_distinct_repeated_occurrences(self) -> None:
        queue = TaskQueue(self.tasks, TaskOrder.FIXED)
        queue.validate_selection("a1")
        with self.assertRaisesRegex(ValueError, "first task"):
            queue.validate_selection("a2")

    def test_reorderable_queue_accepts_any_pending_task_only(self) -> None:
        queue = TaskQueue(self.tasks, TaskOrder.REORDERABLE)
        queue.validate_selection("a2")
        with self.assertRaisesRegex(ValueError, "not pending"):
            queue.validate_selection("absent")

    def test_duplicate_occurrence_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique task_id"):
            TaskQueue((self.tasks[0], self.tasks[0]), TaskOrder.FIXED)


class ReleaseBoundaryTests(unittest.TestCase):
    def test_runtime_stub_never_returns_an_invented_target(self) -> None:
        observation = Observation(
            observation_id="contract-check",
            rgb=RGBFrame(pixels=bytes(3), width=1, height=1),
            pose=Pose(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0),
            timestep=0,
        )
        with self.assertRaises(RuntimeNotAvailableError):
            MultiScaleExplorer().select_target(observation)

    def test_describe_requires_only_standard_library(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", "-m", "wise", "--describe"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("awaiting migration", result.stdout)
        self.assertIn("fixed_order", result.stdout)

    def test_evaluation_fails_before_loading_baseline_dependencies(self) -> None:
        result = subprocess.run(
            [sys.executable, "-S", "-m", "wise", "--evaluate"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("does not run paper evaluations", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
