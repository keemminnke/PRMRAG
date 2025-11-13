"""Tests for data schemas and loaders."""

import pytest
from pathlib import Path
import tempfile
from prmrag.data.schemas import Trajectory, TrajectoryStep, ActionType
from prmrag.data.loaders import save_trajectories, load_trajectories


def test_trajectory_serialization(example_trajectory):
    """Test trajectory to_dict and from_dict."""
    traj_dict = example_trajectory.to_dict()

    assert traj_dict["trajectory_id"] == "test_001"
    assert traj_dict["question"] == "What is 2 + 2?"
    assert len(traj_dict["steps"]) == 2

    # Reconstruct
    traj_restored = Trajectory.from_dict(traj_dict)

    assert traj_restored.trajectory_id == example_trajectory.trajectory_id
    assert traj_restored.question == example_trajectory.question
    assert len(traj_restored.steps) == len(example_trajectory.steps)


def test_trajectory_prefix(example_trajectory):
    """Test getting trajectory prefix."""
    prefix = example_trajectory.get_prefix(0)

    assert len(prefix) == 1
    assert prefix[0].step_id == 0

    prefix_full = example_trajectory.get_prefix(1)
    assert len(prefix_full) == 2


def test_save_and_load_trajectories(example_trajectory):
    """Test saving and loading trajectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_trajectories.jsonl"

        # Save
        save_trajectories([example_trajectory], output_path, format="jsonl")

        assert output_path.exists()

        # Load
        loaded = load_trajectories(output_path, format="jsonl")

        assert len(loaded) == 1
        assert loaded[0].trajectory_id == example_trajectory.trajectory_id
        assert loaded[0].question == example_trajectory.question
