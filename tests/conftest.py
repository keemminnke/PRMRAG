"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.data.schemas import Trajectory, TrajectoryStep, ActionType


@pytest.fixture
def example_trajectory():
    """Create an example trajectory for testing."""
    return Trajectory(
        trajectory_id="test_001",
        question="What is 2 + 2?",
        gold_answer="4",
        supporting_facts=["2 + 2 equals 4"],
        steps=[
            TrajectoryStep(
                step_id=0,
                action_type=ActionType.REASON,
                action="Add 2 and 2",
                observation="Result is 4",
            ),
            TrajectoryStep(
                step_id=1,
                action_type=ActionType.ANSWER,
                action="The answer is 4",
                observation="Final answer",
            ),
        ],
        final_answer="4",
        is_correct=True,
    )


@pytest.fixture
def example_config():
    """Create an example configuration for testing."""
    return {
        "rpe": {
            "model_name": "test-model",
            "num_rollouts": 8,
            "threshold_good": 0.8,
            "threshold_bad": 0.3,
            "temperature": 0.8,
            "device": "cpu",
        },
        "judge": {
            "model_name": "test-judge",
            "temperature": 0.3,
            "max_tokens": 256,
            "max_retries": 1,
            "retry_delay": 1.0,
            "prompt_style": "versaprm",
            "use_gold_answer": True,
            "use_supporting_facts": True,
        },
        "consensus": {
            "strategy": "strict",
            "trajectory_level": True,
            "min_agreement": 0.8,
            "rules": {
                "positive_consensus": {
                    "rpe": ["GOOD"],
                    "judge": ["GOOD"],
                    "label": 1,
                },
                "negative_consensus": {
                    "rpe": ["BAD"],
                    "judge": ["BAD"],
                    "label": 0,
                },
                "filter_disagreement": True,
            },
        },
    }
