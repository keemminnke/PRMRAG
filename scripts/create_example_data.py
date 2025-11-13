#!/usr/bin/env python3
"""
Create example trajectory data for testing the pipeline.

This generates synthetic RAG-CoT trajectories in the expected format.
"""

import json
import jsonlines
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.data.schemas import Trajectory, TrajectoryStep, ActionType


def create_example_trajectories() -> list:
    """Create example trajectories for testing."""

    trajectories = []

    # Example 1: Good trajectory
    traj1 = Trajectory(
        trajectory_id="example_001",
        question="What is the capital of France and what is its population?",
        gold_answer="Paris, approximately 2.2 million",
        supporting_facts=[
            "Paris is the capital of France.",
            "Paris has a population of approximately 2.2 million people within city limits.",
        ],
        steps=[
            TrajectoryStep(
                step_id=0,
                action_type=ActionType.RETRIEVE,
                action="Search for: capital of France",
                observation="Found information about France's capital",
                passages=[
                    "Paris is the capital and most populous city of France.",
                    "France is a country in Western Europe with Paris as its capital.",
                ],
            ),
            TrajectoryStep(
                step_id=1,
                action_type=ActionType.REASON,
                action="The capital is Paris. Now I need to find its population.",
                observation="Identified capital, proceeding to find population",
            ),
            TrajectoryStep(
                step_id=2,
                action_type=ActionType.RETRIEVE,
                action="Search for: population of Paris",
                observation="Found population data",
                passages=[
                    "Paris has a population of approximately 2.2 million people within city limits.",
                    "The Paris metropolitan area has over 12 million inhabitants.",
                ],
            ),
            TrajectoryStep(
                step_id=3,
                action_type=ActionType.ANSWER,
                action="Based on the information, the answer is Paris with population ~2.2 million",
                observation="Final answer formulated",
            ),
        ],
        final_answer="Paris, approximately 2.2 million",
        is_correct=True,
    )
    trajectories.append(traj1)

    # Example 2: Trajectory with some bad steps
    traj2 = Trajectory(
        trajectory_id="example_002",
        question="Who wrote the Harry Potter series?",
        gold_answer="J.K. Rowling",
        supporting_facts=[
            "J.K. Rowling is the author of the Harry Potter series.",
        ],
        steps=[
            TrajectoryStep(
                step_id=0,
                action_type=ActionType.RETRIEVE,
                action="Search for: Harry Potter author",
                observation="Found relevant information",
                passages=[
                    "J.K. Rowling is a British author, best known for the Harry Potter series.",
                    "The Harry Potter series consists of seven books.",
                ],
            ),
            TrajectoryStep(
                step_id=1,
                action_type=ActionType.RETRIEVE,
                action="Search for: Harry Potter movies director",  # Bad step - irrelevant
                observation="Found information about movies",
                passages=[
                    "Chris Columbus directed the first two Harry Potter films.",
                    "The Harry Potter films were produced by Warner Bros.",
                ],
            ),
            TrajectoryStep(
                step_id=2,
                action_type=ActionType.ANSWER,
                action="The author is J.K. Rowling",
                observation="Final answer",
            ),
        ],
        final_answer="J.K. Rowling",
        is_correct=True,
    )
    trajectories.append(traj2)

    # Example 3: Incorrect trajectory
    traj3 = Trajectory(
        trajectory_id="example_003",
        question="What is the largest planet in our solar system?",
        gold_answer="Jupiter",
        supporting_facts=[
            "Jupiter is the largest planet in our solar system.",
        ],
        steps=[
            TrajectoryStep(
                step_id=0,
                action_type=ActionType.RETRIEVE,
                action="Search for: planets in solar system",
                observation="Found list of planets",
                passages=[
                    "The planets in our solar system are Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
                ],
            ),
            TrajectoryStep(
                step_id=1,
                action_type=ActionType.REASON,
                action="Saturn has the most impressive rings, so it must be the largest",  # Bad reasoning
                observation="Concluded Saturn is largest",
            ),
            TrajectoryStep(
                step_id=2,
                action_type=ActionType.ANSWER,
                action="The largest planet is Saturn",
                observation="Final answer",
            ),
        ],
        final_answer="Saturn",
        is_correct=False,
    )
    trajectories.append(traj3)

    return trajectories


def main():
    output_path = Path("data/raw/example_trajectories.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectories = create_example_trajectories()

    print(f"Creating {len(trajectories)} example trajectories...")

    with jsonlines.open(output_path, mode='w') as writer:
        for traj in trajectories:
            writer.write(traj.to_dict())

    print(f"✓ Saved to {output_path}")
    print(f"\nExample usage:")
    print(f"  python scripts/label_dataset.py \\")
    print(f"    --config configs/default.yaml \\")
    print(f"    --input {output_path} \\")
    print(f"    --output data/labeled/example_labeled.jsonl")


if __name__ == "__main__":
    main()
