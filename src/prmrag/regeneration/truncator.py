"""Trajectory truncator for critic-guided regeneration.

Finds the first BAD step using the critic model's binary classification
(critic_score <= 0.5 → BAD), truncates the trajectory at that point.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from .classifier import ScoredTrajectory


@dataclass
class TruncationResult:
    """Result of truncating a trajectory at the first BAD step."""
    trajectory_id: str
    question_id: str
    question: str
    gold_answer: str
    good_steps: List[Dict[str, Any]]  # steps before the BAD step
    bad_step_id: int  # step_id of the first BAD step
    bad_step_critic_score: float
    original_num_steps: int
    original_predicted_answer: str
    original_is_correct: bool


class TrajectoryTruncator:
    """Truncate trajectories at the first BAD step detected by critic binary classification."""

    def truncate(self, scored_trajectory: ScoredTrajectory) -> TruncationResult:
        """Truncate trajectory at the first step classified as BAD by critic.

        Critic model is a binary classifier: score > 0.5 → GOOD, score <= 0.5 → BAD.
        Finds the first BAD step, keeps all prior steps as good_steps.

        Args:
            scored_trajectory: Trajectory with critic scores

        Returns:
            TruncationResult with good_steps and bad_step info
        """
        steps = scored_trajectory.steps
        critic_scores = scored_trajectory.critic_step_scores

        # Find first step classified as BAD by critic (binary classification boundary = 0.5)
        bad_step_idx = None
        bad_step_score = None
        for i, score in enumerate(critic_scores):
            if score <= 0.5:
                bad_step_idx = i
                bad_step_score = score
                break

        # Edge case: critic classifies all steps as GOOD but trajectory is still wrong
        # Truncate at the last step (answer step)
        if bad_step_idx is None:
            bad_step_idx = len(steps) - 1
            bad_step_score = critic_scores[bad_step_idx] if bad_step_idx < len(critic_scores) else 0.0

        # Good steps: everything before the BAD step
        good_steps = steps[:bad_step_idx]

        bad_step = steps[bad_step_idx]
        bad_step_id = bad_step.get('step_id', bad_step_idx + 1)

        return TruncationResult(
            trajectory_id=scored_trajectory.trajectory_id,
            question_id=scored_trajectory.question_id,
            question=scored_trajectory.question,
            gold_answer=scored_trajectory.gold_answer,
            good_steps=good_steps,
            bad_step_id=bad_step_id,
            bad_step_critic_score=bad_step_score,
            original_num_steps=len(steps),
            original_predicted_answer=scored_trajectory.predicted_answer,
            original_is_correct=scored_trajectory.is_correct,
        )
