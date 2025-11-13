"""Tests for consensus module."""

import pytest
from prmrag.data.schemas import RPELabel, JudgeLabel, LabelType
from prmrag.labeling.consensus import ConsensusModule


def test_positive_consensus(example_trajectory, example_config):
    """Test positive consensus (both GOOD)."""
    consensus_module = ConsensusModule(example_config["consensus"])

    rpe_labels = [
        RPELabel(step_id=0, mc_s_t=0.5, mc_s_t_a_t=0.9, rpe=1.8, label=LabelType.GOOD),
        RPELabel(step_id=1, mc_s_t=0.6, mc_s_t_a_t=0.95, rpe=1.58, label=LabelType.GOOD),
    ]

    judge_labels = [
        JudgeLabel(step_id=0, label="GOOD", reasoning="Good step", confidence=0.9),
        JudgeLabel(step_id=1, label="GOOD", reasoning="Good final answer", confidence=0.95),
    ]

    labeled_traj = consensus_module.create_consensus_labels(
        example_trajectory, rpe_labels, judge_labels
    )

    assert not labeled_traj.is_filtered
    assert len(labeled_traj.consensus_labels) == 2
    assert all(c.consensus_label == 1 for c in labeled_traj.consensus_labels)
    assert all(c.is_consensus for c in labeled_traj.consensus_labels)


def test_negative_consensus(example_trajectory, example_config):
    """Test negative consensus (both BAD)."""
    consensus_module = ConsensusModule(example_config["consensus"])

    rpe_labels = [
        RPELabel(step_id=0, mc_s_t=0.8, mc_s_t_a_t=0.2, rpe=0.25, label=LabelType.BAD),
        RPELabel(step_id=1, mc_s_t=0.7, mc_s_t_a_t=0.1, rpe=0.14, label=LabelType.BAD),
    ]

    judge_labels = [
        JudgeLabel(step_id=0, label="BAD", reasoning="Bad step", confidence=0.8),
        JudgeLabel(step_id=1, label="BAD", reasoning="Wrong answer", confidence=0.9),
    ]

    labeled_traj = consensus_module.create_consensus_labels(
        example_trajectory, rpe_labels, judge_labels
    )

    assert not labeled_traj.is_filtered
    assert all(c.consensus_label == 0 for c in labeled_traj.consensus_labels)
    assert all(c.is_consensus for c in labeled_traj.consensus_labels)


def test_disagreement_filters(example_trajectory, example_config):
    """Test that disagreements are filtered."""
    consensus_module = ConsensusModule(example_config["consensus"])

    rpe_labels = [
        RPELabel(step_id=0, mc_s_t=0.5, mc_s_t_a_t=0.9, rpe=1.8, label=LabelType.GOOD),
        RPELabel(step_id=1, mc_s_t=0.8, mc_s_t_a_t=0.2, rpe=0.25, label=LabelType.BAD),
    ]

    judge_labels = [
        JudgeLabel(step_id=0, label="BAD", reasoning="Disagree - bad step", confidence=0.8),
        JudgeLabel(step_id=1, label="GOOD", reasoning="Disagree - good step", confidence=0.9),
    ]

    labeled_traj = consensus_module.create_consensus_labels(
        example_trajectory, rpe_labels, judge_labels
    )

    # With trajectory_level=True, should be filtered
    assert labeled_traj.is_filtered
    assert labeled_traj.filter_reason == "trajectory_level_conflict"


def test_statistics(example_trajectory, example_config):
    """Test consensus statistics computation."""
    consensus_module = ConsensusModule(example_config["consensus"])

    # Create multiple labeled trajectories
    rpe_labels_1 = [
        RPELabel(step_id=0, mc_s_t=0.5, mc_s_t_a_t=0.9, rpe=1.8, label=LabelType.GOOD),
        RPELabel(step_id=1, mc_s_t=0.6, mc_s_t_a_t=0.95, rpe=1.58, label=LabelType.GOOD),
    ]
    judge_labels_1 = [
        JudgeLabel(step_id=0, label="GOOD", confidence=0.9),
        JudgeLabel(step_id=1, label="GOOD", confidence=0.95),
    ]

    labeled_traj_1 = consensus_module.create_consensus_labels(
        example_trajectory, rpe_labels_1, judge_labels_1
    )

    stats = consensus_module.compute_statistics([labeled_traj_1])

    assert stats.total_steps == 2
    assert stats.positive_consensus == 2
    assert stats.negative_consensus == 0
    assert stats.disagreements == 0
    assert stats.agreement_rate == 1.0
