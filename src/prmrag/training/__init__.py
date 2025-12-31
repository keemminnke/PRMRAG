"""Training module for PRMRAG.

This module provides:
- Checkpoint-based training with scaling law experiments
- Data sampling by difficulty (Hard/Med/Easy)
- Incremental evaluation at each checkpoint
- Critic model training for step-level evaluation
"""

from .data_sampler import (
    DifficultySampler,
    DifficultyLevel,
    SamplingConfig,
    TrainingSample,
    create_scaling_datasets,
)
from .checkpoint_trainer import (
    CheckpointTrainer,
    TrainingCheckpoint,
    ScalingExperiment,
    CheckpointStatus,
)
from .scaling_analysis import (
    ScalingAnalyzer,
    ScalingCurve,
    ScalingPoint,
    plot_scaling_curve,
    plot_difficulty_breakdown,
)
from .critic_trainer import (
    CriticTrainer,
    CriticTrainingConfig,
    CriticDataFormatter,
    create_critic_trainer,
)

__all__ = [
    # Data sampling
    "DifficultySampler",
    "DifficultyLevel",
    "SamplingConfig",
    "TrainingSample",
    "create_scaling_datasets",
    # Training
    "CheckpointTrainer",
    "TrainingCheckpoint",
    "ScalingExperiment",
    "CheckpointStatus",
    # Critic training
    "CriticTrainer",
    "CriticTrainingConfig",
    "CriticDataFormatter",
    "create_critic_trainer",
    # Analysis
    "ScalingAnalyzer",
    "ScalingCurve",
    "ScalingPoint",
    "plot_scaling_curve",
    "plot_difficulty_breakdown",
]
