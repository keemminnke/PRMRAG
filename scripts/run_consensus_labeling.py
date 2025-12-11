#!/usr/bin/env python3
"""Run consensus filtering using RPE + Judge labelers.

This script uses the existing labeling modules:
- RPELabeler: MC-based evaluation
- JudgeLabeler: LLM judge evaluation (VersaPRM style)
- ConsensusModule: Combine both and filter
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from prmrag.labeling import RPELabeler, JudgeLabeler, ConsensusModule, print_consensus_report
from prmrag.data.schemas import Trajectory, Step


def load_trajectories(jsonl_path: Path) -> List[Trajectory]:
    """Load trajectories from JSONL file.

    Args:
        jsonl_path: Path to JSONL file with trajectories

    Returns:
        List of Trajectory objects
    """
    trajectories = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Convert to Trajectory object
            steps = []
            for step_data in data['steps']:
                step = Step(
                    action=step_data.get('action', 'Search'),
                    observation=step_data.get('observation', ''),
                    passages=step_data.get('passages', []),
                )
                steps.append(step)

            trajectory = Trajectory(
                question_id=data['question_id'],
                question=data['question'],
                gold_answer=data.get('gold_answer'),
                steps=steps,
                supporting_facts=data.get('supporting_facts', []),
            )

            trajectories.append(trajectory)

    return trajectories


def save_labeled_trajectories(
    labeled_trajectories: List,
    output_path: Path,
    save_filtered: bool = False,
):
    """Save labeled trajectories to JSONL.

    Args:
        labeled_trajectories: List of LabeledTrajectory objects
        output_path: Output file path
        save_filtered: Whether to save filtered trajectories
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for labeled_traj in labeled_trajectories:
            # Skip filtered trajectories unless requested
            if labeled_traj.is_filtered and not save_filtered:
                continue

            # Convert to dict for saving
            traj_dict = {
                'question_id': labeled_traj.trajectory.question_id,
                'question': labeled_traj.trajectory.question,
                'gold_answer': labeled_traj.trajectory.gold_answer,
                'is_filtered': labeled_traj.is_filtered,
                'filter_reason': labeled_traj.filter_reason,
                'steps': [],
            }

            # Add steps with consensus labels
            for i, (step, consensus) in enumerate(
                zip(labeled_traj.trajectory.steps, labeled_traj.consensus_labels)
            ):
                step_dict = {
                    'step_num': i + 1,
                    'action': step.action,
                    'observation': step.observation,
                    'label': consensus.consensus_label,  # 0, 1, or None
                    'rpe_label': consensus.rpe_label.value if consensus.rpe_label else None,
                    'judge_label': consensus.judge_label,
                    'is_consensus': consensus.is_consensus,
                    'confidence': consensus.confidence,
                }

                if not consensus.is_consensus:
                    step_dict['filtered_reason'] = consensus.filtered_reason

                traj_dict['steps'].append(step_dict)

            f.write(json.dumps(traj_dict) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Run consensus filtering with RPE + Judge labelers'
    )

    # Input/Output
    parser.add_argument(
        'input_file',
        type=Path,
        help='Input JSONL file with trajectories',
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSONL file for labeled data',
    )
    parser.add_argument(
        '--save-filtered',
        action='store_true',
        help='Also save filtered trajectories (with filter reasons)',
    )

    # RPE settings
    parser.add_argument(
        '--rpe-model',
        default='Qwen/Qwen2.5-7B-Instruct',
        help='Model for RPE labeling',
    )
    parser.add_argument(
        '--rpe-rollouts',
        type=int,
        default=5,
        help='Number of MC rollouts for RPE',
    )
    parser.add_argument(
        '--rpe-threshold',
        type=float,
        default=0.5,
        help='RPE threshold (>= threshold: GOOD)',
    )

    # Judge settings
    parser.add_argument(
        '--judge-model',
        default='Qwen/Qwen2.5-72B-Instruct',
        help='Model for Judge labeling (use larger model)',
    )
    parser.add_argument(
        '--judge-style',
        choices=['versaprm', 'default'],
        default='versaprm',
        help='Judge prompt style',
    )

    # Consensus settings
    parser.add_argument(
        '--strategy',
        choices=['strict', 'lenient', 'balanced'],
        default='strict',
        help='Consensus strategy',
    )
    parser.add_argument(
        '--min-agreement',
        type=float,
        default=0.8,
        help='Minimum agreement rate to keep trajectory',
    )
    parser.add_argument(
        '--trajectory-level',
        action='store_true',
        help='Filter entire trajectory if any step conflicts',
    )

    # GPU settings
    parser.add_argument(
        '--gpu-memory',
        type=float,
        default=0.8,
        help='GPU memory utilization for vLLM',
    )
    parser.add_argument(
        '--tensor-parallel',
        type=int,
        default=1,
        help='Number of GPUs for tensor parallelism',
    )

    # Batch settings
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of trajectories to process',
    )

    args = parser.parse_args()

    # Check input
    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    print("=" * 60)
    print("CONSENSUS LABELING PIPELINE")
    print("=" * 60)
    print()

    # Load trajectories
    print(f"Loading trajectories from {args.input_file}...")
    trajectories = load_trajectories(args.input_file)

    if args.limit:
        trajectories = trajectories[:args.limit]

    print(f"✓ Loaded {len(trajectories)} trajectories")
    print()

    # Initialize RPE labeler
    print("Initializing RPE labeler...")
    rpe_config = {
        'model_name': args.rpe_model,
        'num_rollouts': args.rpe_rollouts,
        'threshold': args.rpe_threshold,
        'temperature': 0.8,
        'gpu_memory_utilization': args.gpu_memory,
        'tensor_parallel_size': args.tensor_parallel,
    }
    rpe_labeler = RPELabeler(rpe_config)
    print(f"✓ RPE labeler ready (model: {args.rpe_model})")
    print()

    # Initialize Judge labeler
    print("Initializing Judge labeler...")
    judge_config = {
        'model_name': args.judge_model,
        'prompt_style': args.judge_style,
        'temperature': 0.3,
        'gpu_memory_utilization': args.gpu_memory,
        'tensor_parallel_size': args.tensor_parallel,
    }
    judge_labeler = JudgeLabeler(judge_config)
    print(f"✓ Judge labeler ready (model: {args.judge_model})")
    print()

    # Initialize Consensus module
    consensus_config = {
        'strategy': args.strategy,
        'min_agreement': args.min_agreement,
        'trajectory_level': args.trajectory_level,
    }
    consensus_module = ConsensusModule(consensus_config)
    print(f"✓ Consensus module ready (strategy: {args.strategy})")
    print()

    # Run RPE labeling
    print("=" * 60)
    print("STEP 1: RPE LABELING")
    print("=" * 60)
    print()
    rpe_labels_batch = rpe_labeler.label_batch(trajectories, show_progress=True)
    print(f"✓ RPE labeling complete")
    print()

    # Run Judge labeling
    print("=" * 60)
    print("STEP 2: JUDGE LABELING")
    print("=" * 60)
    print()
    judge_labels_batch = judge_labeler.label_batch(trajectories, show_progress=True)
    print(f"✓ Judge labeling complete")
    print()

    # Run Consensus filtering
    print("=" * 60)
    print("STEP 3: CONSENSUS FILTERING")
    print("=" * 60)
    print()
    labeled_trajectories = consensus_module.process_batch(
        trajectories,
        rpe_labels_batch,
        judge_labels_batch,
    )
    print(f"✓ Consensus filtering complete")
    print()

    # Compute and print statistics
    stats = consensus_module.compute_statistics(labeled_trajectories)
    print_consensus_report(stats)

    # Save results
    print(f"Saving results to {args.output}...")
    save_labeled_trajectories(
        labeled_trajectories,
        args.output,
        save_filtered=args.save_filtered,
    )
    print(f"✓ Saved to {args.output}")
    print()

    # Print summary
    kept = sum(1 for lt in labeled_trajectories if not lt.is_filtered)
    filtered = len(labeled_trajectories) - kept

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total trajectories:     {len(labeled_trajectories)}")
    print(f"Kept (high quality):    {kept} ({kept/len(labeled_trajectories)*100:.1f}%)")
    print(f"Filtered (low quality): {filtered} ({filtered/len(labeled_trajectories)*100:.1f}%)")
    print()
    print(f"Agreement rate: {stats.agreement_rate:.1%}")
    print(f"Output: {args.output}")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
