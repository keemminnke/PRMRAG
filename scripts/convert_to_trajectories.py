#!/usr/bin/env python3
"""Convert batch test JSONL results to Trajectory schema format.

This script:
1. Loads JSONL from batch_test_adaptive.py
2. Converts to Trajectory objects (schemas.py)
3. Extracts RPE labels separately
4. Saves in structured format for downstream use (Judge labeling, PRM training, etc.)
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.data.schemas import (
    Trajectory,
    TrajectoryStep,
    ActionType,
    RPELabel,
    LabelType,
)


def parse_step_content(step_data: Dict[str, Any]) -> tuple[str, str]:
    """Parse step content into action and observation.

    For ReAct format:
    - Action: Thought + Action part
    - Observation: Observation part (or result)

    Args:
        step_data: Step data from JSONL

    Returns:
        Tuple of (action, observation)
    """
    content = step_data.get('content', '')
    step_type = step_data.get('type', 'cot')

    if step_type == 'rag':
        # RAG step: Split at "Observation:"
        if 'Observation:' in content:
            parts = content.split('Observation:', 1)
            action = parts[0].strip()
            observation = parts[1].strip() if len(parts) > 1 else ""
        else:
            # Fallback: entire content as action
            action = content
            observation = ""
    else:
        # CoT step: Entire content is reasoning (action)
        # No external observation
        action = content
        observation = ""

    return action, observation


def determine_action_type(step_data: Dict[str, Any]) -> ActionType:
    """Determine ActionType from step data.

    Args:
        step_data: Step data from JSONL

    Returns:
        ActionType enum value
    """
    step_type = step_data.get('type', 'cot')
    is_rag = step_data.get('is_rag', False)

    # Check if this is the final answer step
    content = step_data.get('content', '')
    if 'Final Answer:' in content or 'Therefore, the answer is' in content:
        return ActionType.ANSWER

    # RAG step = retrieve
    if is_rag or step_type == 'rag':
        return ActionType.RETRIEVE

    # Otherwise = reasoning
    return ActionType.REASON


def convert_jsonl_to_trajectory(data: Dict[str, Any]) -> Trajectory:
    """Convert single JSONL entry to Trajectory object.

    Args:
        data: JSONL data for one question

    Returns:
        Trajectory object
    """
    trajectory_id = data['question_id']
    question = data['question']
    gold_answer = data.get('gold_answer')
    final_answer = data.get('predicted_answer', '')
    is_correct = data.get('is_correct', False)

    # Convert steps
    steps = []
    for step_data in data.get('steps', []):
        step_num = step_data['step_num']
        action_type = determine_action_type(step_data)
        action, observation = parse_step_content(step_data)

        # Get passages if available
        passages = []
        if step_data.get('is_rag') and 'passage_titles' in step_data:
            passages = step_data['passage_titles']

        # Store original metadata
        metadata = {
            'mc_before': step_data.get('mc_before', 0.0),
            'mc_after': step_data.get('mc_after', 0.0),
            'rpe': step_data.get('rpe', 1.0),
            'label': step_data.get('label', 'good'),
            'original_type': step_data.get('type', 'cot'),
            'is_rag': step_data.get('is_rag', False),
        }

        step = TrajectoryStep(
            step_id=step_num - 1,  # 0-indexed
            action_type=action_type,
            action=action,
            observation=observation,
            passages=passages,
            metadata=metadata,
        )
        steps.append(step)

    # Store original metadata
    traj_metadata = {
        'num_steps': data.get('num_steps', len(steps)),
        'num_cot_steps': data.get('num_cot_steps', 0),
        'num_rag_steps': data.get('num_rag_steps', 0),
        'has_rag': data.get('has_rag', False),
        'rag_interventions': data.get('rag_interventions', []),
        'format_compliance': data.get('format_compliance', {}),
    }

    trajectory = Trajectory(
        trajectory_id=trajectory_id,
        question=question,
        gold_answer=gold_answer,
        steps=steps,
        final_answer=final_answer,
        is_correct=is_correct,
        metadata=traj_metadata,
    )

    return trajectory


def extract_rpe_labels(trajectory: Trajectory) -> List[RPELabel]:
    """Extract RPE labels from trajectory metadata.

    Args:
        trajectory: Trajectory object with metadata

    Returns:
        List of RPELabel objects
    """
    labels = []

    for step in trajectory.steps:
        metadata = step.metadata

        # Get RPE values from metadata
        mc_s_t = metadata.get('mc_before', 0.0)
        mc_s_t_a_t = metadata.get('mc_after', 0.0)
        rpe = metadata.get('rpe', 1.0)
        label_str = metadata.get('label', 'good')

        # Convert string label to LabelType
        label = LabelType.GOOD if label_str == 'good' else LabelType.BAD

        rpe_label = RPELabel(
            step_id=step.step_id,
            mc_s_t=mc_s_t,
            mc_s_t_a_t=mc_s_t_a_t,
            rpe=rpe,
            label=label,
            confidence=1.0,
            metadata={
                'num_rollouts': 8,  # K=8 from batch_test_adaptive.py
                'threshold': 1.0,   # Default threshold
            },
        )
        labels.append(rpe_label)

    return labels


def convert_file(
    input_file: Path,
    output_dir: Path,
    save_labels: bool = True,
):
    """Convert entire JSONL file to Trajectory format.

    Args:
        input_file: Input JSONL file
        output_dir: Output directory
        save_labels: Whether to save RPE labels separately
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output files
    trajectories_file = output_dir / "trajectories.jsonl"
    labels_file = output_dir / "rpe_labels.jsonl"

    trajectories = []
    all_labels = []

    print(f"Converting {input_file}...")

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)

                # Convert to Trajectory
                trajectory = convert_jsonl_to_trajectory(data)
                trajectories.append(trajectory)

                # Extract RPE labels
                if save_labels:
                    labels = extract_rpe_labels(trajectory)
                    all_labels.append({
                        'trajectory_id': trajectory.trajectory_id,
                        'labels': [label.to_dict() for label in labels],
                    })

                if line_num % 10 == 0:
                    print(f"  Processed {line_num} trajectories...")

            except Exception as e:
                print(f"  Error on line {line_num}: {e}")
                continue

    # Save trajectories
    print(f"\nSaving {len(trajectories)} trajectories to {trajectories_file}...")
    with open(trajectories_file, 'w') as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict()) + '\n')

    # Save labels
    if save_labels and all_labels:
        print(f"Saving RPE labels to {labels_file}...")
        with open(labels_file, 'w') as f:
            for label_data in all_labels:
                f.write(json.dumps(label_data) + '\n')

    # Print summary
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Total steps: {sum(len(t.steps) for t in trajectories)}")
    print(f"Correct trajectories: {sum(1 for t in trajectories if t.is_correct)}")

    if trajectories:
        avg_steps = sum(len(t.steps) for t in trajectories) / len(trajectories)
        print(f"Average steps per trajectory: {avg_steps:.1f}")

        # Label distribution
        good_labels = 0
        bad_labels = 0
        for traj in trajectories:
            for step in traj.steps:
                label = step.metadata.get('label', 'good')
                if label == 'good':
                    good_labels += 1
                else:
                    bad_labels += 1

        total_labels = good_labels + bad_labels
        if total_labels > 0:
            print(f"\nLabel distribution:")
            print(f"  GOOD: {good_labels} ({100*good_labels/total_labels:.1f}%)")
            print(f"  BAD:  {bad_labels} ({100*bad_labels/total_labels:.1f}%)")

    print(f"\nOutput files:")
    print(f"  - {trajectories_file}")
    if save_labels:
        print(f"  - {labels_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert batch test JSONL to Trajectory format"
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input JSONL file from batch_test_adaptive.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/trajectories"),
        help="Output directory (default: data/processed/trajectories)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't save RPE labels separately",
    )

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: Input file not found: {args.input_file}")
        return 1

    convert_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        save_labels=not args.no_labels,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
