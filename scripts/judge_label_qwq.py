#!/usr/bin/env python3
"""Judge labeling using QwQ-32B model.

Optimizations:
1. Trajectory 단위 배치: 한 번의 LLM 호출로 모든 step 평가
2. Incremental save: JSONL로 저장하여 중간에 끊겨도 복구 가능
3. Resume 지원: 이미 처리된 question_id 스킵
"""

import json
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prmrag.labeling.judge_labeler import JudgeLabeler
from prmrag.data.schemas import Trajectory, TrajectoryStep


def load_trajectories(filepath: Path):
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # Convert to Trajectory schema
            steps = []
            for step_data in data['steps']:
                # Determine action_type
                if step_data.get('num_passages', 0) > 0:
                    action_type = 'retrieve'
                elif step_data.get('action') == 'Finish':
                    action_type = 'answer'
                else:
                    action_type = 'reason'

                # Build full action with thought
                thought = step_data.get('thought', '')
                action_name = step_data.get('action', 'Unknown')

                # For Finish action, include the predicted answer
                if action_name == 'Finish':
                    predicted_ans = data.get('predicted_answer', 'Unknown')
                    action_with_answer = f"{action_name}[answer=\"{predicted_ans}\"]"
                else:
                    action_with_answer = action_name

                if thought:
                    full_action = f"Thought: {thought}\nAction: {action_with_answer}"
                else:
                    full_action = f"Action: {action_with_answer}"

                step = TrajectoryStep(
                    step_id=step_data['step_num'] - 1,
                    action_type=action_type,
                    action=full_action,
                    observation=step_data.get('observation', ''),
                    passages=[step_data.get('observation', '')] if step_data.get('observation') else [],
                )
                steps.append(step)

            trajectory = Trajectory(
                trajectory_id=data['question_id'],
                question=data['question'],
                gold_answer=data['gold_answer'],
                final_answer=data.get('predicted_answer', ''),
                steps=steps,
                supporting_facts=[],
            )
            trajectories.append((trajectory, data))

    return trajectories


def load_processed_ids(output_file: Path) -> set:
    """Load already processed question IDs from output file."""
    processed = set()
    if output_file.exists():
        with open(output_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        processed.add(data.get('question_id'))
                    except:
                        pass
    return processed


def main():
    parser = argparse.ArgumentParser(description="Judge labeling with QwQ-32B")
    parser.add_argument("--input", type=str,
                        default="./outputs/hybrid_1000q_medium_fulltext_merged/results_merged.jsonl")
    parser.add_argument("--output", type=str,
                        default="./outputs/judge_labels_qwq_new_prompt.jsonl")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of trajectories (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)

    print("=" * 70)
    print("JUDGE LABELING with QwQ-32B (Optimized)")
    print("=" * 70)
    print()
    print("Optimizations:")
    print("  ✓ Trajectory 단위 배치: 한 번의 LLM 호출로 모든 step 평가")
    print("  ✓ Incremental save: JSONL로 저장 (중간 중단 시 복구 가능)")
    print("  ✓ Resume 지원: --resume로 이어서 처리")
    print()

    # Load trajectories
    print(f"Loading trajectories from: {input_file}")
    trajectory_pairs = load_trajectories(input_file)
    print(f"✓ Loaded {len(trajectory_pairs)} trajectories")

    # Apply limit if specified
    if args.limit:
        trajectory_pairs = trajectory_pairs[:args.limit]
        print(f"✓ Limited to first {len(trajectory_pairs)} trajectories")

    # Load processed IDs for resume
    processed_ids = set()
    if args.resume:
        processed_ids = load_processed_ids(output_file)
        print(f"✓ Resume mode: {len(processed_ids)} already processed")

    # Filter out already processed
    remaining = [(t, d) for t, d in trajectory_pairs if t.trajectory_id not in processed_ids]
    print(f"✓ Remaining to process: {len(remaining)} trajectories")
    print()

    # Initialize Judge labeler
    config = {
        'model_name': 'Qwen/QwQ-32B',
        'temperature': 0.3,
        'max_tokens': 3072,
        'max_model_len': 32768,
        'gpu_memory_utilization': 0.95,
        'tensor_parallel_size': 1,
        'prompt_style': 'versaprm',
        'use_gold_answer': True,
        'use_supporting_facts': False,
    }

    print("Initializing QwQ-32B (one-time)...")
    judge_labeler = JudgeLabeler(config)
    print("✓ Model loaded\n")

    # Open output file for incremental writing
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_fp = open(output_file, 'a' if args.resume else 'w', encoding='utf-8')

    # Statistics
    total_steps = 0
    consensus_count = 0
    error_count = 0

    # Process with progress bar
    for trajectory, original_data in tqdm(remaining, desc="Judge labeling"):
        try:
            # Label all steps in one LLM call (batch mode)
            judge_labels = judge_labeler.label_trajectory(trajectory)

        except Exception as e:
            tqdm.write(f"  ERROR {trajectory.trajectory_id}: {e}")
            error_count += 1
            from prmrag.data.schemas import JudgeLabel
            judge_labels = [
                JudgeLabel(
                    step_id=step_idx,
                    label='UNKNOWN',
                    reasoning=f'Error: {str(e)}',
                    confidence=0.0,
                )
                for step_idx in range(len(trajectory.steps))
            ]

        # Prepare output
        output_data = {
            'question_id': trajectory.trajectory_id,
            'question': trajectory.question,
            'gold_answer': trajectory.gold_answer,
            'predicted_answer': trajectory.final_answer,
            'steps': []
        }

        for step_idx, (step, judge_label) in enumerate(zip(original_data['steps'], judge_labels)):
            step_output = {
                'step_num': step['step_num'],
                'rpe_label': step.get('label', 'unknown'),
                'judge_label': judge_label.label,
                'judge_reasoning': judge_label.reasoning,
                'judge_confidence': judge_label.confidence,
                'mc_before': step.get('mc_before'),
                'mc_after': step.get('mc_after'),
                'rpe': step.get('rpe'),
                'action': step.get('action'),
                'num_passages': step.get('num_passages', 0),
            }
            output_data['steps'].append(step_output)

            # Update statistics
            total_steps += 1
            if step_output['rpe_label'] == judge_label.label.lower():
                consensus_count += 1

        # Write immediately (incremental save)
        out_fp.write(json.dumps(output_data, ensure_ascii=False) + '\n')
        out_fp.flush()

    out_fp.close()

    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total trajectories processed: {len(remaining)}")
    print(f"Total steps evaluated: {total_steps}")
    print(f"Errors: {error_count}")
    if total_steps > 0:
        print(f"Consensus (RPE=Judge): {consensus_count}/{total_steps} ({consensus_count/total_steps*100:.1f}%)")
        print(f"Disagreement: {total_steps - consensus_count}/{total_steps} ({(total_steps-consensus_count)/total_steps*100:.1f}%)")
    print()
    print(f"✓ Output saved to: {output_file}")
    print()


if __name__ == '__main__':
    main()
