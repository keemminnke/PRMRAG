#!/usr/bin/env python3
"""No-RPE Pipeline: Generate trajectories and label with Judge model.

This script implements the no-RPE training data generation pipeline:
1. Policy model generates N trajectories per question (sampling)
2. Judge model evaluates each step (GOOD/BAD with reasoning)
3. Results saved for PRM training

Usage:
    python scripts/run_no_rpe_pipeline.py \
        --data_path data/hotpotqa/train.json \
        --output_path outputs/no_rpe_labeled.jsonl \
        --num_samples 16 \
        --batch_size 100
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prmrag.generation.adaptive_generator import SimpleTrajectoryGenerator, AdaptiveTrajectory
from src.prmrag.labeling.judge_labeler import JudgeLabeler
from src.prmrag.models import load_policy_model
from src.prmrag.retrieval import HybridRetriever
from src.prmrag.data.schemas import Trajectory, TrajectoryStep, ActionType


def load_questions(data_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load questions from JSON/JSONL file."""
    questions = []

    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'data' in data:
                questions = data['data']
            else:
                questions = [data]

    if limit:
        questions = questions[:limit]

    print(f"Loaded {len(questions)} questions from {data_path}")
    return questions


def convert_to_judge_trajectory(adaptive_traj: AdaptiveTrajectory) -> Trajectory:
    """Convert AdaptiveTrajectory to Trajectory format for Judge labeling."""
    steps = []
    for i, step in enumerate(adaptive_traj.steps):
        # Extract action and observation from content
        content = step.content

        # Split content into action part and observation part
        if "\n\nObservation:" in content:
            action_part, obs_part = content.split("\n\nObservation:", 1)
            observation = obs_part.strip()
        else:
            action_part = content
            observation = ""

        # Determine action type
        action_type = ActionType.REASON
        if step.step_type.value == 'rag':
            action_type = ActionType.RETRIEVE
        elif step.step_type.value == 'answer':
            action_type = ActionType.ANSWER

        steps.append(TrajectoryStep(
            step_id=i,
            action_type=action_type,
            action=action_part,
            observation=observation,
            passages=[p.get('text', '') for p in step.used_passages] if step.used_passages else [],
        ))

    return Trajectory(
        trajectory_id=adaptive_traj.trajectory_id,
        question=adaptive_traj.question,
        steps=steps,
        final_answer=adaptive_traj.final_answer,
        gold_answer=adaptive_traj.gold_answer,
        supporting_facts=adaptive_traj.supporting_facts,
    )


def save_results(
    trajectories: List[AdaptiveTrajectory],
    labels: List[List],
    output_path: str,
):
    """Save trajectories with labels to JSONL file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for traj, traj_labels in zip(trajectories, labels):
            record = {
                'trajectory_id': traj.trajectory_id,
                'question': traj.question,
                'gold_answer': traj.gold_answer,
                'final_answer': traj.final_answer,
                'is_correct': traj.is_correct,
                'metadata': traj.metadata,
                'steps': [],
            }

            for step, label in zip(traj.steps, traj_labels):
                step_record = {
                    'step_id': step.step_id,
                    'step_type': step.step_type.value,
                    'content': step.content,
                    'action': step.metadata.get('action'),
                    'used_passages': [
                        {'title': p.get('title', ''), 'text': p.get('text', '')[:200]}
                        for p in step.used_passages
                    ] if step.used_passages else [],
                    'judge_label': label.label,
                    'judge_reasoning': label.reasoning,
                }
                record['steps'].append(step_record)

            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(trajectories)} labeled trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="No-RPE Pipeline")
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to questions JSON/JSONL file')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSONL file')
    parser.add_argument('--num_samples', type=int, default=16,
                        help='Number of trajectories per question (default: 16)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Number of questions per batch (default: 50)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of questions (for testing)')

    # Model settings
    parser.add_argument('--policy_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Policy model name')
    parser.add_argument('--judge_model', type=str, default='Qwen/QwQ-32B',
                        help='Judge model name')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature for policy model')

    # Retriever settings
    parser.add_argument('--corpus_path', type=str, default=None,
                        help='Path to retrieval corpus')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of passages to retrieve')

    # GPU settings
    parser.add_argument('--policy_gpu_util', type=float, default=0.5,
                        help='GPU memory utilization for policy model')
    parser.add_argument('--judge_gpu_util', type=float, default=0.4,
                        help='GPU memory utilization for judge model')

    args = parser.parse_args()

    print("=" * 60)
    print("No-RPE Pipeline: Trajectory Generation + Judge Labeling")
    print("=" * 60)
    print(f"Policy model: {args.policy_model}")
    print(f"Judge model: {args.judge_model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Load questions
    questions = load_questions(args.data_path, args.limit)

    # Initialize policy model
    print("\n[1/4] Loading policy model...")
    policy_config = {
        'model_name': args.policy_model,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.policy_gpu_util,
        'max_model_len': 8192,
    }
    policy_model = load_policy_model(policy_config)

    # Initialize retriever (if corpus provided)
    print("\n[2/4] Loading retriever...")
    if args.corpus_path:
        retriever = HybridRetriever({
            'corpus_path': args.corpus_path,
            'k_sparse': args.top_k,
            'k_dense': args.top_k,
        })
    else:
        # Use Wikipedia API retriever as fallback
        from src.prmrag.retrieval import WikipediaRetriever
        retriever = WikipediaRetriever({'top_k': args.top_k})

    # Initialize trajectory generator
    generator_config = {
        'max_steps': 10,
        'top_k_passages': args.top_k,
        'temperature': args.temperature,
    }
    generator = SimpleTrajectoryGenerator(policy_model, retriever, generator_config)

    # Initialize judge model
    print("\n[3/4] Loading judge model...")
    judge_config = {
        'model_name': args.judge_model,
        'temperature': 0.3,
        'max_tokens': 2048,
        'gpu_memory_utilization': args.judge_gpu_util,
        'max_model_len': 32768,
    }
    judge = JudgeLabeler(judge_config)

    # Process in batches
    print("\n[4/4] Generating trajectories and labeling...")
    all_trajectories = []
    all_labels = []

    num_batches = (len(questions) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ({len(batch_questions)} questions) ---")

        # Generate trajectories (batch_questions * num_samples)
        print(f"  Generating {len(batch_questions) * args.num_samples} trajectories...")
        trajectories = generator.generate_batch(
            batch_questions,
            num_samples=args.num_samples,
            show_progress=True,
        )
        print(f"  Generated {len(trajectories)} trajectories")

        # Convert to Judge format
        judge_trajectories = [convert_to_judge_trajectory(t) for t in trajectories]

        # Label with Judge (batch)
        print(f"  Labeling with Judge model...")
        labels = judge.label_batch(judge_trajectories, show_progress=True)

        all_trajectories.extend(trajectories)
        all_labels.extend(labels)

        # Save intermediate results
        if (batch_idx + 1) % 5 == 0:
            intermediate_path = args.output_path.replace('.jsonl', f'_checkpoint_{batch_idx + 1}.jsonl')
            save_results(all_trajectories, all_labels, intermediate_path)

    # Save final results
    save_results(all_trajectories, all_labels, args.output_path)

    # Print summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"Total questions: {len(questions)}")
    print(f"Total trajectories: {len(all_trajectories)}")
    print(f"Total steps labeled: {sum(len(l) for l in all_labels)}")

    # Calculate statistics
    correct_count = sum(1 for t in all_trajectories if t.is_correct)
    good_step_count = sum(1 for labels in all_labels for l in labels if l.label == 'GOOD')
    bad_step_count = sum(1 for labels in all_labels for l in labels if l.label == 'BAD')

    print(f"Correct answers: {correct_count}/{len(all_trajectories)} ({100*correct_count/len(all_trajectories):.1f}%)")
    print(f"GOOD steps: {good_step_count}")
    print(f"BAD steps: {bad_step_count}")
    print(f"Output: {args.output_path}")


if __name__ == '__main__':
    main()
