#!/usr/bin/env python3
"""Simple trajectory generation script (without RPE/MC rollouts).

This script uses SimpleTrajectoryGenerator which is much faster than
AdaptiveTrajectoryGenerator because it doesn't require MC rollouts.

Usage:
    python scripts/generate_simple.py \
        --config configs/simple_generation.yaml \
        --output_dir outputs/simple_trajectories \
        --limit 100
"""

import argparse
import json
import os
import yaml
from datetime import datetime
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prmrag.generation import SimpleTrajectoryGenerator, AdaptiveTrajectory
from src.prmrag.models import load_policy_model
from src.prmrag.retrieval import HybridRetriever


def load_questions(file_path: str, limit: int = None, level: str = None):
    """Load questions from HotpotQA file."""
    questions = []

    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Filter by level if specified
            if level and data.get('level', '').lower() != level.lower():
                continue

            questions.append({
                '_id': data.get('_id', ''),
                'question': data.get('question', ''),
                'gold_answer': data.get('answer', ''),
                'supporting_facts': data.get('supporting_facts', []),
                'level': data.get('level', ''),
            })

            if limit and len(questions) >= limit:
                break

    return questions


def main():
    parser = argparse.ArgumentParser(description='Generate trajectories without RPE')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions')
    parser.add_argument('--question_level', type=str, default=None, help='Filter by level (easy/medium/hard)')
    parser.add_argument('--questions_file', type=str, default=None, help='Questions file (overrides config)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print("Loading policy model...")
    model_config = config.get('model', {})
    policy_model = load_policy_model(model_config)

    # Load retriever
    print("Loading retriever...")
    retriever_config = config.get('retriever', {})
    retriever = HybridRetriever(retriever_config)

    # Create generator
    gen_config = config.get('generation', {})
    generator = SimpleTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
        config=gen_config,
    )

    # Load questions
    questions_file = args.questions_file or config.get('data', {}).get('questions_file')
    print(f"Loading questions from {questions_file}...")
    questions = load_questions(questions_file, limit=args.limit, level=args.question_level)
    print(f"Loaded {len(questions)} questions")

    # Generate trajectories
    print("Generating trajectories...")
    trajectories = generator.generate_batch(questions, show_progress=True)

    # Save results
    output_file = os.path.join(args.output_dir, 'trajectories.jsonl')
    with open(output_file, 'w') as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + '\n')

    # Calculate stats
    correct = sum(1 for t in trajectories if t.is_correct)
    total = len(trajectories)

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': args.config,
        'questions_file': questions_file,
        'num_questions': len(questions),
        'num_trajectories': total,
        'num_correct': correct,
        'accuracy': correct / total if total > 0 else 0,
        'generator': 'SimpleTrajectoryGenerator',
    }

    summary_file = os.path.join(args.output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {output_file}")
    print(f"Summary: {correct}/{total} correct ({summary['accuracy']*100:.1f}%)")


if __name__ == '__main__':
    main()
