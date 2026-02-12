#!/usr/bin/env python3
"""Critic-guided trajectory regeneration pipeline.

Flow:
1. Load judge labels + critic scores → merge by trajectory_id
2. Classify questions: Case A (has correct) vs Case B (all wrong)
3. Case A → best correct trajectory saved directly (rejection sampling)
4. Case B → truncate at first BAD step → add critic feedback → regenerate
5. Save all results to JSONL

Usage:
    python scripts/regenerate_trajectories.py \
        --trajectories outputs/trajectories_merged_1000q.jsonl \
        --critic_scores outputs/critic_scores_1000q.jsonl \
        --output_path outputs/regenerated_trajectories.jsonl \
        --limit 10
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.regeneration.classifier import QuestionClassifier
from prmrag.regeneration.truncator import TrajectoryTruncator
from prmrag.regeneration.regenerator import CriticGuidedRegenerator


def load_kilt_corpus(corpus_file: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus."""
    print(f"  Loading KILT corpus from {corpus_file}...")

    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)

            if isinstance(doc['text'], list):
                text = ' '.join(doc['text'])
                if len(text) > 1200:
                    text = text[:1200]
            else:
                text = doc['text'][:1200]

            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })

            if (i + 1) % 100000 == 0:
                print(f"    Loaded {i+1:,} documents...")

    print(f"  Loaded {len(corpus):,} documents")
    return corpus


def main():
    parser = argparse.ArgumentParser(
        description="Critic-Guided Trajectory Regeneration Pipeline"
    )
    parser.add_argument('--trajectories', type=str, required=True,
                        help='Path to original trajectories JSONL')
    parser.add_argument('--critic_scores', type=str, required=True,
                        help='Path to critic_scores JSONL')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to output JSONL')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of Case B questions to regenerate')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output (skip already processed)')

    # Model settings
    parser.add_argument('--policy_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Policy model name')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature for regeneration')
    parser.add_argument('--max_steps', type=int, default=10,
                        help='Maximum steps per trajectory')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of passages to retrieve per search')

    # GPU settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.88,
                        help='GPU memory utilization for policy model')
    parser.add_argument('--max_model_len', type=int, default=16384,
                        help='Max model context length')

    args = parser.parse_args()

    print("=" * 70)
    print("Critic-Guided Trajectory Regeneration Pipeline")
    print("=" * 70)

    # =========================================================
    # Step 1: Load & Classify
    # =========================================================
    print("\n[1/5] Loading and merging judge labels + critic scores...")
    scored_trajectories = QuestionClassifier.load_and_merge(
        args.trajectories, args.critic_scores
    )

    print("\n[2/5] Classifying questions (Case A vs Case B)...")
    classified, stats = QuestionClassifier.classify_questions(scored_trajectories)

    print(f"\n  Classification Results:")
    print(f"  Total questions: {stats['total_questions']}")
    print(f"  Case A (has correct): {stats['case_a']} ({stats['case_a_pct']:.1f}%)")
    print(f"  Case B (all wrong):   {stats['case_b']} ({stats['case_b_pct']:.1f}%)")
    print(f"  Total trajectories:   {stats['total_trajectories']}")

    case_a_questions = [q for q in classified if q.case == "A"]
    case_b_questions = [q for q in classified if q.case == "B"]

    # =========================================================
    # Step 2: Resume handling
    # =========================================================
    processed_ids = set()
    existing_records = []
    if args.resume and os.path.exists(args.output_path):
        print(f"\n  Resume mode: loading existing results from {args.output_path}")
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing_records.append(record)
                    processed_ids.add(record.get('question_id', ''))
        print(f"  Found {len(processed_ids)} questions already processed")

    # =========================================================
    # Step 3: Save Case A results (rejection sampling)
    # =========================================================
    print(f"\n[3/5] Saving Case A results (best correct trajectory)...")
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_mode = 'a' if args.resume and existing_records else 'w'
    case_a_saved = 0

    with open(args.output_path, file_mode, encoding='utf-8') as f:
        for cq in case_a_questions:
            if cq.question_id in processed_ids:
                continue

            best = cq.best_trajectory
            record = {
                'trajectory_id': best.trajectory_id,
                'question_id': cq.question_id,
                'question': cq.question,
                'gold_answer': cq.gold_answer,
                'predicted_answer': best.predicted_answer,
                'is_correct': best.is_correct,
                'case': 'A',
                'steps': best.steps,
                'metadata': {
                    'critic_min': best.critic_min,
                    'num_correct': cq.num_correct,
                    'num_total': cq.num_total,
                    'selection': 'best_correct_by_critic_min',
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            case_a_saved += 1

        f.flush()

    print(f"  Saved {case_a_saved} Case A trajectories")

    # =========================================================
    # Step 4: Initialize retriever + model for Case B
    # =========================================================
    case_b_to_process = [
        cq for cq in case_b_questions if cq.question_id not in processed_ids
    ]

    if args.limit is not None:
        case_b_to_process = case_b_to_process[:args.limit]

    if not case_b_to_process:
        print("\n[4/5] No Case B questions to regenerate. Done!")
        _print_summary(args.output_path, case_a_saved, 0, 0)
        return

    print(f"\n[4/5] Initializing retriever and policy model for {len(case_b_to_process)} Case B questions...")

    # Load KILT corpus + BGE-M3 retriever (same as original trajectory generation)
    data_dir = Path(__file__).parent.parent / "data"
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    embedding_cache = str(data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy")

    corpus = load_kilt_corpus(str(corpus_file))

    from prmrag.retrieval.bge_retriever import BGERetriever
    print("  Initializing BGE-M3 retriever...")
    retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=embedding_cache,
        device="cpu",  # CPU for BGE to avoid CUDA conflict with vLLM
    )

    # Load policy model
    from prmrag.models import load_policy_model
    print("  Loading policy model...")
    policy_config = {
        'model_name': args.policy_model,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
    }
    policy_model = load_policy_model(policy_config)

    # =========================================================
    # Step 5: Truncate + Regenerate Case B
    # =========================================================
    print(f"\n[5/5] Truncating and regenerating {len(case_b_to_process)} Case B trajectories...")

    truncator = TrajectoryTruncator()
    regenerator = CriticGuidedRegenerator(
        policy_model=policy_model,
        retriever=retriever,
        max_steps=args.max_steps,
        top_k_passages=args.top_k,
        temperature=args.temperature,
    )

    # Truncate all Case B trajectories
    truncation_results = []
    for cq in case_b_to_process:
        tr = truncator.truncate(cq.best_trajectory)
        truncation_results.append(tr)

    # Print truncation summary
    avg_good = sum(len(tr.good_steps) for tr in truncation_results) / len(truncation_results)
    print(f"  Truncation summary:")
    print(f"    Avg good steps preserved: {avg_good:.1f}")
    print(f"    Cases with 0 good steps: {sum(1 for tr in truncation_results if len(tr.good_steps) == 0)}")

    # Regenerate with incremental saving
    case_b_correct = 0
    case_b_total = 0

    with open(args.output_path, 'a', encoding='utf-8') as f:
        for i, tr in enumerate(truncation_results):
            print(f"  [{i+1}/{len(truncation_results)}] {tr.question_id}: "
                  f"truncated at step {tr.bad_step_id}, "
                  f"{len(tr.good_steps)} good steps")

            result = regenerator.regenerate_single(tr)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            case_b_total += 1
            if result['is_correct']:
                case_b_correct += 1

            if (i + 1) % 10 == 0:
                print(f"    Progress: {i+1}/{len(truncation_results)}, "
                      f"correct: {case_b_correct}/{case_b_total} "
                      f"({case_b_correct/case_b_total*100:.1f}%)")

    _print_summary(args.output_path, case_a_saved, case_b_total, case_b_correct)


def _print_summary(output_path, case_a_saved, case_b_total, case_b_correct):
    """Print final summary."""
    print("\n" + "=" * 70)
    print("Regeneration Complete!")
    print("=" * 70)
    print(f"  Case A saved: {case_a_saved} (all correct by definition)")
    print(f"  Case B regenerated: {case_b_total}")
    if case_b_total > 0:
        print(f"  Case B correct: {case_b_correct}/{case_b_total} "
              f"({case_b_correct/case_b_total*100:.1f}%)")
    total = case_a_saved + case_b_total
    total_correct = case_a_saved + case_b_correct
    if total > 0:
        print(f"  Overall: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
    print(f"\nOutput: {output_path}")


if __name__ == '__main__':
    main()
