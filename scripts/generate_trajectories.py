#!/usr/bin/env python3
"""Stage 1: Generate trajectories only (no Judge evaluation).

Generate N trajectories per question using Policy model.
Uses HYBRID retrieval (BM25 + BGE-M3) with RRF fusion + BGE Reranker.

Usage:
    python scripts/generate_trajectories.py \
        --data_path data/raw/questions/hotpotqa_train.jsonl \
        --output_path outputs/trajectories.jsonl \
        --num_samples 16 \
        --batch_size 50
"""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.generation.adaptive_generator import SimpleTrajectoryGenerator
from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever
from prmrag.retrieval.hybrid_retriever import HybridRetriever
from prmrag.retrieval.bge_reranker import BGEReranker


def get_truncated_text(doc_text_list, max_chars=1200):
    """Character-based truncation with truncation marker."""
    if not doc_text_list:
        return ""
    joined_text = ' '.join(doc_text_list)
    if len(joined_text) > max_chars:
        return joined_text[:max_chars] + " [TRUNCATED]"
    return joined_text


def load_kilt_corpus(corpus_file: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus with smart truncation."""
    print(f"Loading KILT corpus from {corpus_file}...")

    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)

            if isinstance(doc['text'], list):
                text = get_truncated_text(doc['text'], max_chars=1200)
            else:
                text = doc['text'][:1200] + (" [TRUNCATED]" if len(doc['text']) > 1200 else "")

            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })

            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1:,} documents...")

    print(f"✓ Loaded {len(corpus):,} documents")
    return corpus


def load_questions(data_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load questions from JSON/JSONL file."""
    questions = []

    if data_path.endswith('.jsonl'):
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
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


def save_trajectories(trajectories, output_path: str):
    """Save trajectories to JSONL file."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        for traj in trajectories:
            record = {
                'trajectory_id': traj.trajectory_id,
                'question': traj.question,
                'gold_answer': traj.gold_answer,
                'final_answer': traj.final_answer,
                'is_correct': traj.is_correct,
                'supporting_facts': traj.supporting_facts,
                'metadata': traj.metadata,
                'steps': [
                    {
                        'step_id': step.step_id,
                        'step_type': step.step_type.value,
                        'text': step.text,
                        'content': step.content,
                        'used_passages': step.used_passages,
                        'metadata': step.metadata,
                    }
                    for step in traj.steps
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"Saved {len(trajectories)} trajectories to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate Trajectories (Stage 1)")
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
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--max_steps', type=int, default=10,
                        help='Maximum steps per trajectory')

    # Retriever settings
    parser.add_argument('--corpus_limit', type=int, default=None,
                        help='Limit corpus size for testing (default: all)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of passages to retrieve per query')
    parser.add_argument('--rerank_top_n', type=int, default=20,
                        help='Rerank top-N candidates from fusion (default: 20)')

    # GPU settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.7,
                        help='GPU memory utilization for policy model')
    parser.add_argument('--max_model_len', type=int, default=8192,
                        help='Max model context length')

    args = parser.parse_args()

    print("=" * 70)
    print("Stage 1: Trajectory Generation")
    print("=" * 70)
    print(f"Policy model: {args.policy_model}")
    print(f"Samples per question: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Temperature: {args.temperature}")
    print(f"Retriever: HybridRetriever (BM25 + BGE-M3) + RRF + Reranker")
    print("=" * 70)

    # Load questions
    questions = load_questions(args.data_path, args.limit)
    total_trajectories = len(questions) * args.num_samples
    print(f"Will generate {total_trajectories} trajectories")

    # Data directory
    data_dir = Path(__file__).parent.parent / "data"

    # [1] Load KILT corpus
    print(f"\n[1/4] Loading KILT corpus...")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    corpus = load_kilt_corpus(corpus_file, limit=args.corpus_limit)

    # [2] Initialize retrievers
    print(f"\n[2/4] Initializing HYBRID retriever (BM25 + BGE-M3)...")

    # BGE retriever
    embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
    print(f"  [2.1] Initializing BGE-M3 retriever...")
    bge_retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(embedding_cache)
    )

    # BM25 retriever
    bm25_cache = data_dir / "indexes" / "kilt_wikipedia_bm25.pkl"
    print(f"  [2.2] Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever(
        corpus=corpus,
        index_cache_path=str(bm25_cache)
    )

    # BGE Reranker (always enabled)
    print(f"  [2.3] Initializing BGE Reranker...")
    reranker = BGEReranker(device="cuda")

    # Hybrid retriever (RRF fusion + Reranker)
    print(f"  [2.4] Combining with RRF fusion + Reranker...")
    retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        bge_retriever=bge_retriever,
        fusion_method="rrf",
        k_sparse=50,
        k_dense=50,
        reranker=reranker,
        rerank_top_n=args.rerank_top_n,
    )
    print(f"✓ Hybrid retriever initialized (RRF + Reranker)")

    # [3] Initialize policy model
    print(f"\n[3/4] Loading policy model...")
    policy_config = {
        'model_name': args.policy_model,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
    }
    policy_model = load_policy_model(policy_config)
    print("✓ Model loaded")

    # Initialize generator
    generator_config = {
        'max_steps': args.max_steps,
        'top_k_passages': args.top_k,
        'temperature': args.temperature,
    }
    generator = SimpleTrajectoryGenerator(policy_model, retriever, generator_config)

    # [4] Process in batches
    print(f"\n[4/4] Generating trajectories...")
    all_trajectories = []
    num_batches = (len(questions) + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(questions))
        batch_questions = questions[start_idx:end_idx]

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")
        print(f"  Questions: {start_idx + 1} ~ {end_idx}")
        print(f"  Trajectories to generate: {len(batch_questions) * args.num_samples}")

        # Generate trajectories
        trajectories = generator.generate_batch(
            batch_questions,
            num_samples=args.num_samples,
            show_progress=True,
        )

        all_trajectories.extend(trajectories)
        print(f"  Generated: {len(trajectories)} trajectories")
        print(f"  Total so far: {len(all_trajectories)}")

        # Save checkpoint every 10 batches
        if (batch_idx + 1) % 10 == 0:
            checkpoint_path = args.output_path.replace('.jsonl', f'_checkpoint_{batch_idx + 1}.jsonl')
            save_trajectories(all_trajectories, checkpoint_path)

    # Save final results
    save_trajectories(all_trajectories, args.output_path)

    # Print summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"Total questions: {len(questions)}")
    print(f"Total trajectories: {len(all_trajectories)}")

    correct_count = sum(1 for t in all_trajectories if t.is_correct)
    print(f"Correct answers: {correct_count}/{len(all_trajectories)} ({100*correct_count/len(all_trajectories):.1f}%)")

    avg_steps = sum(len(t.steps) for t in all_trajectories) / len(all_trajectories)
    print(f"Average steps per trajectory: {avg_steps:.1f}")

    print(f"\nOutput: {args.output_path}")


if __name__ == '__main__':
    main()
