#!/usr/bin/env python3
"""Generate single trajectory per question for testing.

Usage:
    python scripts/generate_trajectories_simple.py --num-questions 10
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.adaptive_generator import SimpleTrajectoryGenerator
from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever
from prmrag.retrieval.hybrid_retriever import HybridRetriever


def load_kilt_corpus(corpus_file: Path, limit: int = None):
    """Load KILT corpus."""
    print(f"Loading corpus from {corpus_file}...")
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line)
            if isinstance(doc['text'], list):
                text = ' '.join(doc['text'])[:1200]
            else:
                text = doc['text'][:1200]
            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })
            if (i + 1) % 500000 == 0:
                print(f"  Loaded {i+1:,} documents...")
    print(f"✓ Loaded {len(corpus):,} documents")
    return corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=10)
    parser.add_argument("--output", type=str, default="outputs/test_trajectories.jsonl")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    # Paths
    data_dir = Path("data")
    questions_file = data_dir / "raw" / "questions" / "hotpotqa_validation.jsonl"
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"

    # Load config
    config = load_config(Path("configs/adaptive_generation.yaml"))

    # Load questions
    print(f"\n[1] Loading {args.num_questions} questions...")
    questions = []
    with open(questions_file) as f:
        for i, line in enumerate(f):
            if i >= args.num_questions:
                break
            questions.append(json.loads(line))
    print(f"✓ Loaded {len(questions)} questions")

    # Load corpus
    print(f"\n[2] Loading corpus...")
    corpus = load_kilt_corpus(corpus_file)

    # Load policy model with lower GPU memory utilization
    print(f"\n[3] Loading policy model...")
    policy_config = config['policy_model'].copy()
    policy_config['gpu_memory_utilization'] = 0.9  # Use 90% of GPU
    policy_model = load_policy_model(policy_config)
    print("✓ Policy model loaded")

    # Initialize retrievers with cached indexes
    print(f"\n[4] Initializing retrievers...")

    # BGE retriever with cached embeddings
    embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
    print(f"  Loading BGE retriever (cache: {embedding_cache})...")
    bge = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(embedding_cache)
    )

    # BM25 retriever with cached index
    bm25_cache = data_dir / "indexes" / "kilt_wikipedia_bm25.pkl"
    print(f"  Loading BM25 retriever (cache: {bm25_cache})...")
    bm25 = BM25Retriever(
        corpus=corpus,
        index_cache_path=str(bm25_cache)
    )

    retriever = HybridRetriever(bm25_retriever=bm25, bge_retriever=bge)
    print("✓ Retrievers initialized")

    # Create generator
    gen_config = config.get('generation', config.get('adaptive', {}))
    generator = SimpleTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
        config=gen_config,
    )

    # Generate trajectories
    print(f"\n{'='*70}")
    print(f"GENERATING TRAJECTORIES")
    print(f"{'='*70}\n")

    results = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:60]}...")
        print(f"  Gold answer: {q.get('answer', 'N/A')}")

        try:
            traj = generator.generate_trajectory(
                question=q['question'],
                gold_answer=q.get('answer'),
            )

            if traj is None:
                raise ValueError("Generation returned None")

            # Convert to dict format
            traj_dict = traj.to_dict()

            results.append({
                'question_id': q['_id'],
                'question': q['question'],
                'gold_answer': q.get('answer'),
                'steps': traj_dict['steps'],
                'final_answer': traj_dict['final_answer'],
                'is_correct': traj_dict['is_correct'],
                'num_steps': len(traj_dict['steps']),
            })

            print(f"  → {len(traj_dict['steps'])} steps")
            print(f"  → Final answer: {str(traj_dict['final_answer'])[:50]}")
            print(f"  → Correct: {traj_dict['is_correct']}")

        except Exception as e:
            print(f"  → Error: {e}")
            results.append({
                'question_id': q['_id'],
                'question': q['question'],
                'gold_answer': q.get('answer'),
                'steps': [],
                'error': str(e),
            })

    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    success = sum(1 for r in results if r.get('steps'))
    correct = sum(1 for r in results if r.get('is_correct'))

    print(f"Total questions: {total}")
    print(f"Successfully generated: {success}")
    print(f"Correct answers: {correct} ({100*correct/total:.1f}%)")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
