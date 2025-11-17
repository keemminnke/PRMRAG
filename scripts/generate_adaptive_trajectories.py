#!/usr/bin/env python3
"""
Generate adaptive MC-CoT + RAG trajectories (Stage 1).

This script:
1. Loads HotpotQA questions and corpus
2. Initializes BM25 retriever
3. Generates adaptive trajectories with MC monitoring
4. Saves trajectories with mc_before, mc_after for each step
"""

import argparse
import sys
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation import AdaptiveTrajectoryGenerator
from prmrag.retrieval import BM25Retriever, WikipediaRetriever, load_hotpotqa_corpus, create_retriever
from prmrag.utils import setup_logger, load_config


def load_hotpotqa_questions(file_path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load HotpotQA questions.

    Expected format:
    {
        "_id": "question_id",
        "question": "What is...?",
        "answer": "The answer",
        "supporting_facts": [["title1", 0], ["title2", 1]],
        "context": [["title", ["sent1", "sent2", ...]], ...]
    }
    """
    questions = []

    with jsonlines.open(file_path) as reader:
        for i, obj in enumerate(reader):
            if limit and i >= limit:
                break

            questions.append({
                'id': obj.get('_id', f"q_{i}"),
                'question': obj['question'],
                'gold_answer': obj.get('answer', ''),
                'supporting_facts': obj.get('supporting_facts', []),
                'context': obj.get('context', []),
            })

    return questions


def main():
    parser = argparse.ArgumentParser(
        description="Generate adaptive MC-CoT + RAG trajectories"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="configs/adaptive_generation.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        required=True,
        help="Path to HotpotQA questions (JSONL)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to document corpus (JSONL) - only needed for BM25",
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default=None,
        help="Retrieval method: bm25 or wikipedia (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output trajectories",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=3,
        help="Number of trajectories per question",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Setup logging
    log_file = Path(config['logging']['log_dir']) / "generate_trajectories.log"
    logger = setup_logger("prmrag", level=config['logging']['level'], log_file=log_file)

    logger.info("=" * 60)
    logger.info("ADAPTIVE MC-CoT + RAG TRAJECTORY GENERATION")
    logger.info("=" * 60)
    logger.info(f"Questions: {args.questions}")
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")

    # Load questions
    logger.info(f"\n[1/4] Loading HotpotQA questions...")
    questions = load_hotpotqa_questions(args.questions, limit=args.limit)
    logger.info(f"Loaded {len(questions)} questions")

    # Load corpus and build retriever
    logger.info(f"\n[2/4] Building retriever...")

    retrieval_method = args.retrieval_method or config['retrieval']['method']
    logger.info(f"Retrieval method: {retrieval_method}")

    if retrieval_method == "bm25":
        if not args.corpus:
            raise ValueError("--corpus is required for BM25 retrieval")

        logger.info(f"Loading corpus from {args.corpus}")
        corpus = load_hotpotqa_corpus(args.corpus)
        logger.info(f"Loaded {len(corpus)} documents")

        retriever = BM25Retriever(corpus)
        logger.info("BM25 index ready!")

    elif retrieval_method == "wikipedia":
        logger.info("Initializing Wikipedia retriever...")
        wiki_config = config['retrieval'].get('wikipedia', {})

        retriever = WikipediaRetriever(
            lang=wiki_config.get('lang', 'en'),
            user_agent=wiki_config.get('user_agent', 'PRMRAG/1.0'),
            extract_sentences=wiki_config.get('extract_sentences', 5),
        )
        logger.info("Wikipedia retriever ready!")

    else:
        raise ValueError(f"Unknown retrieval method: {retrieval_method}")

    # Initialize generator
    logger.info(f"\n[3/4] Initializing adaptive trajectory generator...")
    logger.info(f"  Policy model: {config['policy_model']['model_name']}")
    logger.info(f"  MC rollouts: {config['adaptive']['num_rollouts']}")
    logger.info(f"  Delta (CoT threshold): {config['adaptive']['delta']}")
    logger.info(f"  Epsilon (RAG threshold): {config['adaptive']['epsilon']}")

    # Load policy model
    from prmrag.models import load_policy_model
    logger.info("Loading policy model...")
    policy_model = load_policy_model(config['policy_model'])
    logger.info("Policy model loaded!")

    generator = AdaptiveTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
        config=config['adaptive'],
    )

    # Generate trajectories
    logger.info(f"\n[4/4] Generating adaptive trajectories...")
    logger.info(f"  Trajectories per question: {args.num_trajectories}")
    logger.info(f"  Total trajectories to generate: {len(questions) * args.num_trajectories}")

    trajectories = generator.generate_batch(
        questions,
        num_trajectories_per_question=args.num_trajectories,
        show_progress=True,
    )

    logger.info(f"\nGenerated {len(trajectories)} trajectories")

    # Compute statistics
    total_steps = sum(len(t.steps) for t in trajectories)
    cot_steps = sum(t.metadata['num_cot_steps'] for t in trajectories)
    rag_steps = sum(t.metadata['num_rag_steps'] for t in trajectories)
    correct = sum(1 for t in trajectories if t.is_correct)

    logger.info(f"\nStatistics:")
    logger.info(f"  Total steps: {total_steps}")
    if total_steps > 0:
        logger.info(f"  CoT steps: {cot_steps} ({cot_steps/total_steps*100:.1f}%)")
        logger.info(f"  RAG steps: {rag_steps} ({rag_steps/total_steps*100:.1f}%)")
    else:
        logger.info(f"  CoT steps: {cot_steps}")
        logger.info(f"  RAG steps: {rag_steps}")
    if len(trajectories) > 0:
        logger.info(f"  Correct answers: {correct}/{len(trajectories)} ({correct/len(trajectories)*100:.1f}%)")
    else:
        logger.info(f"  Correct answers: 0/0")

    # Save trajectories
    logger.info(f"\nSaving trajectories to {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(args.output, mode='w') as writer:
        for traj in trajectories:
            writer.write(traj.to_dict())

    # Save statistics
    stats_output = args.output.parent / f"{args.output.stem}_stats.json"
    with open(stats_output, 'w') as f:
        json.dump({
            'num_questions': len(questions),
            'num_trajectories': len(trajectories),
            'total_steps': total_steps,
            'cot_steps': cot_steps,
            'rag_steps': rag_steps,
            'correct_answers': correct,
            'accuracy': correct / len(trajectories) if trajectories else 0,
            'avg_steps_per_trajectory': total_steps / len(trajectories) if trajectories else 0,
        }, f, indent=2)

    logger.info(f"Statistics saved to {stats_output}")

    logger.info("\n" + "=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
