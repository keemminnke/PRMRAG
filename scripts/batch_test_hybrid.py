#!/usr/bin/env python3
"""Batch test with HYBRID retrieval (BM25 + BGE-M3) on KILT corpus.

This script uses:
1. KILT Wikipedia corpus (5.9M documents)
2. BM25 for keyword matching
3. BGE-M3 for semantic search
4. RRF (Reciprocal Rank Fusion) to combine results
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator
from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever
from prmrag.retrieval.hybrid_retriever import HybridRetriever
import re


def load_kilt_corpus(corpus_file: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus.

    Args:
        corpus_file: Path to KILT knowledge source JSONL file
        limit: Optional limit on number of documents to load

    Returns:
        List of documents with 'id', 'title', 'text' fields
    """
    print(f"Loading KILT corpus from {corpus_file}...")

    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break

            doc = json.loads(line)

            # KILT format: text is a list of paragraphs
            # Join them into a single text
            if isinstance(doc['text'], list):
                text = ' '.join(doc['text'])
            else:
                text = doc['text']

            corpus.append({
                'id': doc['_id'],
                'title': doc['wikipedia_title'],
                'text': text,
            })

            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i+1:,} documents...")

    print(f"✓ Loaded {len(corpus):,} documents")
    return corpus


def load_questions(questions_file: Path, start_idx: int = 0, limit: int = None) -> List[Dict[str, Any]]:
    """Load HotpotQA questions.

    Args:
        questions_file: Path to questions JSONL file
        start_idx: Starting index
        limit: Number of questions to load

    Returns:
        List of questions
    """
    questions = []
    with open(questions_file, 'r') as f:
        for i, line in enumerate(f):
            if i < start_idx:
                continue
            if limit and len(questions) >= limit:
                break
            questions.append(json.loads(line))

    return questions


def validate_rag_step_format(step_content: str, step_type: str) -> Dict[str, Any]:
    """Validate if RAG step follows the required ReAct format."""
    validation = {
        'has_search_tags': False,
        'has_citations': False,
        'search_query': None,
        'num_citations': 0,
        'citation_numbers': [],
    }

    if step_type != 'rag':
        return validation

    # Check for Action: Search[query] format (ReAct style)
    action_match = re.search(r'Action:?\s*Search\s*\[\s*["\']?(.+?)["\']?\s*\]', step_content, re.IGNORECASE | re.DOTALL)
    if action_match:
        validation['has_search_tags'] = True
        validation['search_query'] = action_match.group(1).strip()

    # Check for citations: [1], [2], etc.
    citation_matches = re.findall(r'\[(\d+)\]', step_content)
    if citation_matches:
        validation['has_citations'] = True
        validation['num_citations'] = len(citation_matches)
        validation['citation_numbers'] = [int(c) for c in citation_matches]

    return validation


def format_trajectory_for_review(trajectory, question_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format trajectory for human review, highlighting RAG steps."""

    steps_detail = []
    format_compliance = {
        'total_rag_steps': 0,
        'rag_with_search_tags': 0,
        'rag_with_citations': 0,
        'rag_fully_compliant': 0,
    }

    for i, step in enumerate(trajectory.steps, 1):
        step_info = {
            'step_num': i,
            'content': step.content,
            'mc_before': round(step.mc_before, 3),
            'mc_after': round(step.mc_after, 3),
            'rpe': round(step.rpe, 3),
            'label': step.label,
            'thought': getattr(step, 'thought', None),
            'action': getattr(step, 'action', 'Reason'),
            'action_input': getattr(step, 'action_input', None),
            'observation': getattr(step, 'observation', None),
            'sub_answer': getattr(step, 'sub_answer', None),
            'retrieval_query': getattr(step, 'retrieval_query', None),
            'counterfactual_mc_cot': round(step.counterfactual_mc_cot, 3) if getattr(step, 'counterfactual_mc_cot', None) is not None else None,
            'counterfactual_mc_rag': round(step.counterfactual_mc_rag, 3) if getattr(step, 'counterfactual_mc_rag', None) is not None else None,
            'retrieval_necessity': getattr(step, 'retrieval_necessity', None).value if hasattr(step, 'retrieval_necessity') and step.retrieval_necessity else None,
            'confidence': round(step.confidence, 3) if getattr(step, 'confidence', None) is not None else None,
        }

        if hasattr(step, 'retrieval_evidence') and step.retrieval_evidence:
            step_info['retrieval_evidence'] = step.retrieval_evidence.to_dict()

        if step_info['action'] == 'Search':
            step_info['num_passages'] = len(step.used_passages)
            step_info['passage_titles'] = [p.get('title', 'Unknown') for p in step.used_passages]

            # Validate format
            validation = validate_rag_step_format(step.content, step.step_type.value)
            step_info['format_validation'] = validation

            # Update compliance stats
            format_compliance['total_rag_steps'] += 1
            if validation['has_search_tags']:
                format_compliance['rag_with_search_tags'] += 1
            if validation['has_citations']:
                format_compliance['rag_with_citations'] += 1
            if validation['has_search_tags'] and validation['has_citations']:
                format_compliance['rag_fully_compliant'] += 1

        steps_detail.append(step_info)

    # Find RAG intervention points
    rag_interventions = []
    for i, step in enumerate(steps_detail):
        if step['action'] == 'Search':
            before_mc = step['mc_before']
            after_mc = step['mc_after']
            improvement = after_mc - before_mc

            rag_interventions.append({
                'step_num': step['step_num'],
                'mc_improvement': round(improvement, 3),
                'rpe': step['rpe'],
                'label': step['label'],
                'passage_titles': step.get('passage_titles', []),
            })

    num_cot_steps = sum(1 for s in steps_detail if s['action'] == 'Reason')
    num_rag_steps = sum(1 for s in steps_detail if s['action'] == 'Search')

    return {
        'question_id': trajectory.trajectory_id,
        'question': trajectory.question,
        'gold_answer': trajectory.gold_answer,
        'predicted_answer': trajectory.final_answer,
        'is_correct': trajectory.is_correct,
        'num_steps': len(trajectory.steps),
        'num_cot_steps': num_cot_steps,
        'num_rag_steps': num_rag_steps,
        'has_rag': num_rag_steps > 0,
        'steps': steps_detail,
        'rag_interventions': rag_interventions,
        'format_compliance': format_compliance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch test with HYBRID retrieval (BM25 + BGE)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=20,
        help="Number of questions to test (default: 20)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in dataset (default: 0)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=8,
        help="Number of MC rollouts (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_test_hybrid",
        help="Output directory for results",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=None,
        help="Limit corpus size for testing (default: all)",
    )
    parser.add_argument(
        "--fusion-method",
        type=str,
        default="rrf",
        choices=["rrf", "weighted"],
        help="Fusion method: rrf or weighted (default: rrf)",
    )
    parser.add_argument(
        "--k-sparse",
        type=int,
        default=50,
        help="Top-K for BM25 (default: 50)",
    )
    parser.add_argument(
        "--k-dense",
        type=int,
        default=50,
        help="Top-K for BGE (default: 50)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print(f"BATCH TEST: HYBRID Retrieval (BM25 + BGE-M3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Data split: {args.split}")
    print(f"  - Number of questions: {args.num_questions}")
    print(f"  - Starting index: {args.start_idx}")
    print(f"  - MC rollouts: {args.num_rollouts}")
    print(f"  - Corpus: KILT Wikipedia (5.9M)")
    print(f"  - Fusion method: {args.fusion_method}")
    print(f"  - BM25 top-K: {args.k_sparse}")
    print(f"  - BGE top-K: {args.k_dense}")
    print(f"  - Output directory: {output_dir}")

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"

    print(f"\n[1] Loading KILT corpus...")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    corpus = load_kilt_corpus(corpus_file, limit=args.corpus_limit)

    print(f"\n[2] Loading questions from {args.split} split...")
    questions_file = data_dir / "raw" / "questions" / f"hotpotqa_{args.split}.jsonl"
    questions = load_questions(questions_file, start_idx=args.start_idx, limit=args.num_questions)
    print(f"✓ Loaded {len(questions)} questions")

    # Load model
    print(f"\n[3] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize retrievers
    print(f"\n[4] Initializing HYBRID retriever (BM25 + BGE-M3)...")

    # BGE retriever
    embedding_cache = data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy"
    print(f"  [4.1] Initializing BGE-M3 retriever...")
    bge_retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=str(embedding_cache)
    )

    # BM25 retriever
    bm25_cache = data_dir / "indexes" / "kilt_wikipedia_bm25.pkl"
    print(f"  [4.2] Initializing BM25 retriever...")
    bm25_retriever = BM25Retriever(
        corpus=corpus,
        index_cache_path=str(bm25_cache)
    )

    # Hybrid retriever
    print(f"  [4.3] Combining with {args.fusion_method.upper()} fusion...")
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        bge_retriever=bge_retriever,
        fusion_method=args.fusion_method,
        k_sparse=args.k_sparse,
        k_dense=args.k_dense,
    )
    print(f"✓ Hybrid retriever initialized!")

    # Initialize adaptive generator
    print(f"\n[5] Initializing adaptive generator...")
    gen_config = config.get('adaptive', {})
    gen_config['num_rollouts'] = args.num_rollouts

    generator = AdaptiveTrajectoryGenerator(
        policy_model=policy_model,
        retriever=hybrid_retriever,
        config=gen_config,
    )
    print("✓ Generator initialized")

    # Process questions
    print(f"\n{'=' * 70}")
    print(f"PROCESSING {len(questions)} QUESTIONS")
    print(f"{'=' * 70}\n")

    results = []
    summary_stats = {
        'total': 0,
        'correct': 0,
        'with_rag': 0,
        'with_rag_correct': 0,
        'without_rag': 0,
        'without_rag_correct': 0,
        'failed': 0,
        'format_compliance': {
            'total_rag_steps': 0,
            'rag_with_search_tags': 0,
            'rag_with_citations': 0,
            'rag_fully_compliant': 0,
        },
        'rag_effectiveness': {
            'total_rag_steps': 0,
            'rag_improved_mc': 0,
            'rag_degraded_mc': 0,
            'total_mc_improvement': 0.0,
            'rpe_above_threshold': 0,
        }
    }

    # Open results file for incremental writing
    results_file = output_dir / f"results_hybrid_{timestamp}.jsonl"
    results_fp = open(results_file, 'w')

    for i, q in enumerate(questions, 1):
        question_id = q['_id']
        question = q['question']
        gold_answer = q['answer']

        print(f"[{i}/{len(questions)}] Processing {question_id}...")
        print(f"  Q: {question[:80]}..." if len(question) > 80 else f"  Q: {question}")

        try:
            # Generate trajectory
            trajectory = generator.generate_trajectory(
                question=question,
                gold_answer=gold_answer,
                trajectory_id=question_id,
            )

            if trajectory is None:
                print(f"  ❌ Generation failed (all steps rejected)")
                summary_stats['failed'] += 1
                continue

            # Format result
            result = format_trajectory_for_review(trajectory, q)
            results.append(result)

            # Write to file immediately (incremental save)
            results_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
            results_fp.flush()  # Ensure it's written to disk

            # Update stats
            summary_stats['total'] += 1
            if result['is_correct']:
                summary_stats['correct'] += 1

            if result['has_rag']:
                summary_stats['with_rag'] += 1
                if result['is_correct']:
                    summary_stats['with_rag_correct'] += 1

                # Update format compliance stats
                fc = result['format_compliance']
                summary_stats['format_compliance']['total_rag_steps'] += fc['total_rag_steps']
                summary_stats['format_compliance']['rag_with_search_tags'] += fc['rag_with_search_tags']
                summary_stats['format_compliance']['rag_with_citations'] += fc['rag_with_citations']
                summary_stats['format_compliance']['rag_fully_compliant'] += fc['rag_fully_compliant']

                # Update RAG effectiveness stats
                for intervention in result['rag_interventions']:
                    summary_stats['rag_effectiveness']['total_rag_steps'] += 1
                    mc_imp = intervention['mc_improvement']
                    summary_stats['rag_effectiveness']['total_mc_improvement'] += mc_imp

                    if mc_imp > 0:
                        summary_stats['rag_effectiveness']['rag_improved_mc'] += 1
                    elif mc_imp < 0:
                        summary_stats['rag_effectiveness']['rag_degraded_mc'] += 1

                    if intervention['rpe'] >= 0.8:
                        summary_stats['rag_effectiveness']['rpe_above_threshold'] += 1
            else:
                summary_stats['without_rag'] += 1
                if result['is_correct']:
                    summary_stats['without_rag_correct'] += 1

            # Print summary
            status = "✅" if result['is_correct'] else "❌"
            rag_info = f"({result['num_rag_steps']} RAG steps)" if result['has_rag'] else "(CoT only)"

            if result['has_rag']:
                fc = result['format_compliance']
                compliance_rate = fc['rag_fully_compliant'] / max(fc['total_rag_steps'], 1) * 100
                rag_info += f" [Format: {fc['rag_fully_compliant']}/{fc['total_rag_steps']} ({compliance_rate:.0f}%)]"

            print(f"  {status} {result['num_steps']} steps {rag_info}")

            # Print progress every 100 questions
            if i % 100 == 0:
                print(f"\n✓ Progress: {i}/{len(questions)} questions completed ({summary_stats['correct']}/{summary_stats['total']} correct, {summary_stats['correct']/max(summary_stats['total'],1)*100:.1f}%)\n")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            summary_stats['failed'] += 1
            continue

    # Close results file
    results_fp.close()
    print(f"\n✓ Detailed results saved to: {results_file}")

    # Save summary stats
    summary_file = output_dir / f"summary_hybrid_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✓ Summary stats saved to: {summary_file}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 70}")
    print(f"\nTotal processed: {summary_stats['total']}")
    print(f"Failed: {summary_stats['failed']}")
    print(f"\nOverall accuracy: {summary_stats['correct']}/{summary_stats['total']} ({summary_stats['correct']/max(summary_stats['total'],1)*100:.1f}%)")

    print(f"\n--- RAG Usage Analysis ---")
    print(f"Questions with RAG: {summary_stats['with_rag']}")
    print(f"  - Correct: {summary_stats['with_rag_correct']}/{summary_stats['with_rag']} ({summary_stats['with_rag_correct']/max(summary_stats['with_rag'],1)*100:.1f}%)")

    print(f"\nQuestions without RAG (CoT only): {summary_stats['without_rag']}")
    print(f"  - Correct: {summary_stats['without_rag_correct']}/{summary_stats['without_rag']} ({summary_stats['without_rag_correct']/max(summary_stats['without_rag'],1)*100:.1f}%)")

    # Format compliance report
    if summary_stats['format_compliance']['total_rag_steps'] > 0:
        print(f"\n--- Format Compliance Analysis ---")
        fc = summary_stats['format_compliance']
        total_rag = fc['total_rag_steps']

        print(f"\nTotal RAG steps analyzed: {total_rag}")
        print(f"  ✓ With search tags: {fc['rag_with_search_tags']}/{total_rag} ({fc['rag_with_search_tags']/total_rag*100:.1f}%)")
        print(f"  ✓ With citations [N]: {fc['rag_with_citations']}/{total_rag} ({fc['rag_with_citations']/total_rag*100:.1f}%)")
        print(f"  ✓ Fully compliant (both): {fc['rag_fully_compliant']}/{total_rag} ({fc['rag_fully_compliant']/total_rag*100:.1f}%)")

    # RAG effectiveness report
    if summary_stats['rag_effectiveness']['total_rag_steps'] > 0:
        print(f"\n--- RAG Effectiveness Analysis ---")
        eff = summary_stats['rag_effectiveness']
        total_rag = eff['total_rag_steps']
        avg_improvement = eff['total_mc_improvement'] / total_rag

        print(f"\nTotal RAG interventions: {total_rag}")
        print(f"  ✓ Improved MC: {eff['rag_improved_mc']}/{total_rag} ({eff['rag_improved_mc']/total_rag*100:.1f}%)")
        print(f"  ✗ Degraded MC: {eff['rag_degraded_mc']}/{total_rag} ({eff['rag_degraded_mc']/total_rag*100:.1f}%)")
        print(f"  → Unchanged MC: {total_rag - eff['rag_improved_mc'] - eff['rag_degraded_mc']}/{total_rag}")
        print(f"\n  Average MC improvement: {avg_improvement:+.3f}")
        print(f"  RAG steps with RPE ≥ 0.8: {eff['rpe_above_threshold']}/{total_rag} ({eff['rpe_above_threshold']/total_rag*100:.1f}%)")

        if eff['rag_improved_mc'] > eff['rag_degraded_mc']:
            print(f"\n✅ RAG is effective: {eff['rag_improved_mc']} improvements vs {eff['rag_degraded_mc']} degradations")
        else:
            print(f"\n⚠️  Warning: RAG may not be effective: {eff['rag_improved_mc']} improvements vs {eff['rag_degraded_mc']} degradations")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
