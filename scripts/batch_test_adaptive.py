#!/usr/bin/env python3
"""Batch test adaptive pipeline on multiple questions to analyze RAG effectiveness.

This script:
1. Tests N questions from HotpotQA validation set
2. Tracks which steps used RAG and which used CoT
3. Saves detailed results for manual inspection
4. Generates summary statistics
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
from prmrag.retrieval import BGERetriever
import re


def load_validation_data(data_dir: Path, limit: int = None):
    """Load validation questions and corpus."""
    questions_file = data_dir / "raw" / "hotpotqa_validation.jsonl"
    corpus_file = data_dir / "raw" / "hotpotqa_validation_corpus.jsonl"

    # Load questions
    questions = []
    with open(questions_file, 'r') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            questions.append(json.loads(line))

    # Load corpus
    corpus = []
    with open(corpus_file, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))

    return questions, corpus


def validate_rag_step_format(step_content: str, step_type: str) -> Dict[str, Any]:
    """Validate if RAG step follows the required format.

    Checks for:
    1. <start_search>query</end_search> tags
    2. <document> tags (should be in passages, not step content)
    3. Citation format (e.g., "According to [1]", "Based on [2]")
    4. No hallucination markers

    Returns:
        Dict with validation results
    """
    validation = {
        'has_search_tags': False,
        'has_citations': False,
        'search_query': None,
        'num_citations': 0,
        'citation_numbers': [],
    }

    if step_type != 'rag':
        return validation

    # Check for <start_search> tags
    search_match = re.search(r'<start_search>(.*?)</end_search>', step_content, re.DOTALL)
    if search_match:
        validation['has_search_tags'] = True
        validation['search_query'] = search_match.group(1).strip()

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
            'type': step.step_type.value,  # 'cot' or 'rag'
            'content': step.content,
            'mc_before': round(step.mc_before, 3),
            'mc_after': round(step.mc_after, 3),
            'rpe': round(step.rpe, 3),
            'label': step.label,
            'is_rag': step.step_type.value == 'rag',
        }

        if step_info['is_rag']:
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
        if step['is_rag']:
            # Look at the step before and after to see the impact
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

    return {
        'question_id': trajectory.trajectory_id,
        'question': trajectory.question,
        'gold_answer': trajectory.gold_answer,
        'predicted_answer': trajectory.final_answer,
        'is_correct': trajectory.is_correct,
        'num_steps': len(trajectory.steps),
        'num_cot_steps': trajectory.metadata['num_cot_steps'],
        'num_rag_steps': trajectory.metadata['num_rag_steps'],
        'has_rag': trajectory.metadata['num_rag_steps'] > 0,
        'steps': steps_detail,
        'rag_interventions': rag_interventions,
        'format_compliance': format_compliance,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch test adaptive pipeline to analyze RAG effectiveness"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=100,
        help="Number of questions to test (default: 100)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index in validation set (default: 0)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=8,
        help="Number of MC rollouts (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/batch_test",
        help="Output directory for results (default: outputs/batch_test)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 70)
    print("BATCH TEST: Adaptive MC-CoT + RAG Pipeline")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Number of questions: {args.num_questions}")
    print(f"  - Starting index: {args.start_idx}")
    print(f"  - MC rollouts: {args.num_rollouts}")
    print(f"  - Output directory: {output_dir}")

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load validation data
    print(f"\n[1] Loading validation data...")
    data_dir = Path(__file__).parent.parent / "data"
    questions, corpus = load_validation_data(
        data_dir,
        limit=args.start_idx + args.num_questions
    )
    questions = questions[args.start_idx:args.start_idx + args.num_questions]
    print(f"✓ Loaded {len(questions)} questions and {len(corpus)} corpus documents")

    # Load model
    print(f"\n[2] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize BGE-M3 retriever
    print(f"\n[3] Initializing BGE-M3 retriever...")
    embedding_cache = "data/embeddings/hotpotqa_validation_bge_m3.npy"
    retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=embedding_cache
    )
    print(f"✓ Retriever initialized")

    # Initialize adaptive generator
    print(f"\n[4] Initializing adaptive generator...")
    gen_config = config.get('adaptive', {})
    gen_config['num_rollouts'] = args.num_rollouts

    generator = AdaptiveTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
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
        }
    }

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
            else:
                summary_stats['without_rag'] += 1
                if result['is_correct']:
                    summary_stats['without_rag_correct'] += 1

            # Print summary
            status = "✅" if result['is_correct'] else "❌"
            rag_info = f"({result['num_rag_steps']} RAG steps)" if result['has_rag'] else "(CoT only)"

            # Add format compliance info for RAG steps
            if result['has_rag']:
                fc = result['format_compliance']
                compliance_rate = fc['rag_fully_compliant'] / max(fc['total_rag_steps'], 1) * 100
                rag_info += f" [Format: {fc['rag_fully_compliant']}/{fc['total_rag_steps']} ({compliance_rate:.0f}%)]"

            print(f"  {status} {result['num_steps']} steps {rag_info}")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            summary_stats['failed'] += 1
            continue

    # Save detailed results
    results_file = output_dir / f"results_{timestamp}.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"\n✓ Detailed results saved to: {results_file}")

    # Save summary stats
    summary_file = output_dir / f"summary_{timestamp}.json"
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
        print(f"  ✓ With <start_search> tags: {fc['rag_with_search_tags']}/{total_rag} ({fc['rag_with_search_tags']/total_rag*100:.1f}%)")
        print(f"  ✓ With citations [N]: {fc['rag_with_citations']}/{total_rag} ({fc['rag_with_citations']/total_rag*100:.1f}%)")
        print(f"  ✓ Fully compliant (both): {fc['rag_fully_compliant']}/{total_rag} ({fc['rag_fully_compliant']/total_rag*100:.1f}%)")

        if fc['rag_fully_compliant'] < total_rag:
            print(f"\n⚠️  Warning: {total_rag - fc['rag_fully_compliant']} RAG steps are not fully compliant with format requirements")

    print(f"\n{'=' * 70}")
    print("NEXT STEPS")
    print(f"{'=' * 70}")
    print(f"\nTo analyze RAG effectiveness in detail:")
    print(f"  python3 scripts/analyze_rag_impact.py {results_file}")
    print(f"\nTo view specific trajectories:")
    print(f"  python3 scripts/view_trajectory.py {results_file} --filter-rag")
    print()


if __name__ == "__main__":
    main()
