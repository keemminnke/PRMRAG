#!/usr/bin/env python3
"""Batch test with HYBRID retrieval (BM25 + BGE-M3) on KILT corpus.

This script uses:
1. KILT Wikipedia corpus (5.9M documents)
2. BM25 for keyword matching
3. BGE-M3 for semantic search
4. RRF (Reciprocal Rank Fusion) to combine results
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
from prmrag.utils import load_config
from prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator
from prmrag.retrieval.bge_retriever import BGERetriever
from prmrag.retrieval.bm25_retriever import BM25Retriever
from prmrag.retrieval.hybrid_retriever import HybridRetriever
from prmrag.retrieval.bge_reranker import BGEReranker
import re


def get_truncated_text(doc_text_list, max_chars=1200):
    """Character-based truncation with truncation marker.

    Args:
        doc_text_list: List of paragraphs
        max_chars: Character limit (default: 1200)

    Returns:
        Truncated text string with [TRUNCATED] marker if cut
    """
    if not doc_text_list:
        return ""

    # Join all paragraphs
    joined_text = ' '.join(doc_text_list)

    # Truncate if exceeds limit
    if len(joined_text) > max_chars:
        return joined_text[:max_chars] + " [TRUNCATED]"

    return joined_text


def load_kilt_corpus(corpus_file: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus with smart truncation.

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
            # Character-only truncation (1200 chars max)
            if isinstance(doc['text'], list):
                text = get_truncated_text(doc['text'], max_chars=1200)
            else:
                # Single string: still apply character limit
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


def load_questions(questions_file: Path, start_idx: int = 0, limit: int = None) -> List[Dict[str, Any]]:
    """Load HotpotQA questions.

    Args:
        questions_file: Path to questions JSONL file
        start_idx: Starting line index in dataset (0-based)
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


def load_questions_sequential(
    questions_file: Path,
    *,
    start_idx: int = 0,
    limit: int | None = None,
    level: str | None = None,
    start_filtered_idx: int | None = None,
) -> List[Dict[str, Any]]:
    """Load HotpotQA questions in dataset order with optional filtering.

    Notes:
    - If `level` is provided, only questions with matching `level` are considered.
    - `start_filtered_idx` is the starting index within the filtered subset (0-based).
      It takes precedence over `start_idx` when used with `level`.
    - The returned question dicts are annotated with:
        - `__dataset_idx`: original line index in the JSONL file
        - `__filtered_idx`: index within the filtered subset (0-based, after applying `level`)
    """
    if level is None and start_filtered_idx is not None:
        # Without filtering, `start_filtered_idx` is equivalent to `start_idx`.
        start_idx = start_filtered_idx
        start_filtered_idx = None

    questions: List[Dict[str, Any]] = []
    filtered_seen = 0

    with open(questions_file, 'r') as f:
        # Fast path: no filtering and no filtered start index.
        if level is None and start_filtered_idx is None:
            for dataset_idx, line in enumerate(f):
                if dataset_idx < start_idx:
                    continue
                if limit and len(questions) >= limit:
                    break
                q = json.loads(line)
                q['__dataset_idx'] = dataset_idx
                questions.append(q)
            return questions

        for dataset_idx, line in enumerate(f):
            q = json.loads(line)

            q_level = q.get('level')
            if level is not None and q_level != level:
                continue

            filtered_idx = filtered_seen
            filtered_seen += 1

            if start_filtered_idx is not None:
                if filtered_idx < start_filtered_idx:
                    continue
            else:
                if dataset_idx < start_idx:
                    continue

            q['__dataset_idx'] = dataset_idx
            q['__filtered_idx'] = filtered_idx
            questions.append(q)

            if limit and len(questions) >= limit:
                break

    return questions


def iter_jsonl_safely(path: Path):
    """Yield JSON objects from a JSONL file, skipping malformed trailing lines."""
    if not path.exists():
        return
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Allow resuming even if the last line was partially written.
                continue


def load_processed_question_ids(results_file: Path, failures_file: Path) -> set[str]:
    processed: set[str] = set()
    for obj in iter_jsonl_safely(results_file):
        qid = obj.get('question_id')
        if qid:
            processed.add(qid)
    for obj in iter_jsonl_safely(failures_file):
        qid = obj.get('question_id')
        if qid:
            processed.add(qid)
    return processed


def make_empty_summary_stats() -> Dict[str, Any]:
    return {
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


def update_summary_stats(summary_stats: Dict[str, Any], result: Dict[str, Any]) -> None:
    summary_stats['total'] += 1
    if result.get('is_correct'):
        summary_stats['correct'] += 1

    if result.get('has_rag'):
        summary_stats['with_rag'] += 1
        if result.get('is_correct'):
            summary_stats['with_rag_correct'] += 1

        fc = result.get('format_compliance', {})
        summary_stats['format_compliance']['total_rag_steps'] += fc.get('total_rag_steps', 0)
        summary_stats['format_compliance']['rag_with_search_tags'] += fc.get('rag_with_search_tags', 0)
        summary_stats['format_compliance']['rag_with_citations'] += fc.get('rag_with_citations', 0)
        summary_stats['format_compliance']['rag_fully_compliant'] += fc.get('rag_fully_compliant', 0)

        for intervention in result.get('rag_interventions', []):
            summary_stats['rag_effectiveness']['total_rag_steps'] += 1
            mc_imp = intervention.get('mc_improvement', 0.0)
            summary_stats['rag_effectiveness']['total_mc_improvement'] += mc_imp

            if mc_imp > 0:
                summary_stats['rag_effectiveness']['rag_improved_mc'] += 1
            elif mc_imp < 0:
                summary_stats['rag_effectiveness']['rag_degraded_mc'] += 1

            if intervention.get('rpe', 0.0) >= 0.8:
                summary_stats['rag_effectiveness']['rpe_above_threshold'] += 1
    else:
        summary_stats['without_rag'] += 1
        if result.get('is_correct'):
            summary_stats['without_rag_correct'] += 1


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
        }

        # Include metadata for debugging backtracking
        if hasattr(step, 'metadata') and step.metadata:
            step_info['metadata'] = step.metadata

        # Only include non-null optional fields
        if hasattr(step, 'action_input') and step.action_input is not None:
            step_info['action_input'] = step.action_input
        if hasattr(step, 'observation') and step.observation is not None:
            step_info['observation'] = step.observation
        if hasattr(step, 'sub_answer') and step.sub_answer is not None:
            step_info['sub_answer'] = step.sub_answer
        if hasattr(step, 'retrieval_query') and step.retrieval_query is not None:
            step_info['retrieval_query'] = step.retrieval_query
        if hasattr(step, 'counterfactual_mc_cot') and step.counterfactual_mc_cot is not None:
            step_info['counterfactual_mc_cot'] = round(step.counterfactual_mc_cot, 3)
        if hasattr(step, 'counterfactual_mc_rag') and step.counterfactual_mc_rag is not None:
            step_info['counterfactual_mc_rag'] = round(step.counterfactual_mc_rag, 3)
        if hasattr(step, 'retrieval_necessity') and step.retrieval_necessity is not None:
            step_info['retrieval_necessity'] = step.retrieval_necessity.value
        if hasattr(step, 'confidence') and step.confidence is not None:
            step_info['confidence'] = round(step.confidence, 3)

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
        else:
            # CoT step: no retrieval
            step_info['num_passages'] = 0

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

    result = {
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

    # Include rejected_segments for DPO training if present
    if hasattr(trajectory, 'rejected_segments') and trajectory.rejected_segments:
        result['rejected_segments'] = trajectory.rejected_segments
        result['num_backtracks'] = len(trajectory.rejected_segments)

    return result


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
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "musique"],
        help="Question dataset to use (default: hotpotqa)",
    )
    parser.add_argument(
        "--question-level",
        type=str,
        default=None,
        help="Filter questions by the JSONL 'level' field (e.g., HotpotQA: medium, MuSiQue: 2hop/3hop1/4hop3)",
    )
    parser.add_argument(
        "--start-filtered-idx",
        type=int,
        default=None,
        help="Starting index within the filtered subset (0-based); use with --question-level",
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
        "--run-id",
        type=str,
        default=None,
        help="Run identifier used in output filenames (default: timestamp)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run-id: append to JSONL and skip already processed question_ids",
    )
    parser.add_argument(
        "--fsync",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fsync JSONL after each write for crash-safe incremental saving (default: enabled)",
    )
    parser.add_argument(
        "--use-reranker",
        action="store_true",
        help="Enable BGE reranker for improved retrieval quality (default: disabled)",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=20,
        help="Rerank top-N candidates from fusion (default: 20)",
    )
    parser.add_argument(
        "--question-ids-file",
        type=str,
        default=None,
        help="File containing question IDs to process (one per line)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume and not args.run_id:
        raise SystemExit("--resume requires --run-id so output filenames are deterministic.")
    run_id = args.run_id or timestamp
    results_file = output_dir / f"results_hybrid_{run_id}.jsonl"
    failures_file = output_dir / f"failures_hybrid_{run_id}.jsonl"
    if not args.resume and (results_file.exists() or failures_file.exists()):
        raise SystemExit(
            f"Output files already exist for run_id='{run_id}'. Use --resume or choose a different --run-id."
        )

    print("=" * 70)
    print(f"BATCH TEST: HYBRID Retrieval (BM25 + BGE-M3)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Data split: {args.split}")
    print(f"  - Number of questions: {args.num_questions}")
    print(f"  - Starting index: {args.start_idx}")
    print(f"  - MC rollouts: {args.num_rollouts}")
    print(f"  - Corpus: KILT Wikipedia (5.9M)")
    print(f"  - Fusion method: {args.fusion_method}")
    print(f"  - BM25 top-K: 50 (fixed)")
    print(f"  - BGE top-K: 50 (fixed)")
    print(f"  - Dynamic K: 32/64/128 (based on difficulty)")
    if args.use_reranker:
        print(f"  - Reranker: ENABLED (rerank top-{args.rerank_top_n})")
    else:
        print(f"  - Reranker: disabled")
    print(f"  - Output directory: {output_dir}")
    if args.question_level:
        start_desc = args.start_filtered_idx if args.start_filtered_idx is not None else args.start_idx
        print(f"  - Question level: {args.question_level} (start={start_desc})")
    if args.question_ids_file:
        print(f"  - Question IDs file: {args.question_ids_file}")
    if args.run_id:
        print(f"  - Run ID: {run_id}")
    print(f"  - JSONL fsync: {args.fsync}")

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load data
    data_dir = Path(__file__).parent.parent / "data"

    print(f"\n[1] Loading KILT corpus...")
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    corpus = load_kilt_corpus(corpus_file, limit=args.corpus_limit)

    print(f"\n[2] Loading questions from {args.split} split...")
    questions_file = data_dir / "raw" / "questions" / f"{args.dataset}_{args.split}.jsonl"
    if not questions_file.exists():
        raise SystemExit(
            f"Questions file not found: {questions_file}. "
            f"If you're using MuSiQue, run: python3 scripts/prepare_musique_questions.py"
        )
    questions = load_questions_sequential(
        questions_file,
        start_idx=args.start_idx,
        limit=args.num_questions,
        level=args.question_level,
        start_filtered_idx=args.start_filtered_idx,
    )

    # Filter by question IDs if file provided
    if args.question_ids_file:
        with open(args.question_ids_file, 'r') as f:
            target_ids = set(line.strip() for line in f if line.strip())
        questions = [q for q in questions if q.get('_id') in target_ids]
        print(f"✓ Filtered to {len(questions)} questions from {args.question_ids_file}")
    else:
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

    # BGE Reranker (optional)
    reranker = None
    if args.use_reranker:
        print(f"  [4.3] Initializing BGE Reranker...")
        reranker = BGEReranker(device="cuda")

    # Hybrid retriever
    print(f"  [4.4] Combining with {args.fusion_method.upper()} fusion...")
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        bge_retriever=bge_retriever,
        fusion_method=args.fusion_method,
        k_sparse=50,  # Fixed: BM25 retrieves top-50
        k_dense=50,   # Fixed: BGE retrieves top-50
        reranker=reranker,
        rerank_top_n=args.rerank_top_n if args.use_reranker else 20,
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

    summary_stats = make_empty_summary_stats()

    processed_question_ids: set[str] = set()
    if args.resume:
        if results_file.exists():
            for obj in iter_jsonl_safely(results_file):
                update_summary_stats(summary_stats, obj)
        if failures_file.exists():
            for _ in iter_jsonl_safely(failures_file):
                summary_stats['failed'] += 1
        processed_question_ids = load_processed_question_ids(results_file, failures_file)

    remaining = sum(1 for q in questions if q.get('_id') not in processed_question_ids)
    if processed_question_ids:
        print(f"\n✓ Resume mode: {len(processed_question_ids)} already processed; {remaining} remaining in this selection")

    # Open JSONL files for incremental writing
    results_fp = open(results_file, 'a' if args.resume else 'w', encoding='utf-8')
    failures_fp = open(failures_file, 'a' if args.resume else 'w', encoding='utf-8')

    for i, q in enumerate(questions, 1):
        question_id = q.get('_id')
        question = q['question']
        gold_answer = q['answer']

        if question_id in processed_question_ids:
            print(f"[{i}/{len(questions)}] Skipping {question_id} (already saved)")
            continue

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
                failure_rec = {
                    'question_id': question_id,
                    'question': question,
                    'gold_answer': gold_answer,
                    'failure_type': 'generation_failed',
                    'error': 'all_steps_rejected',
                    'dataset_idx': q.get('__dataset_idx'),
                    'filtered_idx': q.get('__filtered_idx'),
                    'question_level': q.get('level'),
                }
                failures_fp.write(json.dumps(failure_rec, ensure_ascii=False) + '\n')
                failures_fp.flush()
                if args.fsync:
                    os.fsync(failures_fp.fileno())
                processed_question_ids.add(question_id)
                continue

            # Format result
            result = format_trajectory_for_review(trajectory, q)

            # Add selection metadata (helps deterministic continuation)
            result['dataset_idx'] = q.get('__dataset_idx')
            result['filtered_idx'] = q.get('__filtered_idx')
            result['question_level'] = q.get('level')
            result['run_id'] = run_id

            # Write to file immediately (incremental save)
            results_fp.write(json.dumps(result, ensure_ascii=False) + '\n')
            results_fp.flush()
            if args.fsync:
                os.fsync(results_fp.fileno())

            processed_question_ids.add(question_id)
            update_summary_stats(summary_stats, result)

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
            failure_rec = {
                'question_id': question_id,
                'question': question,
                'gold_answer': gold_answer,
                'failure_type': 'exception',
                'error': str(e),
                'dataset_idx': q.get('__dataset_idx'),
                'filtered_idx': q.get('__filtered_idx'),
                'question_level': q.get('level'),
            }
            failures_fp.write(json.dumps(failure_rec, ensure_ascii=False) + '\n')
            failures_fp.flush()
            if args.fsync:
                os.fsync(failures_fp.fileno())
            processed_question_ids.add(question_id)
            continue

    # Close results file
    results_fp.close()
    failures_fp.close()
    print(f"\n✓ Detailed results saved to: {results_file}")
    print(f"✓ Failures saved to: {failures_file}")

    # Save summary stats
    summary_file = output_dir / f"summary_hybrid_{run_id}.json"
    summary_stats['run_id'] = run_id
    summary_stats['selection'] = {
        'dataset': args.dataset,
        'split': args.split,
        'num_questions': args.num_questions,
        'start_idx': args.start_idx,
        'question_level': args.question_level,
        'start_filtered_idx': args.start_filtered_idx,
        'results_file': str(results_file),
        'failures_file': str(failures_file),
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
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
