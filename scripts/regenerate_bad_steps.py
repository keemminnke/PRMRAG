#!/usr/bin/env python3
"""Regenerate BAD steps in trajectories using GenPRM multi-turn pattern.

Takes trajectories with critic labels, truncates at the first BAD step,
and regenerates using the policy model + retriever with critic reasoning feedback.

Auto-resume: If output file exists, automatically skips already processed trajectories.

Input:  hotpotqa_critic_results_v8_per_trajectory.jsonl
        (single file with step-level critic_label, critic_reasoning)

Output: regenerated_trajectories.jsonl
        (same format, with regenerated steps replacing BAD steps)

Usage:
    # Small test
    python scripts/regenerate_bad_steps.py \
        --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --output outputs/regenerated_trajectories.jsonl \
        --limit 10

    # Full run (background, auto-resumes if killed)
    nohup python scripts/regenerate_bad_steps.py \
        --input outputs/hotpotqa_critic_results_v8_per_trajectory.jsonl \
        --output outputs/regenerated_trajectories.jsonl \
        > logs/regenerate_bad_steps.log 2>&1 &
"""

import sys
import os
import json
import gc
import signal
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Graceful shutdown flag
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    sig_name = signal.Signals(signum).name
    print(f"\n  [!] {sig_name} received. Finishing current trajectory then exiting...")
    _shutdown_requested = True


def load_kilt_corpus(corpus_file: str) -> List[Dict[str, Any]]:
    """Load KILT Wikipedia corpus (with pickle cache)."""
    corpus_file = Path(corpus_file)
    cache_path = corpus_file.parent / (corpus_file.stem + "_parsed.pkl")

    if cache_path.exists():
        print(f"  Loading cached corpus from {cache_path}...")
        with open(cache_path, 'rb') as f:
            corpus = pickle.load(f)
        print(f"  Loaded {len(corpus):,} documents from cache")
        return corpus

    print(f"  Loading KILT corpus from {corpus_file}...")
    corpus = []
    with open(corpus_file, 'r') as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            doc_id = doc.get('id', doc.get('_id'))
            title = doc.get('title', doc.get('wikipedia_title'))
            text = doc['text']
            if isinstance(text, list):
                text = ' '.join(text)
            if len(text) > 1200:
                text = text[:1200]
            corpus.append({'id': doc_id, 'title': title, 'text': text})
            if (i + 1) % 100000 == 0:
                print(f"    Loaded {i+1:,} documents...")

    print(f"  Loaded {len(corpus):,} documents")
    print(f"  Saving corpus cache to {cache_path}...")
    with open(cache_path, 'wb') as f:
        pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
    return corpus


def load_trajectories(input_path: str):
    """Load trajectories from hotpotqa_critic_results format."""
    trajectories = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def find_bad_trajectories(trajectories: List[Dict]) -> List[Dict]:
    """Filter trajectories that have at least one BAD step."""
    bad_trajs = []
    for traj in trajectories:
        for step in traj['steps']:
            if step.get('critic_label') == 0:
                bad_trajs.append(traj)
                break
    return bad_trajs


def truncate_at_first_bad(traj: Dict) -> Dict:
    """Find first BAD step, return truncation info.

    Returns dict with:
        good_steps: steps before the first BAD step
        bad_step: the BAD step itself
        bad_step_idx: 0-indexed position
        critic_reasoning: reasoning for why the step is BAD
    """
    for i, step in enumerate(traj['steps']):
        if step.get('critic_label') == 0:
            return {
                'trajectory_id': traj['trajectory_id'],
                'question': traj['question'],
                'gold_answer': traj['gold_answer'],
                'good_steps': traj['steps'][:i],
                'bad_step': step,
                'bad_step_idx': i,
                'critic_reasoning': step.get('critic_reasoning', ''),
                'all_original_steps': traj['steps'],
                'original_predicted_answer': traj.get('predicted_answer', ''),
                'original_is_correct': traj.get('is_correct', False),
            }

    # Should not reach here (filtered by find_bad_trajectories)
    return None


def build_truncation_result(trunc_info: Dict):
    """Convert truncation info dict to TruncationResult for CriticGuidedRegenerator."""
    from prmrag.regeneration.truncator import TruncationResult

    return TruncationResult(
        trajectory_id=trunc_info['trajectory_id'],
        question_id=trunc_info['trajectory_id'].rsplit('_sample_', 1)[0],
        question=trunc_info['question'],
        gold_answer=trunc_info['gold_answer'],
        good_steps=trunc_info['good_steps'],
        bad_step=trunc_info['bad_step'],
        all_original_steps=trunc_info['all_original_steps'],
        bad_step_id=trunc_info['bad_step_idx'] + 1,  # 1-indexed
        bad_step_critic_score=trunc_info['bad_step'].get('critic_score', 0.0),
        original_num_steps=len(trunc_info['all_original_steps']),
        original_predicted_answer=trunc_info['original_predicted_answer'],
        original_is_correct=trunc_info['original_is_correct'],
        critic_reasoning=trunc_info['critic_reasoning'],
    )


def load_processed_ids(output_path: str) -> set:
    """Load already-processed trajectory IDs from output file."""
    processed = set()
    if not os.path.exists(output_path):
        return processed

    valid_lines = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                tid = record.get('trajectory_id', '')
                # Strip _regen suffix to match original trajectory_id
                processed.add(tid.replace('_regen', ''))
                valid_lines += 1
            except json.JSONDecodeError:
                # Truncated line from killed process — ignore
                break

    # If last line was truncated, rewrite file without it
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Validate all lines
        clean_lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                json.loads(line)
                clean_lines.append(line)
            except json.JSONDecodeError:
                print(f"  [!] Removed truncated line from output file")
                break
        if len(clean_lines) != valid_lines:
            # Something was off, but clean_lines should be good
            pass
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines)

    return processed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate BAD steps in trajectories"
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to hotpotqa_critic_results_v8_per_trajectory.jsonl',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to output regenerated trajectories JSONL',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trajectories to regenerate (for debugging)')

    # Policy model settings
    parser.add_argument('--policy_model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_steps', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=5)

    # GPU settings
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--max_model_len', type=int, default=16384)

    return parser.parse_args()


def main():
    global _shutdown_requested

    args = parse_args()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    print("=" * 70)
    print("BAD Step Regeneration Pipeline")
    print("=" * 70)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    if args.limit:
        print(f"  Limit:  {args.limit}")
    print()

    # =========================================================
    # 1. Load trajectories
    # =========================================================
    print("[1/5] Loading trajectories...")
    all_trajectories = load_trajectories(args.input)
    print(f"  Total trajectories: {len(all_trajectories)}")

    bad_trajectories = find_bad_trajectories(all_trajectories)
    all_good_count = len(all_trajectories) - len(bad_trajectories)
    print(f"  All-GOOD (skip): {all_good_count}")
    print(f"  Has BAD step (regenerate): {len(bad_trajectories)}")

    # =========================================================
    # 2. Truncate at first BAD step
    # =========================================================
    print(f"\n[2/5] Truncating at first BAD step...")

    truncation_infos = []
    for traj in bad_trajectories:
        info = truncate_at_first_bad(traj)
        if info:
            truncation_infos.append(info)

    if args.limit:
        truncation_infos = truncation_infos[:args.limit]

    # Auto-resume: load already processed IDs
    print(f"  Checking for existing output...")
    processed_ids = load_processed_ids(args.output)
    if processed_ids:
        print(f"  Auto-resume: {len(processed_ids)} already processed, skipping")
    truncation_infos = [t for t in truncation_infos if t['trajectory_id'] not in processed_ids]

    print(f"  To regenerate: {len(truncation_infos)}")

    if not truncation_infos:
        print("  Nothing to regenerate. Done!")
        return

    # Stats
    avg_good = sum(len(t['good_steps']) for t in truncation_infos) / len(truncation_infos)
    zero_good = sum(1 for t in truncation_infos if len(t['good_steps']) == 0)
    print(f"  Avg good steps preserved: {avg_good:.1f}")
    print(f"  Cases with 0 good steps: {zero_good}")

    # Sample
    for t in truncation_infos[:3]:
        print(f"    [{t['trajectory_id']}] bad at step {t['bad_step_idx']+1}, "
              f"good={len(t['good_steps'])}, reasoning: {t['critic_reasoning'][:80]}...")

    # Free original trajectory data (no longer needed)
    del all_trajectories, bad_trajectories
    gc.collect()

    # =========================================================
    # 3. Load retriever + corpus
    # =========================================================
    print(f"\n[3/5] Loading retriever and corpus...")

    data_dir = Path(__file__).parent.parent / "data"
    corpus_file = data_dir / "kilt" / "kilt_knowledgesource.json"
    embedding_cache = str(data_dir / "embeddings" / "kilt_wikipedia_bge_m3.npy")
    faiss_index_cache = str(data_dir / "indexes" / "kilt_wikipedia_bge_m3.faiss")

    corpus = load_kilt_corpus(str(corpus_file))

    from prmrag.retrieval.bge_retriever import BGERetriever
    print("  Initializing BGE-M3 retriever...")
    retriever = BGERetriever(
        corpus=corpus,
        batch_size=64,
        embedding_cache_path=embedding_cache,
        device="cpu",
        faiss_index_path=faiss_index_cache,
    )

    # =========================================================
    # 4. Load policy model
    # =========================================================
    print(f"\n[4/5] Loading policy model...")

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    from prmrag.models import load_policy_model
    policy_config = {
        'model_name': args.policy_model,
        'temperature': args.temperature,
        'gpu_memory_utilization': args.gpu_memory_utilization,
        'max_model_len': args.max_model_len,
    }
    policy_model = load_policy_model(policy_config)

    # =========================================================
    # 5. Regenerate
    # =========================================================
    print(f"\n[5/5] Regenerating {len(truncation_infos)} trajectories...")

    from prmrag.regeneration.regenerator import CriticGuidedRegenerator

    regenerator = CriticGuidedRegenerator(
        policy_model=policy_model,
        retriever=retriever,
        max_steps=args.max_steps,
        top_k_passages=args.top_k,
        temperature=args.temperature,
    )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    correct_count = 0
    total_count = 0
    # Count existing correct from resume
    if processed_ids and os.path.exists(args.output):
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        total_count += 1
                        if r.get('is_correct'):
                            correct_count += 1
                    except json.JSONDecodeError:
                        pass
        if total_count > 0:
            print(f"  Resume stats: {correct_count}/{total_count} correct "
                  f"({correct_count/total_count*100:.1f}%)")

    with open(args.output, 'a', encoding='utf-8') as f:
        for i, trunc_info in enumerate(truncation_infos):
            if _shutdown_requested:
                print(f"\n  [!] Graceful shutdown after {total_count} trajectories.")
                break

            tr = build_truncation_result(trunc_info)

            print(f"  [{i+1}/{len(truncation_infos)}] {tr.trajectory_id}: "
                  f"truncated at step {tr.bad_step_id}, "
                  f"{len(tr.good_steps)} good steps")

            try:
                result = regenerator.regenerate_single(tr)
            except Exception as e:
                print(f"    -> ERROR: {e}")
                # Write error record so we don't retry this trajectory
                result = {
                    'trajectory_id': tr.trajectory_id + '_regen',
                    'question_id': tr.question_id,
                    'question': tr.question,
                    'gold_answer': tr.gold_answer,
                    'predicted_answer': '',
                    'is_correct': False,
                    'case': 'error',
                    'steps': [],
                    'metadata': {'error': str(e)},
                }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            os.fsync(f.fileno())  # Force write to disk

            total_count += 1
            if result.get('is_correct'):
                correct_count += 1
                print(f"    -> CORRECT: '{result['predicted_answer']}'")
            else:
                print(f"    -> wrong: '{result.get('predicted_answer', '')}' "
                      f"(gold: '{tr.gold_answer}')")

            if (i + 1) % 50 == 0:
                print(f"\n    === Progress: {total_count} done, "
                      f"correct: {correct_count}/{total_count} "
                      f"({correct_count/total_count*100:.1f}%) ===\n")

    # Summary
    print()
    print("=" * 70)
    if _shutdown_requested:
        print("Regeneration Paused (will auto-resume on next run)")
    else:
        print("Regeneration Complete!")
    print("=" * 70)
    print(f"  Total regenerated: {total_count}")
    if total_count > 0:
        print(f"  Correct: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
    print(f"  Output: {args.output}")
    remaining = len(truncation_infos) - (i + 1) if not _shutdown_requested else len(truncation_infos) - i
    if remaining > 0 or _shutdown_requested:
        print(f"  Remaining: {remaining}")
        print(f"  To resume: re-run the same command")


if __name__ == '__main__':
    main()
