#!/usr/bin/env python3
"""Critic-guided trajectory regeneration pipeline.

Flow:
1. Load trajectories + critic scores → merge by trajectory_id
2. Classify questions: Case A (has correct) vs Case B (all wrong)
3. Case A → best correct trajectory saved directly (rejection sampling)
4. Case B → truncate at first BAD step
5. Phase 1: Load critic model → generate reasoning for BAD steps → unload
6. Phase 2: Load retriever + policy model → regenerate from truncation point
7. Save all results to JSONL

Usage:
    python scripts/regenerate_trajectories.py \
        --trajectories outputs/trajectories_merged_1000q.jsonl \
        --critic_scores outputs/critic_scores_1000q.jsonl \
        --output_path outputs/regenerated_trajectories.jsonl \
        --limit 10
"""

import sys
import os
import re
import json
import argparse
import gc
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


def build_critic_prompt(tokenizer, question: str, steps: List[Dict], step_idx: int) -> str:
    """Build prompt for critic model to generate reasoning for a step.

    Uses the same format as CriticDataFormatter (training format).
    """
    system_content = (
        "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
        "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, <documents> for retrieved passages. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
    )

    input_parts = [f"Question: {question}", ""]

    # Previous steps (NO truncation — match training format)
    if step_idx > 0:
        input_parts.append("Previous Steps:")
        for j, prev in enumerate(steps[:step_idx]):
            input_parts.append(f"## Step {j+1}")
            if prev.get('think'):
                input_parts.append(f"<think>{prev['think']}</think>")
            if prev.get('search'):
                input_parts.append(f"<search>{prev['search']}</search>")
            elif prev.get('answer'):
                input_parts.append(f"<answer>{prev['answer']}</answer>")
            if prev.get('documents'):
                input_parts.append(f"<documents>{prev['documents']}</documents>")
            input_parts.append("")

    # Current step to evaluate (NO truncation — match training format)
    current_step = steps[step_idx]
    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {step_idx + 1}")
    if current_step.get('think'):
        input_parts.append(f"<think>{current_step['think']}</think>")
    if current_step.get('search'):
        input_parts.append(f"<search>{current_step['search']}</search>")
    elif current_step.get('answer'):
        input_parts.append(f"<answer>{current_step['answer']}</answer>")
    if current_step.get('documents'):
        input_parts.append(f"<documents>{current_step['documents']}</documents>")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

    user_content = "\n".join(input_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_reasoning(text: str) -> str:
    """Extract [REASONING] content from critic model output."""
    match = re.search(r'\[REASONING\]\s*(.*?)\s*\[/REASONING\]', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: take everything before "Label:"
    label_match = re.search(r'Label:\s*\d', text)
    if label_match:
        return text[:label_match.start()].strip()

    return text.strip()


def generate_critic_reasonings(
    truncation_results,
    scored_trajectories_by_id,
    args,
) -> Dict[str, str]:
    """Phase 1: Load critic model, generate reasoning for all BAD steps, unload.

    Returns:
        Dict mapping trajectory_id → critic reasoning string
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    download_dir = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

    print("  Loading critic model tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.critic_base_model,
        trust_remote_code=True,
        cache_dir=download_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading critic model with vLLM...")
    print(f"    Base model: {args.critic_base_model}")
    print(f"    LoRA adapter: {args.critic_model}")
    llm = LLM(
        model=args.critic_base_model,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.4,
        max_model_len=8192,
        trust_remote_code=True,
        download_dir=download_dir,
    )
    lora_request = LoRARequest("critic", 1, args.critic_model)

    # Build prompts for all BAD steps
    prompts = []
    traj_ids = []

    for tr in truncation_results:
        # Get the original trajectory's steps (with full content)
        scored_traj = scored_trajectories_by_id.get(tr.trajectory_id)
        if scored_traj is None:
            continue

        steps = scored_traj.steps
        # bad_step_id is 1-indexed, convert to 0-indexed for step_idx
        bad_step_idx = tr.bad_step_id - 1
        if bad_step_idx >= len(steps):
            bad_step_idx = len(steps) - 1

        prompt = build_critic_prompt(tokenizer, tr.question, steps, bad_step_idx)
        prompts.append(prompt)
        traj_ids.append(tr.trajectory_id)

    print(f"  Generating reasoning for {len(prompts)} BAD steps...")

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
    )

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # Parse reasonings
    reasonings = {}
    for traj_id, output_obj in zip(traj_ids, outputs):
        text = output_obj.outputs[0].text
        reasoning = extract_reasoning(text)
        reasonings[traj_id] = reasoning

    print(f"  Generated {len(reasonings)} critic reasonings")

    # Sample output
    for i, (tid, reason) in enumerate(reasonings.items()):
        if i >= 3:
            break
        print(f"    [{tid}]: {reason[:100]}...")

    # Unload critic model to free GPU
    print("  Unloading critic model...")
    del llm
    del tokenizer
    del lora_request
    gc.collect()

    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  Critic model unloaded, GPU memory freed")

    return reasonings


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

    # Critic model settings
    parser.add_argument('--critic_model', type=str,
                        default='outputs/critic_model_v7_1000q/final_model',
                        help='Path to critic LoRA adapter')
    parser.add_argument('--critic_base_model', type=str,
                        default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                        help='Critic base model name')

    # Policy model settings
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
    print("\n[1/7] Loading and merging trajectories + critic scores...")
    scored_trajectories = QuestionClassifier.load_and_merge(
        args.trajectories, args.critic_scores
    )

    # Build lookup by trajectory_id for critic reasoning generation
    scored_trajectories_by_id = {st.trajectory_id: st for st in scored_trajectories}

    print("\n[2/7] Classifying questions (Case A vs Case B)...")
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
    print(f"\n[3/7] Saving Case A results (best correct trajectory)...")
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
    # Step 4: Truncate Case B trajectories
    # =========================================================
    case_b_to_process = [
        cq for cq in case_b_questions if cq.question_id not in processed_ids
    ]

    if args.limit is not None:
        case_b_to_process = case_b_to_process[:args.limit]

    if not case_b_to_process:
        print("\n[4/7] No Case B questions to regenerate. Done!")
        _print_summary(args.output_path, case_a_saved, 0, 0)
        return

    print(f"\n[4/7] Truncating {len(case_b_to_process)} Case B trajectories...")

    truncator = TrajectoryTruncator()
    truncation_results = []
    for cq in case_b_to_process:
        tr = truncator.truncate(cq.best_trajectory)
        truncation_results.append(tr)

    avg_good = sum(len(tr.good_steps) for tr in truncation_results) / len(truncation_results)
    print(f"  Truncation summary:")
    print(f"    Avg good steps preserved: {avg_good:.1f}")
    print(f"    Cases with 0 good steps: {sum(1 for tr in truncation_results if len(tr.good_steps) == 0)}")

    # =========================================================
    # Step 5: Use judge reasoning from judge_labels (skip critic model)
    # =========================================================
    print(f"\n[5/7] Loading judge reasoning for BAD steps...")

    for tr in truncation_results:
        scored_traj = scored_trajectories_by_id.get(tr.trajectory_id)
        if scored_traj is None:
            tr.critic_reasoning = ""
            continue
        bad_step_idx = tr.bad_step_id - 1  # 1-indexed → 0-indexed
        if bad_step_idx < len(scored_traj.steps):
            tr.critic_reasoning = scored_traj.steps[bad_step_idx].get('judge_reasoning', '')
        else:
            tr.critic_reasoning = ""

    has_reasoning = sum(1 for tr in truncation_results if tr.critic_reasoning)
    print(f"  Loaded {has_reasoning}/{len(truncation_results)} judge reasonings")
    for tr in truncation_results[:3]:
        print(f"    [{tr.trajectory_id}]: {tr.critic_reasoning[:100]}...")

    # =========================================================
    # Step 6: Phase 2 - Load retriever + policy model
    # =========================================================
    print(f"\n[6/7] Phase 2: Loading retriever and policy model...")

    # Load KILT corpus + BGE-M3 retriever
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
        device="cpu",
    )

    # Clean up CUDA context before vLLM fork
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  CUDA context cleaned before vLLM init")

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
    # Step 7: Regenerate Case B
    # =========================================================
    print(f"\n[7/7] Regenerating {len(truncation_results)} Case B trajectories...")

    regenerator = CriticGuidedRegenerator(
        policy_model=policy_model,
        retriever=retriever,
        max_steps=args.max_steps,
        top_k_passages=args.top_k,
        temperature=args.temperature,
    )

    case_b_correct = 0
    case_b_total = 0

    with open(args.output_path, 'a', encoding='utf-8') as f:
        for i, tr in enumerate(truncation_results):
            print(f"  [{i+1}/{len(truncation_results)}] {tr.question_id}: "
                  f"truncated at step {tr.bad_step_id}, "
                  f"{len(tr.good_steps)} good steps, "
                  f"reasoning: {tr.critic_reasoning[:50]}...")

            result = regenerator.regenerate_single(tr)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

            case_b_total += 1
            if result['is_correct']:
                case_b_correct += 1
                print(f"    → CORRECT: '{result['predicted_answer']}'")
            else:
                print(f"    → wrong: '{result['predicted_answer']}'")

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
