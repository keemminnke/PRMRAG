#!/usr/bin/env python3
"""Score regenerated trajectories with the Critic v8 model.

Takes regenerated trajectories (from regenerate_bad_steps.py) and evaluates
each step with the critic model, producing step-level labels, scores, and reasoning.

Input:  regenerated_trajectories.jsonl
        (output from regenerate_bad_steps.py with steps[])

Output: regenerated_critic_scored.jsonl
        (same format + critic_label, critic_score, critic_reasoning per step,
         plus trajectory-level critic_step_labels, critic_step_scores, critic_min)

Usage:
    # Small test
    python scripts/score_regenerated.py \
        --input outputs/regenerated_trajectories.jsonl \
        --output outputs/regenerated_critic_scored.jsonl \
        --limit 10

    # Full run
    nohup python scripts/score_regenerated.py \
        --input outputs/regenerated_trajectories.jsonl \
        --output outputs/regenerated_critic_scored.jsonl \
        > logs/score_regenerated.log 2>&1 &
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# Critic prompt building (reused from compare_voting_methods.py)
# ============================================================

def parse_step_content_xml(step: dict) -> dict:
    """Parse step content from XML format."""
    return {
        'think': step.get('think', ''),
        'search': step.get('search', ''),
        'answer': step.get('answer', ''),
        'documents': step.get('documents', ''),
        'step_type': step.get('step_type', 'unknown'),
    }


def build_critic_prompt_xml(tokenizer, question: str, steps: List[Dict], step_idx: int) -> str:
    """Build prompt for critic evaluation using XML format and binary labels (1/0)."""
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
            parsed = parse_step_content_xml(prev)
            input_parts.append(f"## Step {j+1}")
            if parsed['think']:
                input_parts.append(f"<think>{parsed['think']}</think>")
            if parsed['search']:
                input_parts.append(f"<search>{parsed['search']}</search>")
            elif parsed['answer']:
                input_parts.append(f"<answer>{parsed['answer']}</answer>")
            if parsed['documents']:
                input_parts.append(f"<documents>{parsed['documents']}</documents>")
            input_parts.append("")

    # Current step
    parsed = parse_step_content_xml(steps[step_idx])
    input_parts.append("Current Step to Evaluate:")
    input_parts.append(f"## Step {step_idx + 1}")
    if parsed['think']:
        input_parts.append(f"<think>{parsed['think']}</think>")
    if parsed['search']:
        input_parts.append(f"<search>{parsed['search']}</search>")
    elif parsed['answer']:
        input_parts.append(f"<answer>{parsed['answer']}</answer>")
    if parsed['documents']:
        input_parts.append(f"<documents>{parsed['documents']}</documents>")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

    user_content = "\n".join(input_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _extract_soft_score(logprobs_list, token_id_1: int, token_id_0: int, binary_label: int) -> float:
    """Extract soft score from logprobs at the label token position."""
    search_start = len(logprobs_list) - 1
    search_end = max(len(logprobs_list) - 5, -1)

    for i in range(search_start, search_end, -1):
        if i < 0:
            break
        token_logprobs = logprobs_list[i]
        if token_logprobs is None:
            continue

        logp_1 = None
        logp_0 = None

        for tid, lp in token_logprobs.items():
            if tid == token_id_1:
                logp_1 = lp.logprob
            elif tid == token_id_0:
                logp_0 = lp.logprob

        if logp_1 is not None or logp_0 is not None:
            if logp_1 is not None and logp_0 is not None:
                p_1 = math.exp(logp_1)
                p_0 = math.exp(logp_0)
                return p_1 / (p_1 + p_0)
            elif logp_1 is not None:
                return 1.0 - 1e-6
            elif logp_0 is not None:
                return 1e-6

    return 1.0 if binary_label == 1 else (0.0 if binary_label == 0 else 0.5)


# ============================================================
# Critic model loading
# ============================================================

def load_critic_model(critic_path: str, base_model: str, gpu_memory_utilization: float = 0.9):
    """Load critic model with vLLM + LoRA."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    download_dir = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

    print(f"Loading Critic model with vLLM...")
    print(f"  Base model: {base_model}")
    print(f"  LoRA adapter: {critic_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        cache_dir=download_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=download_dir,
    )

    lora_request = LoRARequest("critic", 1, critic_path)

    print("  Critic model loaded")
    return llm, tokenizer, lora_request


# ============================================================
# Batch scoring
# ============================================================

def score_trajectories_batch(
    llm, tokenizer, lora_request,
    trajectories: List[Dict],
    batch_size: int = 64,
) -> List[Dict]:
    """Score all trajectories with critic model.

    For each trajectory, evaluates every step and adds:
      - per-step: critic_label, critic_score, critic_reasoning
      - per-trajectory: critic_step_labels, critic_step_scores, critic_min

    Returns list of scored trajectory dicts.
    """
    from vllm import SamplingParams

    token_id_1 = tokenizer.encode("1", add_special_tokens=False)[-1]
    token_id_0 = tokenizer.encode("0", add_special_tokens=False)[-1]
    print(f"  Soft scoring: token_id('1')={token_id_1}, token_id('0')={token_id_0}")

    # Collect all prompts
    all_prompts = []
    prompt_info = []  # (traj_idx, step_idx)

    for traj_idx, traj in enumerate(trajectories):
        question = traj['question']
        steps = traj.get('steps', [])
        for step_idx in range(len(steps)):
            prompt = build_critic_prompt_xml(tokenizer, question, steps, step_idx)
            all_prompts.append(prompt)
            prompt_info.append((traj_idx, step_idx))

    total_prompts = len(all_prompts)
    print(f"  Total prompts: {total_prompts} ({len(trajectories)} trajectories)")

    if not all_prompts:
        return trajectories

    # Process in batches
    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0,
        logprobs=20,
        stop=["Label: 1", "Label: 0", "Label:1", "Label:0"],
        include_stop_str_in_output=True,
    )

    # Collect results
    results = {i: {'labels': [], 'scores': [], 'reasonings': []}
               for i in range(len(trajectories))}

    for batch_start in range(0, total_prompts, batch_size):
        batch_end = min(batch_start + batch_size, total_prompts)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_info = prompt_info[batch_start:batch_end]

        print(f"    Batch {batch_start//batch_size + 1}: "
              f"prompts {batch_start+1}-{batch_end}/{total_prompts}")

        outputs = llm.generate(
            batch_prompts,
            sampling_params,
            lora_request=lora_request,
        )

        for (traj_idx, step_idx), output_obj in zip(batch_info, outputs):
            response = output_obj.outputs[0]
            text = response.text

            # Binary label
            label = -1
            if "Label: 1" in text or "Label:1" in text or text.strip().endswith("1"):
                label = 1
            elif "Label: 0" in text or "Label:0" in text or text.strip().endswith("0"):
                label = 0

            # Soft score
            if response.logprobs:
                score = _extract_soft_score(response.logprobs, token_id_1, token_id_0, label)
            else:
                score = 1.0 if label == 1 else (0.0 if label == 0 else 0.5)

            # Reasoning
            clean_reasoning = text
            if "[REASONING]" in clean_reasoning and "[/REASONING]" in clean_reasoning:
                clean_reasoning = clean_reasoning.split("[REASONING]")[1].split("[/REASONING]")[0]
            else:
                clean_reasoning = clean_reasoning.replace("[REASONING]", "").replace("[/REASONING]", "")
                if "Label:" in clean_reasoning:
                    clean_reasoning = clean_reasoning.split("Label:")[0]
            clean_reasoning = clean_reasoning.strip()

            results[traj_idx]['labels'].append(label)
            results[traj_idx]['scores'].append(score)
            results[traj_idx]['reasonings'].append(clean_reasoning)

    # Merge results into trajectories
    scored_trajectories = []
    all_scores = []

    for traj_idx, traj in enumerate(trajectories):
        scored_traj = dict(traj)
        labels = results[traj_idx]['labels']
        scores = results[traj_idx]['scores']
        reasonings = results[traj_idx]['reasonings']

        # Add per-step critic info
        steps = scored_traj.get('steps', [])
        for i, step in enumerate(steps):
            if i < len(labels):
                step['critic_label'] = labels[i]
                step['critic_score'] = scores[i]
                step['critic_reasoning'] = reasonings[i]

        # Add trajectory-level aggregates
        scored_traj['critic_step_labels'] = labels
        scored_traj['critic_step_scores'] = scores
        scored_traj['critic_min'] = min(scores) if scores else 0.0

        scored_trajectories.append(scored_traj)
        all_scores.extend(scores)

    # Print diagnostics
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        good_count = sum(1 for s in all_scores if s >= 0.5)
        bad_count = len(all_scores) - good_count
        print(f"\n  Scoring stats ({len(all_scores)} steps):")
        print(f"    Mean score: {avg_score:.4f}")
        print(f"    GOOD (>=0.5): {good_count} ({good_count/len(all_scores)*100:.1f}%)")
        print(f"    BAD  (<0.5):  {bad_count} ({bad_count/len(all_scores)*100:.1f}%)")

    return scored_trajectories


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score regenerated trajectories with Critic model"
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to regenerated_trajectories.jsonl',
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path to output scored trajectories JSONL',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of trajectories (for debugging)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing output (skip already processed)')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Number of prompts per vLLM batch (default: 256)')

    # Critic model settings
    parser.add_argument('--critic-model', type=str,
                        default='outputs/critic_model_v8_2000q/final_model',
                        help='Path to critic LoRA adapter')
    parser.add_argument('--critic-base', type=str,
                        default='deepseek-ai/DeepSeek-R1-0528-Qwen3-8B',
                        help='Base model for critic')
    parser.add_argument('--gpu-memory', type=float, default=0.9,
                        help='GPU memory utilization for vLLM')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Critic Re-evaluation of Regenerated Trajectories")
    print("=" * 70)
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Critic: {args.critic_model}")
    print(f"  Base:   {args.critic_base}")
    if args.limit:
        print(f"  Limit:  {args.limit}")
    print()

    # =========================================================
    # 1. Load regenerated trajectories
    # =========================================================
    print("[1/3] Loading regenerated trajectories...")
    trajectories = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"  Loaded {len(trajectories)} trajectories")

    total_steps = sum(len(t.get('steps', [])) for t in trajectories)
    print(f"  Total steps: {total_steps}")

    if args.limit:
        trajectories = trajectories[:args.limit]
        print(f"  Limited to {len(trajectories)} trajectories")

    # Resume handling
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        print(f"  Resume mode: loading existing results...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    processed_ids.add(record.get('trajectory_id', ''))
        print(f"  Found {len(processed_ids)} already processed")
        trajectories = [t for t in trajectories if t['trajectory_id'] not in processed_ids]
        print(f"  Remaining: {len(trajectories)} trajectories")

    if not trajectories:
        print("  Nothing to score. Done!")
        return

    # =========================================================
    # 2. Load critic model
    # =========================================================
    print(f"\n[2/3] Loading critic model...")
    llm, tokenizer, lora_request = load_critic_model(
        args.critic_model, args.critic_base,
        gpu_memory_utilization=args.gpu_memory,
    )

    # =========================================================
    # 3. Score trajectories
    # =========================================================
    print(f"\n[3/3] Scoring {len(trajectories)} trajectories...")

    scored = score_trajectories_batch(
        llm, tokenizer, lora_request,
        trajectories,
        batch_size=args.batch_size,
    )

    # Write output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    file_mode = 'a' if args.resume and processed_ids else 'w'
    with open(args.output, file_mode, encoding='utf-8') as f:
        for traj in scored:
            f.write(json.dumps(traj, ensure_ascii=False) + '\n')

    # Summary
    correct_count = sum(1 for t in scored if t.get('is_correct', False))
    all_good_count = sum(1 for t in scored
                         if all(l == 1 for l in t.get('critic_step_labels', [])))
    has_bad_count = sum(1 for t in scored
                        if any(l == 0 for l in t.get('critic_step_labels', [])))

    print()
    print("=" * 70)
    print("Scoring Complete!")
    print("=" * 70)
    print(f"  Total scored:    {len(scored)}")
    print(f"  Correct answer:  {correct_count}/{len(scored)} "
          f"({correct_count/len(scored)*100:.1f}%)" if scored else "")
    print(f"  All-GOOD steps:  {all_good_count} ({all_good_count/len(scored)*100:.1f}%)" if scored else "")
    print(f"  Has BAD steps:   {has_bad_count} ({has_bad_count/len(scored)*100:.1f}%)" if scored else "")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
