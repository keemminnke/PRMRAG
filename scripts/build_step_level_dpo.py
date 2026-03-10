#!/usr/bin/env python3
"""Build Step-level DPO dataset from existing scored trajectories.

Step-level DPO (ReasonRAG 방식):
  - Trajectory-level DPO: 전체 trajectory 비교 → 신호 희석
  - Step-level DPO: 동일 prefix에서 다음 1 step만 비교 → 명확한 학습 신호

Source 1: 기존 데이터에서 동일 prefix 공유하는 correct vs incorrect trajectory 비교
Source 2: BAD step 지점에서 K번 재생성 → GOOD이면 chosen, 원본 BAD는 rejected

Usage:
    # Source 1 only (기존 데이터만, 재생성 없음)
    python scripts/build_step_level_dpo.py \
        --input outputs/all_5000q_critic_v9_filtered.jsonl \
        --output outputs/step_dpo_5000q.jsonl

    # Source 1 + Source 2 (재생성 포함)
    python scripts/build_step_level_dpo.py \
        --input outputs/all_5000q_critic_v9_filtered.jsonl \
        --output outputs/step_dpo_5000q.jsonl \
        --regen --regen-k 3 \
        --policy-model Qwen/Qwen2.5-7B-Instruct \
        --critic-model outputs/critic_model_v9_3000q/final_model
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

HF_CACHE = "/home/work/.conda/storage/MINKEON_KIM/external_cache/huggingface"

SYSTEM_PROMPT = """You are an advanced AI agent capable of Adaptive RAG (Retrieval-Augmented Generation).
Your goal is to answer questions accurately by combining internal reasoning with external retrieval when needed.

# OUTPUT FORMAT

Use these XML tags for your response:

1. <think>Your reasoning</think>
   - Analyze the question, plan next action, evaluate evidence
   - ALWAYS start each step with <think>

2. <search>query</search>
   - Query external knowledge base
   - Use when you need factual information

3. <answer>final answer</answer>
   - Provide final answer (entity name or short answer only)
   - Use when you have sufficient evidence

After <search>, you will receive:
<documents>Retrieved passages</documents>

# STEP TYPES

- Search step: <think>...</think> followed by <search>...</search>
- Reason step: <think>...</think> only (no search)
- Finish step: <think>...</think> followed by <answer>...</answer>

# EXAMPLE

Question: Who directed the movie that won Best Picture at the 2020 Oscars?

<think>I need to find which movie won Best Picture at the 2020 Oscars, then identify its director.</think>
<search>Best Picture winner 2020 Oscars</search>
<documents>
[1] 92nd Academy Awards: "Parasite" won Best Picture at the 92nd Academy Awards (2020)...
[2] Parasite (2019 film): Directed by Bong Joon-ho, the film also won Best Director...
</documents>
<think>The observation states "Parasite" won and was directed by Bong Joon-ho. I have sufficient evidence.</think>
<answer>Bong Joon-ho</answer>

# RULES

1. One action per step - Either <search> or <answer>, not both
2. Always <think> first - Explain your reasoning before action
3. Search before guessing - If uncertain, use <search>
4. Trust observations - Retrieved information takes priority
5. Concise answer - Output only the entity name in <answer>

Begin."""


def get_question_id(trajectory_id: str) -> str:
    parts = trajectory_id.rsplit("_sample_", 1)
    return parts[0] if len(parts) == 2 else trajectory_id


def step_to_text(step: Dict) -> str:
    """Convert a single step to XML text."""
    parts = []
    think = step.get("think", "")
    if think:
        parts.append(f"<think>{think}</think>")
    search = step.get("search", "")
    answer = step.get("answer", "")
    if search:
        parts.append(f"<search>{search}</search>")
    elif answer:
        parts.append(f"<answer>{answer}</answer>")
    return "\n".join(parts)


def prefix_to_text(steps: List[Dict]) -> str:
    """Convert prefix steps (with documents) to text."""
    parts = []
    for step in steps:
        parts.append(step_to_text(step))
        docs = step.get("documents", "")
        if docs:
            parts.append(f"<documents>{docs}</documents>")
    return "\n".join(parts)


def build_prompt_messages(question: str, prefix_steps: List[Dict]) -> List[Dict]:
    """Build prompt as messages format: system + user + (optional) assistant prefix."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"},
    ]
    if prefix_steps:
        # Include prefix as start of assistant response
        prefix_text = prefix_to_text(prefix_steps)
        messages.append({"role": "assistant", "content": prefix_text})
    return messages


def build_step_messages(step: Dict) -> List[Dict]:
    """Build chosen/rejected as messages format: just the diverging step."""
    return [{"role": "assistant", "content": step_to_text(step)}]


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: Extract step-level pairs from existing data
# ─────────────────────────────────────────────────────────────────────────────

def extract_step_pairs(trajs: List[Dict]) -> List[Dict]:
    """Extract step-level DPO pairs from existing scored trajectories.

    Criterion: critic step label (GOOD=1 / BAD=0), NOT trajectory correctness.
    For each question, find trajectories sharing the same prefix (query sequence)
    up to step t-1, then compare their step t:
      - chosen: step with GOOD critic label
      - rejected: step with BAD critic label
    """
    by_q = defaultdict(list)
    for t in trajs:
        qid = get_question_id(t["trajectory_id"])
        by_q[qid].append(t)

    pairs = []
    stats = {"step_counts": defaultdict(int), "questions_used": set()}

    for qid, q_trajs in by_q.items():
        question = q_trajs[0].get("question", "")

        # For each step level, group by prefix
        max_steps = max(len(t.get("steps", [])) for t in q_trajs)

        for step_idx in range(min(max_steps, 8)):  # up to step 8
            prefix_groups = defaultdict(lambda: {"good": [], "bad": []})

            for t in q_trajs:
                steps = t.get("steps", [])
                labels = t.get("critic_step_labels", [])
                if len(steps) <= step_idx or len(labels) <= step_idx:
                    continue

                # Prefix key = tuple of queries from step 0 to step_idx-1
                if step_idx == 0:
                    prefix_key = ()
                else:
                    prefix_key = tuple(
                        s.get("search", "") for s in steps[:step_idx]
                    )

                # Classify by critic label at this step
                if labels[step_idx] == 1:  # GOOD
                    prefix_groups[prefix_key]["good"].append(t)
                else:  # BAD
                    prefix_groups[prefix_key]["bad"].append(t)

            for prefix_key, group in prefix_groups.items():
                if not group["good"] or not group["bad"]:
                    continue

                # Get prefix steps from any trajectory in this group
                ref_traj = group["good"][0]
                prefix_steps = ref_traj["steps"][:step_idx]

                # Create pairs: GOOD step vs BAD step at step_idx
                seen_pairs = set()
                for good_traj in group["good"]:
                    good_step = good_traj["steps"][step_idx]
                    good_step_text = step_to_text(good_step)

                    for bad_traj in group["bad"]:
                        bad_step = bad_traj["steps"][step_idx]
                        bad_step_text = step_to_text(bad_step)

                        # Skip if steps are identical
                        if good_step_text == bad_step_text:
                            continue

                        # Dedup
                        pair_key = (good_step_text, bad_step_text)
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)

                        prompt = build_prompt_messages(question, prefix_steps)
                        chosen = build_step_messages(good_step)
                        rejected = build_step_messages(bad_step)

                        pairs.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": rejected,
                            "question_id": qid,
                            "step_level": step_idx + 1,
                            "source": "existing",
                        })

                        stats["step_counts"][step_idx + 1] += 1
                        stats["questions_used"].add(qid)

    print(f"\n[Source 1] Existing data step-level pairs (GOOD vs BAD critic labels):")
    print(f"  Total pairs: {len(pairs):,}")
    print(f"  Questions used: {len(stats['questions_used']):,}")
    for step_num in sorted(stats["step_counts"]):
        print(f"  Step {step_num}: {stats['step_counts'][step_num]:,} pairs")

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: Regeneration at BAD step points
# ─────────────────────────────────────────────────────────────────────────────

def find_regen_targets(trajs: List[Dict]) -> List[Dict]:
    """Find unique prefix points where BAD steps occur."""
    by_q = defaultdict(list)
    for t in trajs:
        qid = get_question_id(t["trajectory_id"])
        by_q[qid].append(t)

    targets = []
    seen_prefixes = set()

    for qid, q_trajs in by_q.items():
        question = q_trajs[0].get("question", "")

        for t in q_trajs:
            steps = t.get("steps", [])
            labels = t.get("critic_step_labels", [])
            if not labels:
                continue

            for i, label in enumerate(labels):
                if label == 0:  # BAD step
                    # Prefix = steps[:i]
                    prefix_key = (qid, tuple(s.get("search", "") for s in steps[:i]))

                    if prefix_key in seen_prefixes:
                        break
                    seen_prefixes.add(prefix_key)

                    targets.append({
                        "question_id": qid,
                        "question": question,
                        "trajectory_id": t["trajectory_id"],
                        "prefix_steps": steps[:i],
                        "bad_step": steps[i],
                        "bad_step_idx": i,
                    })
                    break  # first BAD step only

    print(f"\n[Regen] Found {len(targets):,} unique regen targets")
    by_step = defaultdict(int)
    for t in targets:
        by_step[t["bad_step_idx"] + 1] += 1
    for s in sorted(by_step):
        print(f"  Step {s}: {by_step[s]:,} targets")

    return targets


def regenerate_steps(
    targets: List[Dict],
    policy_model: str,
    critic_model: str,
    critic_base: str,
    regen_k: int = 3,
    gpu_memory: float = 0.45,
) -> List[Dict]:
    """Regenerate steps at BAD points and score with critic.

    Returns list of step-level DPO pairs.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    import gc
    import torch

    # Check for cached regen results (skip generation if exists)
    regen_cache_path = "outputs/regen_step_results_cache.jsonl"
    if os.path.exists(regen_cache_path):
        print(f"\n[Regen] Loading cached regen results from {regen_cache_path}")
        regen_results = []
        with open(regen_cache_path) as f:
            for line in f:
                if line.strip():
                    regen_results.append(json.loads(line))
        print(f"  Loaded {len(regen_results):,} cached results — skipping generation")
    else:
        regen_results = _generate_regen_steps(
            targets, policy_model, regen_k, gpu_memory
        )

    # Score regenerated steps with critic
    print(f"\n[Regen] Scoring {len(regen_results):,} regenerated steps with critic...")

    critic_llm = LLM(
        model=critic_base,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
    )

    critic_model_abs = os.path.abspath(critic_model)
    lora_request = LoRARequest("critic", 1, critic_model_abs)

    critic_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=512,
    )

    # Build critic prompts
    critic_prompts = []
    for rr in regen_results:
        target = targets[rr["target_idx"]]
        step = rr["regen_step"]

        # Use same prompt format as compare_voting_methods.py
        system_content = (
            "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
            "The trajectory uses XML tags: <think> for reasoning, <search> for queries, <answer> for final answers, <documents> for retrieved passages. "
            "Analyze each step's logical soundness and evidence grounding. "
            "First explain your reasoning inside [REASONING] tags, then output a label (1=good, 0=bad)."
        )

        input_parts = [f"Question: {target['question']}", ""]
        # Previous steps
        if target["prefix_steps"]:
            input_parts.append("Previous Steps:")
            for j, prev in enumerate(target["prefix_steps"]):
                input_parts.append(f"## Step {j+1}")
                if prev.get('think'): input_parts.append(f"<think>{prev['think']}</think>")
                if prev.get('search'): input_parts.append(f"<search>{prev['search']}</search>")
                elif prev.get('answer'): input_parts.append(f"<answer>{prev['answer']}</answer>")
                if prev.get('documents'): input_parts.append(f"<documents>{prev['documents']}</documents>")
                input_parts.append("")
        # Current step to evaluate
        input_parts.append("Current Step to Evaluate:")
        input_parts.append(f"## Step {target['bad_step_idx'] + 1}")
        if step.get('think'): input_parts.append(f"<think>{step['think']}</think>")
        if step.get('search'): input_parts.append(f"<search>{step['search']}</search>")
        elif step.get('answer'): input_parts.append(f"<answer>{step['answer']}</answer>")
        input_parts.append("")
        input_parts.append("Task: Evaluate the quality of the Current Step. Explain your reasoning in [REASONING] tags, then provide a label (1=good, 0=bad).")

        user_content = "\n".join(input_parts)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        prompt_text = critic_llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        critic_prompts.append(prompt_text)

    critic_outputs = critic_llm.generate(critic_prompts, critic_sampling, lora_request=lora_request)

    # Parse critic scores — critic outputs "1" (good) or "0" (bad)
    for rr, critic_out in zip(regen_results, critic_outputs):
        text = critic_out.outputs[0].text.strip()
        # Find "1" or "0" in generated text
        rr["critic_label"] = 1 if "1" in text else 0

    del critic_llm
    gc.collect()
    torch.cuda.empty_cache()

    # Build DPO pairs from regen results
    pairs = []
    stats = {"good": 0, "bad": 0, "pairs": 0, "questions": set()}

    for rr in regen_results:
        target = targets[rr["target_idx"]]

        if rr["critic_label"] == 1:
            stats["good"] += 1
            # Chosen = regen step (GOOD), Rejected = original BAD step
            prompt = build_prompt_messages(
                target["question"], target["prefix_steps"]
            )
            chosen = build_step_messages(rr["regen_step"])
            rejected = build_step_messages(target["bad_step"])

            # Skip if identical
            if step_to_text(rr["regen_step"]) == step_to_text(target["bad_step"]):
                continue

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "question_id": target["question_id"],
                "step_level": target["bad_step_idx"] + 1,
                "source": "regen",
            })
            stats["pairs"] += 1
            stats["questions"].add(target["question_id"])
        else:
            stats["bad"] += 1

    print(f"\n[Regen] Results:")
    print(f"  GOOD regenerations: {stats['good']:,}")
    print(f"  BAD regenerations: {stats['bad']:,}")
    print(f"  Success rate: {100 * stats['good'] / max(1, stats['good'] + stats['bad']):.1f}%")
    print(f"  DPO pairs created: {stats['pairs']:,}")
    print(f"  Questions covered: {len(stats['questions']):,}")

    return pairs


def _generate_regen_steps(
    targets: List[Dict],
    policy_model: str,
    regen_k: int,
    gpu_memory: float,
) -> List[Dict]:
    """Generate regenerated steps with policy model."""
    from vllm import LLM, SamplingParams
    import gc, torch

    # Load policy model
    print(f"\n[Regen] Loading policy model: {policy_model}")
    policy_llm = LLM(
        model=policy_model,
        tensor_parallel_size=2,
        gpu_memory_utilization=gpu_memory,
        max_model_len=16384,
        trust_remote_code=True,
        download_dir=HF_CACHE,
        seed=42,
    )

    tokenizer = policy_llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        stop=["<search>", "<answer>", "</search>", "</answer>"],
        include_stop_str_in_output=True,
        n=regen_k,
    )

    # Build prompts for regeneration
    print(f"[Regen] Building prompts for {len(targets):,} targets (K={regen_k})...")
    prompts = []
    for target in targets:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {target['question']}"},
        ]
        if target["prefix_steps"]:
            prefix_text = prefix_to_text(target["prefix_steps"])
            messages.append({"role": "assistant", "content": prefix_text + "\n<think>"})
        else:
            messages.append({"role": "assistant", "content": "<think>"})

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompts.append(prompt_text)

    # Generate
    print(f"[Regen] Generating {len(prompts) * regen_k:,} step alternatives...")
    outputs = policy_llm.generate(prompts, sampling_params)

    # Collect regenerated steps
    regen_results = []
    for i, (target, output) in enumerate(zip(targets, outputs)):
        for j, completion in enumerate(output.outputs):
            text = "<think>" + completion.text
            # Parse the regenerated step
            regen_step = {"think": "", "search": "", "answer": ""}

            import re
            think_match = re.search(r"<think>(.*?)(?:</think>|$)", text, re.DOTALL)
            if think_match:
                regen_step["think"] = think_match.group(1).strip()

            search_match = re.search(r"<search>(.*?)(?:</search>|$)", text, re.DOTALL)
            answer_match = re.search(r"<answer>(.*?)(?:</answer>|$)", text, re.DOTALL)

            if search_match:
                regen_step["search"] = search_match.group(1).strip()
            elif answer_match:
                regen_step["answer"] = answer_match.group(1).strip()

            regen_results.append({
                "target_idx": i,
                "regen_idx": j,
                "regen_step": regen_step,
                "regen_text": text,
            })

    # Save regen results to disk (중간 결과 보존)
    regen_cache_path = "outputs/regen_step_results_cache.jsonl"
    with open(regen_cache_path, "w") as f:
        for rr in regen_results:
            f.write(json.dumps(rr, ensure_ascii=False) + "\n")
    print(f"[Regen] Saved {len(regen_results):,} regen results → {regen_cache_path}")

    # Free policy model
    del policy_llm
    gc.collect()
    torch.cuda.empty_cache()

    return regen_results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build Step-level DPO dataset")

    p.add_argument("--input", type=str, required=True,
                   help="Critic-scored trajectories (filtered)")
    p.add_argument("--output", type=str, required=True,
                   help="Output step-level DPO dataset")

    # Regen options
    p.add_argument("--regen", action="store_true",
                   help="Enable Source 2 (regeneration at BAD steps)")
    p.add_argument("--regen-k", type=int, default=3,
                   help="Number of regeneration attempts per BAD step")
    p.add_argument("--policy-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--critic-model", type=str,
                   default="outputs/critic_model_v9_3000q/final_model")
    p.add_argument("--critic-base", type=str,
                   default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B")
    p.add_argument("--gpu-memory", type=float, default=0.45)

    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Step-level DPO Dataset Builder")
    print("=" * 70)

    # Load data
    print(f"\n[Loading] {args.input}")
    trajs = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                trajs.append(json.loads(line))
    print(f"  Loaded {len(trajs):,} trajectories")

    # Source 1: existing data pairs
    print("\n" + "=" * 70)
    print("[Source 1] Extracting step-level pairs from existing data...")
    print("=" * 70)
    source1_pairs = extract_step_pairs(trajs)

    # Source 2: regeneration (optional)
    source2_pairs = []
    if args.regen:
        print("\n" + "=" * 70)
        print(f"[Source 2] Regeneration at BAD steps (K={args.regen_k})...")
        print("=" * 70)

        targets = find_regen_targets(trajs)
        source2_pairs = regenerate_steps(
            targets,
            policy_model=args.policy_model,
            critic_model=args.critic_model,
            critic_base=args.critic_base,
            regen_k=args.regen_k,
            gpu_memory=args.gpu_memory,
        )

    # Combine and save
    all_pairs = source1_pairs + source2_pairs

    print("\n" + "=" * 70)
    print("[Final] Step-level DPO Dataset Summary")
    print("=" * 70)
    print(f"  Source 1 (existing): {len(source1_pairs):,} pairs")
    print(f"  Source 2 (regen):    {len(source2_pairs):,} pairs")
    print(f"  Total:               {len(all_pairs):,} pairs")

    # Step distribution
    step_dist = defaultdict(int)
    source_dist = defaultdict(int)
    for p in all_pairs:
        step_dist[p["step_level"]] += 1
        source_dist[p["source"]] += 1

    print(f"\n  By step level:")
    for s in sorted(step_dist):
        print(f"    Step {s}: {step_dist[s]:,}")

    print(f"\n  By source:")
    for s in sorted(source_dist):
        print(f"    {s}: {source_dist[s]:,}")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n  Saved → {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
