#!/usr/bin/env python3
"""Compare three voting methods on trajectories:
1. Our Critic Model - step-level evaluation, select best trajectory
2. Majority Voting - most common final answer
3. VersaPRM - process reward model scoring

Usage:
    python scripts/compare_voting_methods.py \
        --trajectories outputs/test_trajectories_full.jsonl \
        --critic-model outputs/critic_model_debug/final_model \
        --output outputs/voting_comparison.json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# Method 1: Our Critic Model (vLLM)
# ============================================================

def load_critic_model_vllm(critic_path: str, base_model: str, gpu_memory_utilization: float = 0.4):
    """Load our trained critic model with vLLM for fast inference."""
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    print(f"Loading Critic model with vLLM...")
    print(f"  Base model: {base_model}")
    print(f"  LoRA adapter: {critic_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load vLLM with LoRA support
    llm = LLM(
        model=base_model,
        enable_lora=True,
        max_lora_rank=64,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Create LoRA request
    lora_request = LoRARequest("critic", 1, critic_path)

    print("✓ Critic model loaded with vLLM")
    return llm, tokenizer, lora_request


def parse_step_content(step: dict) -> dict:
    """Parse step content."""
    if step.get('thought') or step.get('action'):
        return {
            'thought': step.get('thought', ''),
            'action': step.get('action', 'Unknown'),
            'action_input': step.get('action_input', ''),
            'observation': step.get('observation', ''),
        }

    content = step.get('content', step.get('text', ''))
    thought, action, action_input, observation = '', 'Unknown', '', ''

    if 'Thought:' in content:
        start = content.find('Thought:') + len('Thought:')
        end = content.find('Action:') if 'Action:' in content else len(content)
        thought = content[start:end].strip()

    if 'Action:' in content:
        start = content.find('Action:') + len('Action:')
        end = content.find('Observation:') if 'Observation:' in content else len(content)
        action_text = content[start:end].strip()
        if '[' in action_text and ']' in action_text:
            bracket_start = action_text.find('[')
            action = action_text[:bracket_start].strip()
            action_input = action_text[bracket_start+1:action_text.rfind(']')].strip()
        else:
            action = action_text.split('\n')[0].strip()

    if 'Observation:' in content:
        start = content.find('Observation:') + len('Observation:')
        observation = content[start:].strip()[:500]

    return {'thought': thought, 'action': action, 'action_input': action_input, 'observation': observation}


def build_critic_prompt(tokenizer, question: str, steps: List[Dict], step_idx: int) -> str:
    """Build prompt for critic evaluation of a single step."""
    system_content = (
        "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First, provide a reasoning explanation inside <think> tags, and then output the final label (GOOD/BAD)."
    )

    input_parts = [f"Question: {question}", ""]

    # Previous steps
    if step_idx > 0:
        input_parts.append("Previous Steps:")
        for j, prev in enumerate(steps[:step_idx]):
            parsed = parse_step_content(prev)
            input_parts.append(f"Step {j+1}:")
            if parsed['thought']:
                input_parts.append(f"Thought: {parsed['thought'][:300]}")
            if parsed['action_input']:
                input_parts.append(f"Action: {parsed['action']}[{parsed['action_input']}]")
            else:
                input_parts.append(f"Action: {parsed['action']}")
            if parsed['observation']:
                input_parts.append(f"Observation: {parsed['observation'][:300]}")
            input_parts.append("")

    # Current step
    parsed = parse_step_content(steps[step_idx])
    input_parts.append("Current Step to Evaluate:")
    if parsed['thought']:
        input_parts.append(f"Thought: {parsed['thought'][:500]}")
    if parsed['action_input']:
        input_parts.append(f"Action: {parsed['action']}[{parsed['action_input']}]")
    else:
        input_parts.append(f"Action: {parsed['action']}")
    if parsed['observation']:
        input_parts.append(f"Observation: {parsed['observation'][:500]}")
    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step.")

    user_content = "\n".join(input_parts)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def batch_evaluate_with_critic_vllm(
    llm, tokenizer, lora_request, trajectories: List[Dict]
) -> List[Tuple[List[str], float]]:
    """Batch evaluate multiple trajectories with critic using vLLM.

    Returns:
        List of (step_labels, trajectory_score) for each trajectory
    """
    from vllm import SamplingParams

    # Collect all (traj_idx, step_idx, prompt) pairs
    all_prompts = []
    prompt_info = []  # (traj_idx, step_idx)

    for traj_idx, traj in enumerate(trajectories):
        question = traj['question']
        steps = traj.get('steps', [])
        for step_idx in range(len(steps)):
            prompt = build_critic_prompt(tokenizer, question, steps, step_idx)
            all_prompts.append(prompt)
            prompt_info.append((traj_idx, step_idx))

    if not all_prompts:
        return [([], 0.0) for _ in trajectories]

    # vLLM batch generate (all at once - vLLM handles batching internally)
    print(f"  Generating {len(all_prompts)} critic evaluations with vLLM...")
    sampling_params = SamplingParams(
        max_tokens=128,  # Reduced - we just need the label
        temperature=0,
        stop=["Label: GOOD", "Label: BAD", "Label:GOOD", "Label:BAD"],
        include_stop_str_in_output=True,
    )

    outputs = llm.generate(
        all_prompts,
        sampling_params,
        lora_request=lora_request,
    )

    # Extract responses
    all_responses = [output.outputs[0].text for output in outputs]

    # Parse labels and aggregate scores
    results = {i: {'labels': [], 'good_count': 0, 'total': 0} for i in range(len(trajectories))}

    for (traj_idx, step_idx), response in zip(prompt_info, all_responses):
        label = "UNKNOWN"
        if "Label: GOOD" in response or "Label:GOOD" in response or response.strip().endswith("GOOD"):
            label = "GOOD"
            results[traj_idx]['good_count'] += 1
        elif "Label: BAD" in response or "Label:BAD" in response or response.strip().endswith("BAD"):
            label = "BAD"
        results[traj_idx]['labels'].append(label)
        results[traj_idx]['total'] += 1

    # Build final results
    final_results = []
    for i in range(len(trajectories)):
        labels = results[i]['labels']
        total = results[i]['total']
        good = results[i]['good_count']
        score = good / total if total > 0 else 0.0
        final_results.append((labels, score))

    return final_results


# ============================================================
# Method 2: Majority Voting
# ============================================================

def majority_vote(trajectories: List[Dict]) -> Tuple[str, int]:
    """Simple majority voting on final answers.

    Returns:
        Tuple of (best_answer, vote_count)
    """
    answers = []
    for traj in trajectories:
        answer = traj.get('final_answer', '')
        if answer:
            answers.append(answer.strip().lower())

    if not answers:
        return "", 0

    counter = Counter(answers)
    best_answer, count = counter.most_common(1)[0]

    # Return original case
    for traj in trajectories:
        if traj.get('final_answer', '').strip().lower() == best_answer:
            return traj.get('final_answer', ''), count

    return best_answer, count


# ============================================================
# Method 3: VersaPRM
# ============================================================

def load_versaprm(model_id: str = "UW-Madison-Lee-Lab/VersaPRM-Base-8B"):
    """Load VersaPRM model."""
    print(f"Loading VersaPRM from {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print("✓ VersaPRM loaded")
    return model, tokenizer


def get_versaprm_score(model, tokenizer, question: str, steps: List[Dict], device) -> float:
    """Get trajectory score from VersaPRM."""
    scores = batch_get_versaprm_scores(model, tokenizer, [{'question': question, 'steps': steps}], device)
    return scores[0]


def batch_get_versaprm_scores(
    model, tokenizer, trajectories: List[Dict], device, batch_size: int = 8
) -> List[float]:
    """Batch get VersaPRM scores for multiple trajectories."""
    candidate_tokens = [12, 10]
    step_token_id = 23535

    # Format all inputs
    all_texts = []
    for traj in trajectories:
        question = traj['question']
        steps = traj.get('steps', [])

        step_texts = []
        for i, step in enumerate(steps):
            content = step.get('content', step.get('text', ''))[:500]
            step_texts.append(f"Step {i+1}: {content}")

        step_separator = ' \n\n\n\n'
        input_text = f"Question: {question} \n\n" + step_separator.join(step_texts) + step_separator
        all_texts.append(input_text)

    # Batch process
    all_scores = []
    num_batches = (len(all_texts) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(all_texts), batch_size), total=num_batches, desc="VersaPRM eval"):
        batch_texts = all_texts[i:i+batch_size]

        # Tokenize with padding
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(device)

        with torch.no_grad():
            logits = model(inputs['input_ids'], attention_mask=inputs['attention_mask']).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 1]

            # Extract scores for each item in batch
            for j in range(len(batch_texts)):
                input_ids = inputs['input_ids'][j]
                item_scores = scores[j]

                step_mask = (input_ids == step_token_id)
                if step_mask.sum() > 0:
                    step_scores = item_scores[step_mask].tolist()
                    traj_score = min(step_scores) if step_scores else 0.0
                else:
                    # Fallback: use last non-padding token
                    valid_len = inputs['attention_mask'][j].sum().item()
                    traj_score = item_scores[valid_len - 1].item()

                all_scores.append(traj_score)

    return all_scores


# ============================================================
# Main Comparison
# ============================================================

def group_trajectories_by_question(trajectories: List[Dict]) -> Dict[str, List[Dict]]:
    """Group trajectories by question."""
    groups = {}
    for traj in trajectories:
        # Extract question_id from trajectory_id (format: "{qid}_sample_{n}")
        traj_id = traj.get('trajectory_id', '')
        if '_sample_' in traj_id:
            qid = traj_id.rsplit('_sample_', 1)[0]
        else:
            qid = traj_id or traj.get('question', '')[:50]

        if qid not in groups:
            groups[qid] = []
        groups[qid].append(traj)

    return groups


def check_answer(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold."""
    if not predicted or not gold:
        return False
    pred_norm = predicted.strip().lower()
    gold_norm = gold.strip().lower()
    return pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm


def main():
    parser = argparse.ArgumentParser(description="Compare voting methods")
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to trajectories JSONL file")
    parser.add_argument("--critic-model", type=str, default="outputs/critic_model_debug/final_model",
                        help="Path to our critic model")
    parser.add_argument("--critic-base", type=str, default="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                        help="Base model for critic")
    parser.add_argument("--versaprm", type=str, default="UW-Madison-Lee-Lab/VersaPRM-Base-8B",
                        help="VersaPRM model ID")
    parser.add_argument("--output", type=str, default="outputs/voting_comparison.json",
                        help="Output path for results")
    parser.add_argument("--skip-critic", action="store_true",
                        help="Skip critic evaluation (for testing)")
    parser.add_argument("--skip-versaprm", action="store_true",
                        help="Skip VersaPRM evaluation (for testing)")
    parser.add_argument("--gpu-memory", type=float, default=0.4,
                        help="GPU memory utilization for vLLM (default: 0.4)")
    args = parser.parse_args()

    # Load trajectories
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = []
    with open(args.trajectories) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"✓ Loaded {len(trajectories)} trajectories")

    # Group by question
    groups = group_trajectories_by_question(trajectories)
    print(f"✓ {len(groups)} unique questions")

    # Load models
    critic_llm, critic_tokenizer, critic_lora_request = None, None, None
    versaprm_model, versaprm_tokenizer, versaprm_device = None, None, None

    if not args.skip_critic:
        critic_llm, critic_tokenizer, critic_lora_request = load_critic_model_vllm(
            args.critic_model, args.critic_base, gpu_memory_utilization=args.gpu_memory
        )

    # VersaPRM will be loaded later (after critic is unloaded) to save GPU memory

    # Results
    results = {
        'majority_voting': {'correct': 0, 'total': 0},
        'our_critic': {'correct': 0, 'total': 0},
        'versaprm': {'correct': 0, 'total': 0},
        'details': [],
    }

    print(f"\n{'='*70}")
    print("COMPARING VOTING METHODS")
    print(f"{'='*70}\n")

    # Flatten all trajectories with question info for batch processing
    all_trajs_flat = []
    qid_list = list(groups.keys())
    for qid in qid_list:
        for traj in groups[qid]:
            all_trajs_flat.append(traj)

    # Method 1: Majority Voting (fast, no batching needed)
    print("Computing Majority Voting...")
    majority_results = {}
    for qid, trajs in groups.items():
        majority_answer, vote_count = majority_vote(trajs)
        gold_answer = trajs[0].get('gold_answer')
        majority_correct = check_answer(majority_answer, gold_answer)
        majority_results[qid] = {
            'answer': majority_answer,
            'votes': vote_count,
            'correct': majority_correct,
        }
        results['majority_voting']['total'] += 1
        if majority_correct:
            results['majority_voting']['correct'] += 1

    # Method 2: Our Critic (vLLM batch processing)
    critic_results = {}
    if critic_llm:
        print(f"Evaluating with Critic (vLLM)... {len(all_trajs_flat)} trajectories")
        critic_scores = batch_evaluate_with_critic_vllm(
            critic_llm, critic_tokenizer, critic_lora_request, all_trajs_flat
        )

        # Map scores back to trajectories and find best per question
        score_idx = 0
        for qid in qid_list:
            trajs = groups[qid]
            best_score = -1
            best_traj = trajs[0]
            for traj in trajs:
                _, score = critic_scores[score_idx]
                if score > best_score:
                    best_score = score
                    best_traj = traj
                score_idx += 1

            gold_answer = trajs[0].get('gold_answer')
            critic_answer = best_traj.get('final_answer', '')
            critic_correct = check_answer(critic_answer, gold_answer)
            critic_results[qid] = {
                'answer': critic_answer,
                'score': best_score,
                'correct': critic_correct,
            }
            results['our_critic']['total'] += 1
            if critic_correct:
                results['our_critic']['correct'] += 1

    # Free up GPU memory before loading VersaPRM
    if critic_llm and not args.skip_versaprm:
        print("Unloading Critic model to free GPU memory...")
        del critic_llm
        torch.cuda.empty_cache()

    # Load VersaPRM now (after critic is done)
    if not args.skip_versaprm and versaprm_model is None:
        versaprm_model, versaprm_tokenizer = load_versaprm(args.versaprm)
        versaprm_device = next(versaprm_model.parameters()).device

    # Method 3: VersaPRM (batch processing)
    versaprm_results = {}
    if versaprm_model:
        print(f"Evaluating with VersaPRM (batch)... {len(all_trajs_flat)} trajectories")
        versaprm_scores = batch_get_versaprm_scores(
            versaprm_model, versaprm_tokenizer, all_trajs_flat, versaprm_device, batch_size=8
        )

        # Map scores back to trajectories and find best per question
        score_idx = 0
        for qid in qid_list:
            trajs = groups[qid]
            best_score = -1
            best_traj = trajs[0]
            for traj in trajs:
                score = versaprm_scores[score_idx]
                if score > best_score:
                    best_score = score
                    best_traj = traj
                score_idx += 1

            gold_answer = trajs[0].get('gold_answer')
            versaprm_answer = best_traj.get('final_answer', '')
            versaprm_correct = check_answer(versaprm_answer, gold_answer)
            versaprm_results[qid] = {
                'answer': versaprm_answer,
                'score': best_score,
                'correct': versaprm_correct,
            }
            results['versaprm']['total'] += 1
            if versaprm_correct:
                results['versaprm']['correct'] += 1

    # Build details
    for qid in qid_list:
        trajs = groups[qid]
        detail = {
            'question_id': qid,
            'question': trajs[0]['question'][:100],
            'gold_answer': trajs[0].get('gold_answer'),
            'num_trajectories': len(trajs),
            'majority': majority_results.get(qid, {}),
        }
        if critic_results:
            detail['critic'] = critic_results.get(qid, {})
        if versaprm_results:
            detail['versaprm'] = versaprm_results.get(qid, {})
        results['details'].append(detail)

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    for method in ['majority_voting', 'our_critic', 'versaprm']:
        data = results[method]
        if data['total'] > 0:
            acc = 100 * data['correct'] / data['total']
            print(f"{method:20s}: {data['correct']}/{data['total']} ({acc:.1f}%)")

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
