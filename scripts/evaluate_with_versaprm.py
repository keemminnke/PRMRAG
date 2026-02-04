#!/usr/bin/env python3
"""Evaluate trajectories with VersaPRM (Process Reward Model).

VersaPRM: UW-Madison-Lee-Lab/VersaPRM-Base-8B
- Based on Llama-3.1-8B-Instruct
- Provides step-level reward scores
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def get_tokenizer(model_id: str):
    """Load tokenizer with VersaPRM settings."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    return tokenizer


def load_versaprm(model_id: str = "UW-Madison-Lee-Lab/VersaPRM-Base-8B", device: str = "cuda"):
    """Load VersaPRM model."""
    print(f"Loading VersaPRM from {model_id}...")

    tokenizer = get_tokenizer(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print("✓ VersaPRM loaded")
    return model, tokenizer


def parse_step_content(step: dict) -> str:
    """Parse step content to text format."""
    # If already has thought/action fields
    if step.get('thought') or step.get('action'):
        parts = []
        if step.get('thought'):
            parts.append(f"Thought: {step['thought']}")
        if step.get('action'):
            action_input = step.get('action_input', '')
            if action_input:
                parts.append(f"Action: {step['action']}[{action_input}]")
            else:
                parts.append(f"Action: {step['action']}")
        if step.get('observation'):
            parts.append(f"Observation: {step['observation'][:500]}")
        return ' '.join(parts)

    # Parse from content field
    content = step.get('content', step.get('text', ''))
    return content[:1000]  # Truncate long content


def format_trajectory_for_versaprm(question: str, steps: List[Dict]) -> str:
    """Format trajectory for VersaPRM scoring.

    VersaPRM expects:
    - Question + steps separated by ' \\n\\n\\n\\n'
    """
    # Format question
    question_text = f"Question: {question}"

    # Format steps
    step_texts = []
    for i, step in enumerate(steps):
        step_content = parse_step_content(step)
        step_texts.append(f"Step {i+1}: {step_content}")

    # Join with VersaPRM separator
    step_separator = ' \n\n\n\n'
    solution_text = step_separator.join(step_texts) + step_separator

    return question_text + ' \n\n' + solution_text


def get_step_scores(
    model,
    tokenizer,
    question: str,
    steps: List[Dict],
    device: str = "cuda"
) -> List[float]:
    """Get per-step scores from VersaPRM.

    Returns:
        List of scores (one per step), higher = better
    """
    # VersaPRM uses these token IDs for scoring
    candidate_tokens = [12, 10]  # From model card
    step_token_id = 23535  # Token ID for step separator

    # Format input
    input_text = format_trajectory_for_versaprm(question, steps)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(input_ids).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 1]  # Probability of "good" token

        # Extract scores at step separator positions
        step_mask = (input_ids == step_token_id)
        if step_mask.sum() > 0:
            step_scores = scores[step_mask].tolist()
        else:
            # Fallback: use last token score for each step
            step_scores = [scores[0, -1].item()] * len(steps)

    return step_scores


def aggregate_trajectory_score(step_scores: List[float], method: str = "min") -> float:
    """Aggregate step scores into trajectory score.

    Args:
        step_scores: Per-step scores
        method: Aggregation method
            - "min": Minimum score (conservative, PRM-style)
            - "mean": Average score
            - "prod": Product of scores
            - "last": Last step score
    """
    if not step_scores:
        return 0.0

    if method == "min":
        return min(step_scores)
    elif method == "mean":
        return sum(step_scores) / len(step_scores)
    elif method == "prod":
        score = 1.0
        for s in step_scores:
            score *= s
        return score
    elif method == "last":
        return step_scores[-1]
    else:
        return min(step_scores)


def evaluate_trajectories(
    model,
    tokenizer,
    trajectories: List[Dict],
    device: str = "cuda",
    show_progress: bool = True,
) -> List[Dict]:
    """Evaluate all trajectories with VersaPRM."""
    results = []

    iterator = tqdm(trajectories, desc="Evaluating") if show_progress else trajectories

    for traj in iterator:
        question = traj['question']
        steps = traj.get('steps', [])

        if not steps:
            results.append({
                'trajectory_id': traj.get('trajectory_id'),
                'question': question,
                'step_scores': [],
                'trajectory_score': 0.0,
                'final_answer': traj.get('final_answer'),
            })
            continue

        # Get step scores
        step_scores = get_step_scores(model, tokenizer, question, steps, device)

        # Aggregate to trajectory score
        traj_score = aggregate_trajectory_score(step_scores, method="min")

        results.append({
            'trajectory_id': traj.get('trajectory_id'),
            'question': question,
            'gold_answer': traj.get('gold_answer'),
            'final_answer': traj.get('final_answer'),
            'is_correct': traj.get('is_correct'),
            'step_scores': step_scores,
            'trajectory_score': traj_score,
            'num_steps': len(steps),
        })

    return results


def select_best_trajectory(
    trajectories: List[Dict],
    scores: List[Dict],
    method: str = "best_score"
) -> Dict:
    """Select best trajectory based on scores.

    Args:
        trajectories: List of trajectory dicts
        scores: List of score dicts (from evaluate_trajectories)
        method: Selection method
            - "best_score": Highest trajectory score
            - "majority_vote": Most common answer (weighted by score)
    """
    if method == "best_score":
        best_idx = max(range(len(scores)), key=lambda i: scores[i]['trajectory_score'])
        return trajectories[best_idx], scores[best_idx]

    elif method == "majority_vote":
        # Weight votes by trajectory score
        answer_scores = {}
        for traj, score in zip(trajectories, scores):
            answer = traj.get('final_answer', '')
            if answer:
                answer_lower = answer.strip().lower()
                if answer_lower not in answer_scores:
                    answer_scores[answer_lower] = {'score': 0, 'count': 0, 'original': answer}
                answer_scores[answer_lower]['score'] += score['trajectory_score']
                answer_scores[answer_lower]['count'] += 1

        if not answer_scores:
            return trajectories[0], scores[0]

        best_answer = max(answer_scores.items(), key=lambda x: x[1]['score'])
        # Find first trajectory with this answer
        for traj, score in zip(trajectories, scores):
            if traj.get('final_answer', '').strip().lower() == best_answer[0]:
                return traj, score

        return trajectories[0], scores[0]

    return trajectories[0], scores[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trajectories with VersaPRM")
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to trajectories JSONL file")
    parser.add_argument("--output", type=str, default="outputs/versaprm_evaluation.jsonl",
                        help="Output path for evaluation results")
    parser.add_argument("--model", type=str, default="UW-Madison-Lee-Lab/VersaPRM-Base-8B",
                        help="VersaPRM model ID")
    parser.add_argument("--aggregate", type=str, default="min",
                        choices=["min", "mean", "prod", "last"],
                        help="Score aggregation method")
    args = parser.parse_args()

    # Load trajectories
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = []
    with open(args.trajectories) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    print(f"✓ Loaded {len(trajectories)} trajectories")

    # Load model
    model, tokenizer = load_versaprm(args.model)
    device = next(model.parameters()).device

    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATING WITH VERSAPRM")
    print(f"{'='*70}\n")

    results = evaluate_trajectories(model, tokenizer, trajectories, str(device))

    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    avg_score = sum(r['trajectory_score'] for r in results) / total if total > 0 else 0
    correct = sum(1 for r in results if r.get('is_correct'))

    print(f"Total trajectories: {total}")
    print(f"Average trajectory score: {avg_score:.4f}")
    print(f"Correct answers: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
