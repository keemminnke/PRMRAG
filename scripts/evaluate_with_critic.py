#!/usr/bin/env python3
"""Evaluate generated trajectories with trained critic model."""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_critic_model(critic_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load trained critic model."""
    print(f"Loading critic model from {critic_path}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, critic_path)
    model.eval()

    print("✓ Critic model loaded")
    return model, tokenizer


def parse_step_content(step: dict) -> dict:
    """Parse step content to extract thought, action, observation."""
    # If already has thought/action fields, use them
    if step.get('thought') or step.get('action'):
        return {
            'thought': step.get('thought', ''),
            'action': step.get('action', 'Unknown'),
            'action_input': step.get('action_input', ''),
            'observation': step.get('observation', ''),
        }

    # Parse from content field
    content = step.get('content', step.get('text', ''))

    thought = ''
    action = 'Unknown'
    action_input = ''
    observation = ''

    # Extract Thought
    if 'Thought:' in content:
        thought_start = content.find('Thought:') + len('Thought:')
        thought_end = content.find('Action:') if 'Action:' in content else len(content)
        thought = content[thought_start:thought_end].strip()

    # Extract Action
    if 'Action:' in content:
        action_start = content.find('Action:') + len('Action:')
        action_end = content.find('Observation:') if 'Observation:' in content else len(content)
        action_text = content[action_start:action_end].strip()

        # Parse action type and input (e.g., "Search[query]")
        if '[' in action_text and ']' in action_text:
            bracket_start = action_text.find('[')
            action = action_text[:bracket_start].strip()
            action_input = action_text[bracket_start+1:action_text.rfind(']')].strip()
        else:
            action = action_text.split('\n')[0].strip()

    # Extract Observation
    if 'Observation:' in content:
        obs_start = content.find('Observation:') + len('Observation:')
        observation = content[obs_start:].strip()[:1000]  # Truncate

    return {
        'thought': thought,
        'action': action,
        'action_input': action_input,
        'observation': observation,
    }


def format_step_for_critic(question: str, previous_steps: list, current_step: dict) -> str:
    """Format step for critic evaluation."""
    input_parts = [f"Question: {question}", ""]

    if previous_steps:
        input_parts.append("Previous Steps:")
        for i, prev in enumerate(previous_steps, 1):
            parsed = parse_step_content(prev)
            input_parts.append(f"Step {i}:")
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
    parsed = parse_step_content(current_step)
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
    input_parts.append("You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First, provide a reasoning explanation, and then output the final label (GOOD/BAD)."
        )

    return "\n".join(input_parts)


def evaluate_step(model, tokenizer, question: str, previous_steps: list, current_step: dict) -> dict:
    """Evaluate a single step with critic."""
    user_content = format_step_for_critic(question, previous_steps, current_step)
    system_content = (
        "You are a step-level critic for evaluating reasoning quality in multi-hop question answering. "
        "Analyze each step's logical soundness and evidence grounding. "
        "First, provide a reasoning explanation, and then output the final label (GOOD/BAD)."
        )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse label
    label = "UNKNOWN"
    if "Label: GOOD" in response or "Label:GOOD" in response:
        label = "GOOD"
    elif "Label: BAD" in response or "Label:BAD" in response:
        label = "BAD"

    return {"label": label, "reasoning": response}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, default="outputs/test_trajectories_10.jsonl")
    parser.add_argument("--critic-model", type=str, default="outputs/critic_model_full/final_model")
    parser.add_argument("--output", type=str, default="outputs/critic_evaluation_results.jsonl")
    args = parser.parse_args()

    # Load trajectories
    print(f"Loading trajectories from {args.trajectories}...")
    trajectories = []
    with open(args.trajectories) as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"✓ Loaded {len(trajectories)} trajectories")

    # Load critic
    model, tokenizer = load_critic_model(args.critic_model)

    # Evaluate
    print(f"\n{'='*70}")
    print("EVALUATING TRAJECTORIES")
    print(f"{'='*70}\n")

    results = []
    for traj in trajectories:
        if not traj.get('steps'):
            print(f"Skipping {traj.get('question_id', 'unknown')} - no steps")
            continue

        print(f"\nQ: {traj['question'][:60]}...")
        print(f"   Gold: {traj.get('gold_answer', 'N/A')}")
        print(f"   Pred: {str(traj.get('final_answer', 'N/A'))[:50]}")
        print(f"   Correct: {traj.get('is_correct')}")

        step_evals = []
        steps = traj['steps']

        for i, step in enumerate(steps):
            prev_steps = steps[:i]
            parsed = parse_step_content(step)
            result = evaluate_step(model, tokenizer, traj['question'], prev_steps, step)

            step_evals.append({
                'step_num': i + 1,
                'thought': parsed['thought'][:300] if parsed['thought'] else '',
                'action': parsed['action'],
                'action_input': parsed['action_input'][:100] if parsed['action_input'] else '',
                'observation': parsed['observation'][:300] if parsed['observation'] else '',
                'critic_label': result['label'],
                'critic_reasoning': result['reasoning'],
            })

            print(f"   Step {i+1} ({parsed['action'][:10]}): {result['label']}")

        results.append({
            'question_id': traj.get('question_id'),
            'question': traj['question'],
            'gold_answer': traj.get('gold_answer'),
            'final_answer': traj.get('final_answer'),
            'is_correct': traj.get('is_correct'),
            'step_evaluations': step_evals,
        })

    # Save
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total_steps = sum(len(r['step_evaluations']) for r in results)
    good = sum(1 for r in results for s in r['step_evaluations'] if s['critic_label'] == 'GOOD')
    bad = sum(1 for r in results for s in r['step_evaluations'] if s['critic_label'] == 'BAD')

    print(f"Trajectories: {len(results)}")
    print(f"Total steps: {total_steps}")
    print(f"GOOD: {good} ({100*good/total_steps:.1f}%)")
    print(f"BAD: {bad} ({100*bad/total_steps:.1f}%)")
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
