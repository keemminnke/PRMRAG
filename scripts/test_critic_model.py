#!/usr/bin/env python3
"""Test critic model on trajectories.

Usage:
    # Test on existing training data (has ground truth labels)
    python scripts/test_critic_model.py --mode eval --num-samples 20

    # Test on new trajectories (generate + evaluate)
    python scripts/test_critic_model.py --mode generate --num-questions 5
"""

import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_critic_model(critic_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load trained critic model with LoRA."""
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


def format_step_for_critic(question: str, previous_steps: list, current_step: dict) -> str:
    """Format step for critic evaluation."""
    input_parts = []
    input_parts.append(f"Question: {question}")
    input_parts.append("")

    if previous_steps:
        input_parts.append("Previous Steps:")
        for i, prev_step in enumerate(previous_steps, 1):
            input_parts.append(f"Step {i}:")
            thought = prev_step.get('thought')
            if thought:
                input_parts.append(f"Thought: {thought}")

            action = prev_step.get('action', 'Unknown')
            action_input = prev_step.get('action_input', '')
            if action_input:
                input_parts.append(f"Action: {action}[{action_input}]")
            else:
                input_parts.append(f"Action: {action}")

            obs = prev_step.get('observation')
            if obs:
                input_parts.append(f"Observation: {obs}")
            input_parts.append("")

    input_parts.append("Current Step to Evaluate:")
    thought = current_step.get('thought')
    if thought:
        input_parts.append(f"Thought: {thought}")

    action = current_step.get('action', 'Unknown')
    action_input = current_step.get('action_input', '')
    if action_input:
        input_parts.append(f"Action: {action}[{action_input}]")
    else:
        input_parts.append(f"Action: {action}")

    obs = current_step.get('observation')
    if obs:
        input_parts.append(f"Observation: {obs}")

    input_parts.append("")
    input_parts.append("Task: Evaluate the quality of the Current Step. Analyze whether the reasoning is logically sound and grounded in evidence, then provide a label (GOOD/BAD).")

    return "\n".join(input_parts)


def evaluate_step_with_critic(model, tokenizer, question: str, previous_steps: list, current_step: dict) -> dict:
    """Run critic model on a single step."""
    user_content = format_step_for_critic(question, previous_steps, current_step)

    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Parse label
    label = "UNKNOWN"
    if "Label: GOOD" in response or "Label:GOOD" in response or response.strip().endswith("GOOD"):
        label = "GOOD"
    elif "Label: BAD" in response or "Label:BAD" in response or response.strip().endswith("BAD"):
        label = "BAD"

    return {"reasoning": response, "label": label}


def test_on_training_data(args):
    """Test critic accuracy on training data samples (with ground truth)."""
    print(f"\n{'='*70}")
    print("TEST CRITIC ON TRAINING DATA (Ground Truth Available)")
    print(f"{'='*70}\n")

    # Load critic model
    critic_model, tokenizer = load_critic_model(args.critic_model)

    # Load training data samples
    data_file = Path("outputs/training_data_merged_fixed.jsonl")
    print(f"Loading samples from {data_file}...")

    samples = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if i >= args.num_samples:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} trajectories\n")

    # Evaluate
    correct = 0
    total = 0
    results = []

    for traj_idx, traj in enumerate(samples):
        question = traj['question']
        steps = traj['steps']

        print(f"\n[{traj_idx+1}/{len(samples)}] {question[:50]}...")

        for step_idx, step in enumerate(steps):
            ground_truth = step.get('judge_label', 'UNKNOWN').upper()
            if ground_truth not in ['GOOD', 'BAD']:
                continue

            previous_steps = steps[:step_idx]
            result = evaluate_step_with_critic(critic_model, tokenizer, question, previous_steps, step)
            predicted = result['label']

            is_correct = (predicted == ground_truth)
            if is_correct:
                correct += 1
            total += 1

            status = "✓" if is_correct else "✗"
            print(f"  Step {step_idx+1}: pred={predicted}, truth={ground_truth} {status}")

            results.append({
                'question': question[:50],
                'step_num': step_idx + 1,
                'action': step.get('action'),
                'ground_truth': ground_truth,
                'predicted': predicted,
                'correct': is_correct,
            })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Total steps evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.1f}%")

    # Breakdown by ground truth
    good_correct = sum(1 for r in results if r['ground_truth'] == 'GOOD' and r['correct'])
    good_total = sum(1 for r in results if r['ground_truth'] == 'GOOD')
    bad_correct = sum(1 for r in results if r['ground_truth'] == 'BAD' and r['correct'])
    bad_total = sum(1 for r in results if r['ground_truth'] == 'BAD')

    print(f"\nGOOD steps: {good_correct}/{good_total} ({100*good_correct/good_total:.1f}% recall)" if good_total > 0 else "")
    print(f"BAD steps: {bad_correct}/{bad_total} ({100*bad_correct/bad_total:.1f}% recall)" if bad_total > 0 else "")

    # Save results
    output_file = Path(args.output)
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nResults saved to: {output_file}")


def test_on_new_trajectories(args):
    """Generate new trajectories and evaluate with critic (no ground truth)."""
    print(f"\n{'='*70}")
    print("TEST CRITIC ON NEW TRAJECTORIES")
    print(f"{'='*70}\n")

    # First generate trajectories
    from prmrag.models import load_policy_model
    from prmrag.utils import load_config

    config = load_config(Path("configs/adaptive_generation.yaml"))

    print("Loading policy model...")
    policy_model = load_policy_model(config['policy_model'])

    # Load questions
    questions_file = Path("data/raw/questions/hotpotqa_validation.jsonl")
    questions = []
    with open(questions_file) as f:
        for i, line in enumerate(f):
            if i >= args.num_questions:
                break
            questions.append(json.loads(line))

    print(f"Generating trajectories for {len(questions)} questions...")

    trajectories = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] {q['question'][:50]}...")

        # Simple CoT generation (no RAG for simplicity)
        prompt = f"""You are a helpful assistant solving multi-hop questions step by step.

Question: {q['question']}

Think step by step. For each step, provide your thought process.
When you have the final answer, output: Finish[your answer]

Begin:"""

        try:
            response = policy_model.generate(prompt, max_tokens=500)

            # Parse into steps (simple parsing)
            steps = []
            lines = response.strip().split('\n')
            current_thought = []

            for line in lines:
                if line.strip().startswith('Finish['):
                    if current_thought:
                        steps.append({
                            'thought': ' '.join(current_thought),
                            'action': 'Finish',
                            'action_input': line.strip()[7:-1] if line.strip().endswith(']') else line.strip()[7:],
                        })
                    break
                elif line.strip():
                    current_thought.append(line.strip())
                    if len(current_thought) >= 2:
                        steps.append({
                            'thought': ' '.join(current_thought),
                            'action': 'Think',
                            'action_input': '',
                        })
                        current_thought = []

            trajectories.append({
                'question_id': q['_id'],
                'question': q['question'],
                'gold_answer': q.get('answer'),
                'steps': steps,
                'raw_response': response,
            })

        except Exception as e:
            print(f"  Error: {e}")
            trajectories.append({
                'question_id': q['_id'],
                'question': q['question'],
                'steps': [],
                'error': str(e),
            })

    # Unload policy model
    del policy_model
    torch.cuda.empty_cache()

    # Save trajectories
    traj_file = Path(args.trajectories_file)
    with open(traj_file, 'w') as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f"\nTrajectories saved to: {traj_file}")

    # Now evaluate with critic
    print(f"\n{'='*70}")
    print("EVALUATING WITH CRITIC")
    print(f"{'='*70}\n")

    critic_model, tokenizer = load_critic_model(args.critic_model)

    results = []
    for traj in trajectories:
        if not traj.get('steps'):
            continue

        print(f"\nQuestion: {traj['question'][:50]}...")
        steps = traj['steps']

        for step_idx, step in enumerate(steps):
            previous_steps = steps[:step_idx]
            result = evaluate_step_with_critic(critic_model, tokenizer, traj['question'], previous_steps, step)
            print(f"  Step {step_idx+1} ({step.get('action', 'N/A')}): {result['label']}")

            results.append({
                'question_id': traj['question_id'],
                'step_num': step_idx + 1,
                'action': step.get('action'),
                'critic_label': result['label'],
                'critic_reasoning': result['reasoning'][:200],
            })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    total = len(results)
    good = sum(1 for r in results if r['critic_label'] == 'GOOD')
    bad = sum(1 for r in results if r['critic_label'] == 'BAD')

    print(f"Total steps: {total}")
    print(f"GOOD: {good} ({100*good/total:.1f}%)" if total > 0 else "")
    print(f"BAD: {bad} ({100*bad/total:.1f}%)" if total > 0 else "")

    # Save
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"\nResults saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["eval", "generate"], default="eval",
                        help="eval: test on training data, generate: test on new trajectories")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of trajectories to sample (eval mode)")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions (generate mode)")
    parser.add_argument("--critic-model", type=str, default="outputs/critic_model_full/final_model")
    parser.add_argument("--output", type=str, default="outputs/critic_test_results.jsonl")
    parser.add_argument("--trajectories-file", type=str, default="outputs/test_trajectories.jsonl")
    args = parser.parse_args()

    if args.mode == "eval":
        test_on_training_data(args)
    else:
        test_on_new_trajectories(args)


if __name__ == "__main__":
    main()
