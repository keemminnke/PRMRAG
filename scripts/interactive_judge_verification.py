#!/usr/bin/env python3
"""Interactive LLM-as-Judge verification tool.

This script allows you to:
1. View Step 1 trajectories one by one
2. Run LLM Judge on each step
3. Compare RPE labels vs Judge labels
4. Track agreement/disagreement statistics
5. Manually review and flag problematic cases

Usage:
    python scripts/interactive_judge_verification.py \
        --input outputs/hybrid_100q_from_11/results_hybrid_20251205_061918.jsonl \
        --start-idx 0 \
        --num-samples 10
"""

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.utils.answer_utils import check_answer_match, normalize_answer


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def color(text: str, color_code: str) -> str:
    """Apply color to text."""
    return f"{color_code}{text}{Colors.ENDC}"


def print_header(text: str):
    """Print a section header."""
    print(f"\n{color('='*70, Colors.BLUE)}")
    print(color(text, Colors.BOLD + Colors.BLUE))
    print(color('='*70, Colors.BLUE))


def print_step(step: Dict[str, Any], step_num: int):
    """Print a single step with formatting."""
    action = step.get('action', 'Reason')
    label = step.get('label', 'N/A')
    rpe = step.get('rpe', 0)
    mc_before = step.get('mc_before', 0)
    mc_after = step.get('mc_after', 0)
    content = step.get('content', '')

    # Color based on label
    if label.lower() == 'good':
        label_color = Colors.GREEN
    elif label.lower() == 'bad':
        label_color = Colors.RED
    else:
        label_color = Colors.YELLOW

    # Action color
    if action == 'Search':
        action_color = Colors.CYAN
    else:
        action_color = Colors.YELLOW

    print(f"\n{color(f'--- Step {step_num} ---', Colors.BOLD)}")
    print(f"Action: {color(action, action_color)}")
    print(f"RPE Label: {color(label.upper(), label_color)} (RPE: {rpe:.3f})")
    print(f"MC: {mc_before:.3f} → {mc_after:.3f} ({mc_after - mc_before:+.3f})")

    # Print content (truncated)
    print(f"\n{color('Content:', Colors.UNDERLINE)}")

    # Pretty print ReAct format
    lines = content.split('\n')
    for line in lines[:15]:  # Limit lines
        if line.strip().startswith('Thought:'):
            print(f"  {color('Thought:', Colors.CYAN)} {line.split('Thought:', 1)[1].strip()[:100]}")
        elif line.strip().startswith('Action:'):
            print(f"  {color('Action:', Colors.YELLOW)} {line.split('Action:', 1)[1].strip()[:100]}")
        elif line.strip().startswith('Observation:'):
            print(f"  {color('Observation:', Colors.GREEN)} {line.split('Observation:', 1)[1].strip()[:150]}")
        elif line.strip().startswith('Sub-answer:'):
            print(f"  {color('Sub-answer:', Colors.BLUE)} {line.split('Sub-answer:', 1)[1].strip()[:100]}")
        elif line.strip():
            print(f"  {line.strip()[:120]}")

    if len(lines) > 15:
        print(f"  ... ({len(lines) - 15} more lines)")


def build_judge_prompt(question: str, gold_answer: str, step: Dict[str, Any],
                       prev_steps: List[Dict[str, Any]] = None) -> str:
    """Build a judge prompt for a step."""

    prompt_parts = [
        "You are an expert evaluator for question-answering systems.",
        "",
        "# Task",
        "Evaluate whether this reasoning step is GOOD or BAD for answering the question correctly.",
        "",
        "# Evaluation Criteria",
        "A step is GOOD if:",
        "- It makes correct logical inferences based on the question",
        "- It retrieves relevant information (for Search steps)",
        "- It moves toward the correct answer",
        "- The reasoning is sound and well-structured",
        "",
        "A step is BAD if:",
        "- It makes incorrect or illogical inferences",
        "- It retrieves irrelevant or misleading information",
        "- It leads away from the correct answer",
        "- The reasoning contains factual errors",
        "",
        f"# Question",
        f"{question}",
        "",
        f"# Correct Answer",
        f"{gold_answer}",
    ]

    # Add previous steps context
    if prev_steps:
        prompt_parts.extend([
            "",
            "# Previous Steps",
        ])
        for i, ps in enumerate(prev_steps[-3:], 1):  # Last 3 steps
            prompt_parts.append(f"Step {i}: {ps.get('content', '')[:200]}...")

    # Add current step
    prompt_parts.extend([
        "",
        "# Step to Evaluate",
        f"{step.get('content', '')}",
        "",
        "# Your Evaluation",
        "Provide your evaluation in this exact format:",
        "",
        "Reasoning: [Your detailed explanation of why this step is good or bad]",
        "Label: [GOOD or BAD]",
        "Confidence: [0.0 to 1.0]",
    ])

    return "\n".join(prompt_parts)


def run_judge_on_step(model, question: str, gold_answer: str,
                      step: Dict[str, Any], prev_steps: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run LLM judge on a single step."""

    prompt = build_judge_prompt(question, gold_answer, step, prev_steps)

    response = model.generate_with_chat_template(
        user_message=prompt,
        max_tokens=512,
        temperature=0.3,
    )

    # Parse response
    label = "GOOD"
    reasoning = ""
    confidence = 0.5

    for line in response.strip().split('\n'):
        line = line.strip()
        if line.startswith('Reasoning:'):
            reasoning = line.split('Reasoning:', 1)[1].strip()
        elif line.startswith('Label:'):
            label_text = line.split('Label:', 1)[1].strip().upper()
            if 'BAD' in label_text:
                label = 'BAD'
            elif 'GOOD' in label_text:
                label = 'GOOD'
        elif line.startswith('Confidence:'):
            try:
                confidence = float(line.split('Confidence:', 1)[1].strip())
            except:
                confidence = 0.5

    return {
        'label': label,
        'reasoning': reasoning,
        'confidence': confidence,
        'raw_response': response,
    }


def interactive_review(results: List[Dict], start_idx: int = 0,
                       num_samples: int = 10, use_judge: bool = True):
    """Interactive review of trajectories with optional LLM judge."""

    # Initialize model if using judge
    model = None
    if use_judge:
        print(color("\nInitializing LLM Judge model...", Colors.YELLOW))
        from prmrag.models import load_policy_model
        model_config = {
            'model_name': 'Qwen/Qwen2.5-7B-Instruct',
            'temperature': 0.3,
            'max_tokens': 512,
            'gpu_memory_utilization': 0.5,
        }
        model = load_policy_model(model_config)
        print(color("Model loaded!", Colors.GREEN))

    # Statistics
    stats = {
        'total_steps': 0,
        'rpe_good': 0,
        'rpe_bad': 0,
        'judge_good': 0,
        'judge_bad': 0,
        'agreement': 0,
        'disagreement': 0,
        'positive_consensus': 0,  # Both GOOD
        'negative_consensus': 0,  # Both BAD
        'rpe_good_judge_bad': 0,  # RPE says GOOD, Judge says BAD
        'rpe_bad_judge_good': 0,  # RPE says BAD, Judge says GOOD
    }

    # Review log
    review_log = []

    # Process samples
    end_idx = min(start_idx + num_samples, len(results))

    for q_idx in range(start_idx, end_idx):
        result = results[q_idx]

        print_header(f"Question {q_idx + 1}/{len(results)}")
        print(f"\n{color('Question:', Colors.BOLD)} {result['question']}")
        print(f"{color('Gold Answer:', Colors.GREEN)} {result['gold_answer']}")
        print(f"{color('Predicted:', Colors.YELLOW)} {result['predicted_answer'][:100]}...")

        # Check correctness with new metric
        is_correct = check_answer_match(result['predicted_answer'], result['gold_answer'])
        old_correct = result['is_correct']

        if is_correct:
            print(f"{color('Correct:', Colors.GREEN)} ✅ YES")
        else:
            print(f"{color('Correct:', Colors.RED)} ❌ NO")

        if is_correct != old_correct:
            print(f"{color('(Changed from old evaluation)', Colors.YELLOW)}")

        num_steps = len(result['steps'])
        print(f"\n{color(f'Total Steps: {num_steps}', Colors.CYAN)}")

        # Process each step
        prev_steps = []
        step_reviews = []

        for step_idx, step in enumerate(result['steps']):
            stats['total_steps'] += 1

            rpe_label = step.get('label', 'good').upper()
            if rpe_label == 'GOOD':
                stats['rpe_good'] += 1
            else:
                stats['rpe_bad'] += 1

            print_step(step, step_idx + 1)

            judge_result = None
            if use_judge and model:
                print(f"\n{color('Running Judge...', Colors.YELLOW)}", end=" ", flush=True)

                judge_result = run_judge_on_step(
                    model,
                    result['question'],
                    result['gold_answer'],
                    step,
                    prev_steps
                )

                judge_label = judge_result['label']

                if judge_label == 'GOOD':
                    stats['judge_good'] += 1
                    label_color = Colors.GREEN
                else:
                    stats['judge_bad'] += 1
                    label_color = Colors.RED

                print(color(f"Judge: {judge_label}", label_color))
                print(f"  Reasoning: {judge_result['reasoning'][:150]}...")
                print(f"  Confidence: {judge_result['confidence']:.2f}")

                # Compare with RPE
                if rpe_label == judge_label:
                    stats['agreement'] += 1
                    if rpe_label == 'GOOD':
                        stats['positive_consensus'] += 1
                        print(color("  ✅ CONSENSUS: Both GOOD", Colors.GREEN))
                    else:
                        stats['negative_consensus'] += 1
                        print(color("  ✅ CONSENSUS: Both BAD", Colors.GREEN))
                else:
                    stats['disagreement'] += 1
                    if rpe_label == 'GOOD':
                        stats['rpe_good_judge_bad'] += 1
                        print(color("  ⚠️  CONFLICT: RPE=GOOD, Judge=BAD", Colors.RED))
                    else:
                        stats['rpe_bad_judge_good'] += 1
                        print(color("  ⚠️  CONFLICT: RPE=BAD, Judge=GOOD", Colors.YELLOW))

                step_reviews.append({
                    'step_idx': step_idx,
                    'rpe_label': rpe_label,
                    'judge_label': judge_label,
                    'judge_reasoning': judge_result['reasoning'],
                    'judge_confidence': judge_result['confidence'],
                    'consensus': rpe_label == judge_label,
                })

            prev_steps.append(step)

        # Summary for this question
        print(f"\n{color('--- Question Summary ---', Colors.BOLD)}")
        consensus_count = sum(1 for r in step_reviews if r.get('consensus', False))
        total_reviewed = len(step_reviews)
        if total_reviewed > 0:
            print(f"Consensus: {consensus_count}/{total_reviewed} steps ({consensus_count/total_reviewed*100:.1f}%)")

        # Log this question
        review_log.append({
            'question_id': result.get('question_id', q_idx),
            'question': result['question'],
            'gold_answer': result['gold_answer'],
            'is_correct': is_correct,
            'num_steps': len(result['steps']),
            'step_reviews': step_reviews,
        })

        # Interactive prompt
        print(f"\n{color('Options:', Colors.CYAN)}")
        print("  [Enter] Continue to next question")
        print("  [s] Skip to summary")
        print("  [q] Quit and save")

        user_input = input("\nChoice: ").strip().lower()

        if user_input == 'q':
            break
        elif user_input == 's':
            break

    # Print final statistics
    print_header("FINAL STATISTICS")

    print(f"\n{color('Overall:', Colors.BOLD)}")
    print(f"  Total steps reviewed: {stats['total_steps']}")

    print(f"\n{color('RPE Labels:', Colors.BOLD)}")
    print(f"  GOOD: {stats['rpe_good']} ({stats['rpe_good']/max(stats['total_steps'],1)*100:.1f}%)")
    print(f"  BAD:  {stats['rpe_bad']} ({stats['rpe_bad']/max(stats['total_steps'],1)*100:.1f}%)")

    if use_judge:
        print(f"\n{color('Judge Labels:', Colors.BOLD)}")
        print(f"  GOOD: {stats['judge_good']} ({stats['judge_good']/max(stats['total_steps'],1)*100:.1f}%)")
        print(f"  BAD:  {stats['judge_bad']} ({stats['judge_bad']/max(stats['total_steps'],1)*100:.1f}%)")

        print(f"\n{color('Consensus Analysis:', Colors.BOLD)}")
        print(f"  Agreement:    {stats['agreement']} ({stats['agreement']/max(stats['total_steps'],1)*100:.1f}%)")
        print(f"  Disagreement: {stats['disagreement']} ({stats['disagreement']/max(stats['total_steps'],1)*100:.1f}%)")

        print(f"\n{color('Consensus Breakdown:', Colors.BOLD)}")
        print(f"  Both GOOD (keep, label=1):     {stats['positive_consensus']}")
        print(f"  Both BAD (keep, label=0):      {stats['negative_consensus']}")
        print(f"  RPE=GOOD, Judge=BAD (discard): {stats['rpe_good_judge_bad']}")
        print(f"  RPE=BAD, Judge=GOOD (discard): {stats['rpe_bad_judge_good']}")

        kept = stats['positive_consensus'] + stats['negative_consensus']
        discarded = stats['disagreement']
        print(f"\n{color('Data Retention:', Colors.BOLD)}")
        print(f"  Kept:      {kept}/{stats['total_steps']} ({kept/max(stats['total_steps'],1)*100:.1f}%)")
        print(f"  Discarded: {discarded}/{stats['total_steps']} ({discarded/max(stats['total_steps'],1)*100:.1f}%)")

    # Save log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(f"outputs/judge_review_log_{timestamp}.json")
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with open(log_file, 'w') as f:
        json.dump({
            'stats': stats,
            'reviews': review_log,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n{color(f'Review log saved to: {log_file}', Colors.GREEN)}")

    return stats, review_log


def main():
    parser = argparse.ArgumentParser(description="Interactive LLM-as-Judge verification")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/hybrid_100q_from_11/results_hybrid_20251205_061918.jsonl",
        help="Input results file",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting index",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to review",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge (just view data)",
    )
    args = parser.parse_args()

    # Load results
    print(color(f"\nLoading results from {args.input}...", Colors.CYAN))

    results = []
    with open(args.input, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(color(f"Loaded {len(results)} trajectories", Colors.GREEN))

    # Run interactive review
    interactive_review(
        results,
        start_idx=args.start_idx,
        num_samples=args.num_samples,
        use_judge=not args.no_judge,
    )


if __name__ == "__main__":
    main()
