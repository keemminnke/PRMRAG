#!/usr/bin/env python3
"""Debug MC estimation with real HotpotQA data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.data.loaders import load_hotpotqa_questions
from prmrag.generation.step_forcing import StepParser


def test_single_step_generation(policy_model, step_parser, question: str, answer: str):
    """Test generating a single reasoning step."""
    print("\n" + "=" * 70)
    print("[1] Test Single Step Generation")
    print("=" * 70)

    user_prompt = f"""Question: {question}

Solve this step by step. You MUST respond with EXACTLY ONE step in this format:
"Step 1: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 1.

Step 1:"""

    print(f"\nQuestion: {question}")
    print(f"Gold Answer: {answer}")
    print("\nGenerating Step 1...")

    response = policy_model.generate_with_chat_template(
        user_message=user_prompt,
        max_tokens=200,
        temperature=0.8,
        stop_sequences=None
    )

    # Post-process: Extract first step only
    import re
    next_step_match = re.search(r'\n+Step\s+\d+:', response)
    if next_step_match:
        response = response[:next_step_match.start()]

    # Parse step content
    step_content = step_parser.parse_single_step_response(response, expected_step_num=1)

    print("\nGenerated Step 1:")
    print("-" * 70)
    print(step_content)
    print("-" * 70)

    return step_content


def test_continuation(policy_model, step_parser, question: str, step1: str):
    """Test generating continuation step."""
    print("\n" + "=" * 70)
    print("[2] Test Step Continuation")
    print("=" * 70)

    user_prompt = f"""Question: {question}

Step 1: {step1}

Continue solving. You MUST respond with EXACTLY ONE step in this format:
"Step 2: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 2.

Step 2:"""

    print("\nGenerating Step 2...")

    response = policy_model.generate_with_chat_template(
        user_message=user_prompt,
        max_tokens=200,
        temperature=0.8,
        stop_sequences=None
    )

    # Post-process: Extract first step only
    import re
    next_step_match = re.search(r'\n+Step\s+\d+:', response)
    if next_step_match:
        response = response[:next_step_match.start()]

    # Parse step content
    step_content = step_parser.parse_single_step_response(response, expected_step_num=2)

    print("\nGenerated Step 2:")
    print("-" * 70)
    print(step_content)
    print("-" * 70)

    return step_content


def test_mc_rollouts(policy_model, question: str, answer: str, step1: str, num_rollouts: int = 5):
    """Test MC rollouts from a given step."""
    print("\n" + "=" * 70)
    print(f"[3] Test MC Rollouts ({num_rollouts} rollouts from Step 1)")
    print("=" * 70)

    rollout_prompt = f"""Question: {question}

Step 1: {step1}

Continue solving to reach the final answer. Provide your final answer at the end."""

    print(f"\nRunning {num_rollouts} rollouts...")
    correct_count = 0

    for i in range(num_rollouts):
        print(f"\n--- Rollout {i+1}/{num_rollouts} ---")
        rollout_response = policy_model.generate_with_chat_template(
            user_message=rollout_prompt,
            max_tokens=400,
            temperature=0.8,
        )

        # Show truncated response
        truncated = rollout_response[:300] + "..." if len(rollout_response) > 300 else rollout_response
        print(f"Response: {truncated}")

        # Check if answer appears in rollout (case-insensitive)
        answer_lower = answer.lower()
        response_lower = rollout_response.lower()

        # Simple check: does the answer appear in the response?
        contains_answer = answer_lower in response_lower

        # For multi-part answers, check if key terms appear
        if not contains_answer and ',' in answer:
            # Split by comma and check each part
            answer_parts = [part.strip() for part in answer_lower.split(',')]
            contains_answer = all(part in response_lower for part in answer_parts if len(part) > 3)

        if contains_answer:
            correct_count += 1
            print(f"✓ Contains correct answer")
        else:
            print(f"✗ Does NOT contain correct answer")

    success_rate = correct_count / num_rollouts
    print(f"\nMC Estimation Results:")
    print(f"  Correct rollouts: {correct_count}/{num_rollouts}")
    print(f"  Success rate: {success_rate:.2%}")

    return success_rate


def test_complete_trajectory(policy_model, step_parser, question: str, answer: str, max_steps: int = 5):
    """Test generating a complete multi-step trajectory."""
    print("\n" + "=" * 70)
    print(f"[4] Test Complete Trajectory (max {max_steps} steps)")
    print("=" * 70)

    steps = []

    for step_num in range(1, max_steps + 1):
        print(f"\n--- Generating Step {step_num} ---")

        # Build prompt with previous steps
        if step_num == 1:
            user_prompt = f"""Question: {question}

Solve this step by step. You MUST respond with EXACTLY ONE step in this format:
"Step 1: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 1.

Step 1:"""
        else:
            steps_text = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
            user_prompt = f"""Question: {question}

{steps_text}

Continue solving. You MUST respond with EXACTLY ONE step in this format:
"Step {step_num}: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step {step_num}.

Step {step_num}:"""

        response = policy_model.generate_with_chat_template(
            user_message=user_prompt,
            max_tokens=200,
            temperature=0.8,
            stop_sequences=None
        )

        # Post-process: Extract first step only
        import re
        next_step_match = re.search(r'\n+Step\s+\d+:', response)
        if next_step_match:
            response = response[:next_step_match.start()]

        # Parse step content
        step_text = step_parser.parse_single_step_response(response, expected_step_num=step_num)
        steps.append(step_text)

        print(f"Step {step_num}: {step_text}")

        # Check if this step contains the answer (early stopping)
        if answer.lower() in step_text.lower():
            print(f"\n✓ Answer found in Step {step_num}!")
            break

    print("\n" + "=" * 70)
    print("Complete Trajectory:")
    print("=" * 70)
    for i, step in enumerate(steps, 1):
        print(f"Step {i}: {step}")

    return steps


def main():
    print("=" * 70)
    print("DEBUG: MC Estimation with HotpotQA")
    print("=" * 70)

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load HotpotQA questions (use validation data for testing)
    questions_path = Path("data/raw/hotpotqa_validation.jsonl")
    print(f"\nLoading questions from: {questions_path}")
    questions = load_hotpotqa_questions(questions_path, limit=10)
    print(f"✓ Loaded {len(questions)} questions")

    # Load model
    print("\n[0] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize step parser
    step_parser = StepParser()

    # Track overall results
    all_results = []

    # Test on all questions
    for idx, q in enumerate(questions, 1):
        question_text = q['question']
        gold_answer = q['answer']

        print("\n" + "=" * 70)
        print(f"QUESTION {idx}/{len(questions)}")
        print("=" * 70)
        print(f"ID: {q['_id']}")
        print(f"Question: {question_text}")
        print(f"Gold Answer: {gold_answer}")

        # Test 1: Single step generation
        step1 = test_single_step_generation(policy_model, step_parser, question_text, gold_answer)

        # Test 2: MC rollouts (quick test with 3 rollouts)
        success_rate = test_mc_rollouts(policy_model, question_text, gold_answer, step1, num_rollouts=3)

        # Store results
        all_results.append({
            'question_id': q['_id'],
            'question': question_text,
            'answer': gold_answer,
            'step1': step1,
            'mc_success_rate': success_rate,
        })

        print(f"\n→ MC Success Rate for this question: {success_rate:.2%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: All 10 Questions")
    print("=" * 70)

    avg_success_rate = sum(r['mc_success_rate'] for r in all_results) / len(all_results)

    print(f"\nOverall Statistics:")
    print(f"  Total questions tested: {len(all_results)}")
    print(f"  Average MC success rate: {avg_success_rate:.2%}")

    print(f"\nPer-question breakdown:")
    for i, r in enumerate(all_results, 1):
        print(f"  {i}. {r['question_id']}: {r['mc_success_rate']:.0%} success")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
    print("\nKey Observations:")
    print("1. ✓ Can load HotpotQA validation data")
    print("2. ✓ Can generate individual reasoning steps")
    print("3. ✓ Can perform MC rollouts for estimation")
    print(f"4. Average MC Success Rate: {avg_success_rate:.2%}")
    print("\nNext Steps:")
    print("- Implement adaptive generation with MC estimation")
    print("- Add RAG retrieval when CoT fails")
    print("- Test on more complex multi-hop questions")


if __name__ == "__main__":
    main()
