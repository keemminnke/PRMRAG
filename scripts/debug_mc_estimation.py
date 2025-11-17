#!/usr/bin/env python3
"""Debug MC estimation with real Qwen model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.step_forcing import StepParser

def main():
    print("=" * 70)
    print("DEBUG: MC Estimation with Qwen2.5-7B")
    print("=" * 70)

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load model
    print("\n[1] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize step parser
    step_parser = StepParser()

    # Test question
    question = "What is the capital of France?"

    # Test 1: Generate Step 1 with chat template
    print("\n" + "=" * 70)
    print("[2] Test Step 1 Generation with Chat Template")
    print("=" * 70)

    user_prompt = f"""Question: {question}

Solve this step by step. You MUST respond with EXACTLY ONE step in this format:
"Step 1: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 1.

Step 1:"""

    print(f"\nPrompt (user message):")
    print("-" * 70)
    print(user_prompt)
    print("-" * 70)

    print("\nGenerating with chat template...")
    response = policy_model.generate_with_chat_template(
        user_message=user_prompt,
        max_tokens=200,
        temperature=0.8,
        stop_sequences=None  # No stop sequences - post-process instead
    )

    # Post-process: Extract first step only (remove future steps)
    import re
    next_step_match = re.search(r'\n+Step\s+\d+:', response)
    if next_step_match:
        response = response[:next_step_match.start()]

    print("\n✓ Generated Response (raw):")
    print("-" * 70)
    print(response)
    print("-" * 70)

    # Parse to extract step content (removes "Step 1:" prefix if present)
    step1_content = step_parser.parse_single_step_response(response, expected_step_num=1)

    print("\n✓ Parsed Step Content (without 'Step 1:' prefix):")
    print("-" * 70)
    print(step1_content)
    print("-" * 70)

    # Test 2: Generate Step 2 (continuation)
    print("\n" + "=" * 70)
    print("[3] Test Step 2 Generation (Continuation)")
    print("=" * 70)

    # step1_content already parsed above (without "Step 1:" prefix)
    user_prompt_2 = f"""Question: {question}

Step 1: {step1_content}

Continue solving. You MUST respond with EXACTLY ONE step in this format:
"Step 2: [your reasoning]"

Do NOT write multiple steps. Write ONLY Step 2.

Step 2:"""

    print(f"\nPrompt (user message):")
    print("-" * 70)
    print(user_prompt_2)
    print("-" * 70)

    print("\nGenerating with chat template...")
    response2 = policy_model.generate_with_chat_template(
        user_message=user_prompt_2,
        max_tokens=200,
        temperature=0.8,
        stop_sequences=None  # No stop sequences - post-process instead
    )

    # Post-process: Extract first step only (remove future steps)
    next_step_match = re.search(r'\n+Step\s+\d+:', response2)
    if next_step_match:
        response2 = response2[:next_step_match.start()]

    print("\n✓ Generated Response (raw):")
    print("-" * 70)
    print(response2)
    print("-" * 70)

    # Parse to extract step content (removes "Step 2:" prefix if present)
    step2_content = step_parser.parse_single_step_response(response2, expected_step_num=2)

    print("\n✓ Parsed Step Content (without 'Step 2:' prefix):")
    print("-" * 70)
    print(step2_content)
    print("-" * 70)

    # Test 3: MC Rollout simulation
    print("\n" + "=" * 70)
    print("[4] Test MC Rollout (5 rollouts from Step 1)")
    print("=" * 70)

    rollout_prompt = f"""Question: {question}

Step 1: {step1_content}

Continue solving to reach the final answer. Provide your final answer."""

    print(f"\nRollout Prompt:")
    print("-" * 70)
    print(rollout_prompt)
    print("-" * 70)

    print("\nRunning 5 rollouts...")
    for i in range(5):
        print(f"\n--- Rollout {i+1}/5 ---")
        rollout_response = policy_model.generate_with_chat_template(
            user_message=rollout_prompt,
            max_tokens=300,
            temperature=0.8,
        )
        print(f"Response: {rollout_response[:200]}...")

        # Check if contains answer
        contains_paris = "paris" in rollout_response.lower()
        print(f"Contains 'Paris': {contains_paris}")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)
    print("\nKey Observations:")
    print("1. Does Step 1 generation work correctly?")
    print("2. Are stop sequences working?")
    print("3. Do rollouts contain correct answers?")
    print("4. Is the chat template format correct?")


if __name__ == "__main__":
    main()
