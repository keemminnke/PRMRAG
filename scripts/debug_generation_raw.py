#!/usr/bin/env python3
"""Debug raw model generation for Question 2."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.step_forcing import StepForcingPrompt, ForcedStep


def test_raw_generation():
    """Test what model actually generates."""

    print("=" * 70)
    print("DEBUG: Raw Model Generation for Question 2")
    print("=" * 70)

    # Load config and model
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    print("\n[1] Loading model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Simulate Question 2 state after Step 5
    question = "Who wrote Harry Potter and when was the first book published?"

    previous_steps = [
        ForcedStep(1, 'Identify the author of the Harry Potter series.\nThe author of the Harry Potter series is J.K. Rowling.'),
        ForcedStep(2, 'The first Harry Potter book, "Harry Potter and the Philosopher\'s Stone," was published in 1997 in the United Kingdom.'),
        ForcedStep(3, 'The first Harry Potter book was published in 1997 in the United Kingdom by Bloomsbury Publishing.'),
        ForcedStep(4, 'J.K. Rowling wrote Harry Potter, and the first book was published in 1997 in the United Kingdom.'),
        ForcedStep(5, 'J.K. Rowling wrote Harry Potter, and the first book, "Harry Potter and the Philosopher\'s Stone," was published in 1997 in the United Kingdom.'),
    ]

    # Build continuation prompt for Step 6
    prompt_builder = StepForcingPrompt()
    prompt = prompt_builder.build_continuation_prompt(
        question=question,
        previous_steps=previous_steps,
        context=""
    )

    print("\n[2] Prompt for Step 6:")
    print("-" * 70)
    print(prompt)
    print("-" * 70)

    # Generate with DIFFERENT stop sequences to see what happens
    print("\n[3] Testing different stop sequences:\n")

    # Test 1: No stop sequences
    print("Test 1: No stop sequences")
    response1 = policy_model.generate_with_chat_template(
        user_message=prompt,
        max_tokens=200,
        temperature=0.8,
        top_p=0.95,
        stop_sequences=None,
    )
    print(f"Response: '{response1}'")
    print()

    # Test 2: Only "\nStep"
    print("Test 2: stop_sequences=[\"\\nStep\"]")
    response2 = policy_model.generate_with_chat_template(
        user_message=prompt,
        max_tokens=200,
        temperature=0.8,
        top_p=0.95,
        stop_sequences=["\nStep"],
    )
    print(f"Response: '{response2}'")
    print()

    # Test 3: Both "\nStep" and "\n\n"
    print("Test 3: stop_sequences=[\"\\nStep\", \"\\n\\n\"]")
    response3 = policy_model.generate_with_chat_template(
        user_message=prompt,
        max_tokens=200,
        temperature=0.8,
        top_p=0.95,
        stop_sequences=["\nStep", "\n\n"],
    )
    print(f"Response: '{response3}'")
    print()

    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)


if __name__ == "__main__":
    test_raw_generation()
