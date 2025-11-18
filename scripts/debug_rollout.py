#!/usr/bin/env python3
"""Debug rollout mechanism to see why MC = 0."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config


def test_rollout():
    """Test a simple rollout."""
    print("=" * 70)
    print("DEBUG: Testing Rollout Mechanism")
    print("=" * 70)

    # Load config and model
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    print("\n[1] Loading model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Test question
    question = "What is the capital of France and what is its population?"
    gold_answer = "Paris, approximately 2.2 million"

    print(f"\n[2] Test Question:")
    print(f"    Q: {question}")
    print(f"    Gold Answer: {gold_answer}")

    # Build rollout prompt (empty state)
    print(f"\n[3] Testing rollout from empty state...")

    prompt_lines = [
        f"Question: {question}\n",
        "Solve this problem step by step.",
        "End your response with: 'Therefore, the answer is [your answer].'"
    ]
    prompt = "\n".join(prompt_lines)

    print("\n--- Prompt ---")
    print(prompt)
    print("--- End Prompt ---\n")

    # Generate rollout
    print("[4] Generating rollout (this may take a moment)...\n")

    response = policy_model.generate_with_chat_template(
        user_message=prompt,
        max_tokens=800,
        temperature=0.8,
        top_p=0.95,
    )

    print("--- Rollout Response ---")
    print(response)
    print("--- End Response ---\n")

    # Extract answer
    from prmrag.generation.adaptive_generator import extract_answer_from_text, normalize_answer

    extracted = extract_answer_from_text(response)
    print(f"[5] Extracted Answer: '{extracted}'")

    # Check if correct
    pred_norm = normalize_answer(extracted)
    gold_norm = normalize_answer(gold_answer)

    print(f"\n[6] Normalized:")
    print(f"    Predicted: '{pred_norm}'")
    print(f"    Gold:      '{gold_norm}'")

    # Token-level matching (new approach)
    pred_tokens = set(pred_norm.split())
    gold_tokens = set(gold_norm.split())

    print(f"\n[7] Token Sets:")
    print(f"    Predicted tokens: {pred_tokens}")
    print(f"    Gold tokens:      {gold_tokens}")

    is_correct = gold_tokens.issubset(pred_tokens)
    print(f"\n[8] Are all gold tokens in prediction? {is_correct}")

    if not is_correct:
        print("\n❌ PROBLEM FOUND: Gold answer not found in prediction!")
        print("    This is why MC = 0 (no rollouts succeed)")
    else:
        print("\n✅ Rollout succeeded!")

    print("\n" + "=" * 70)
    print("DEBUG COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_rollout()
