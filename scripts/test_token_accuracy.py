#!/usr/bin/env python3
"""Test token-level accuracy matching"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import (
    normalize_answer,
    extract_answer_from_text,
    compute_em
)

def check_token_accuracy(predicted: str, gold: str, threshold: float = 0.8) -> dict:
    """Check using token-level accuracy."""
    # Extract and normalize
    pred_extracted = extract_answer_from_text(predicted)
    gold_extracted = extract_answer_from_text(gold)

    pred_norm = normalize_answer(pred_extracted)
    gold_norm = normalize_answer(gold_extracted)

    # Exact match
    em = compute_em(pred_norm, gold_norm)
    if em:
        return {'match': True, 'accuracy': 1.0, 'method': 'EM'}

    # Token accuracy
    gold_tokens = gold_norm.split()
    pred_tokens = pred_norm.split()

    if not gold_tokens:
        return {'match': False, 'accuracy': 0.0, 'method': 'Empty gold'}

    matches = sum(1 for token in gold_tokens if token in pred_tokens)
    accuracy = matches / len(gold_tokens)

    return {
        'match': accuracy >= threshold,
        'accuracy': accuracy,
        'method': f'Token Acc ({matches}/{len(gold_tokens)})',
        'gold_tokens': gold_tokens,
        'pred_tokens': pred_tokens
    }

# Test cases
test_cases = [
    {
        "gold": "yes",
        "predictions": [
            ("yes", True),
            ("yes, because both are American", True),
            ("no", False),
        ]
    },
    {
        "gold": "Chief of Protocol",
        "predictions": [
            ("Chief of Protocol", True),
            ("She was Chief of Protocol", True),  # 3/3 = 100%
            ("Chief", False),  # 1/3 = 33%
            ("Protocol Chief for the US", True),  # 2/3 = 67% < 80%... hmm
        ]
    },
]

print("=" * 80)
print("TOKEN-LEVEL ACCURACY TEST (threshold = 80%)")
print("=" * 80)

for case in test_cases:
    print(f"\n{'='*80}")
    print(f"Gold: '{case['gold']}'")
    print(f"{'='*80}")

    for pred, expected in case['predictions']:
        result = check_token_accuracy(pred, case['gold'])

        status = "✅" if result['match'] else "❌"
        expected_status = "✅" if expected else "❌"

        print(f"\n{status} Pred: '{pred}'")
        print(f"   Accuracy: {result['accuracy']:.2%} ({result['method']})")

        if result['match'] != expected:
            print(f"   ⚠️  Expected: {expected_status}")

        if 'gold_tokens' in result:
            print(f"   Gold: {result['gold_tokens']}")
            print(f"   Pred: {result['pred_tokens']}")

print("\n" + "=" * 80)
print("Note: Token accuracy = (# gold tokens in prediction) / (# gold tokens)")
print("Accept if >= 80%")
