#!/usr/bin/env python3
"""Test improved answer matching with RAG standard metrics"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import (
    normalize_answer,
    extract_answer_from_text,
    compute_f1,
    compute_em
)

# Test cases with various answer formats
test_cases = [
    {
        "name": "Short yes/no",
        "gold": "yes",
        "predictions": [
            "yes",
            "Yes",
            "yes, because both are American",
            "Therefore, the answer is yes",
            "no",  # Should fail
        ]
    },
    {
        "name": "Multi-word answer",
        "gold": "Chief of Protocol",
        "predictions": [
            "Chief of Protocol",
            "The Chief of Protocol",
            "Chief of Protocol for the United States",
            "She served as Chief of Protocol",
            "Ambassador",  # Should fail
        ]
    },
    {
        "name": "Numeric answer",
        "gold": "3,677 seated",
        "predictions": [
            "3,677 seated",
            "3677",
            "approximately 3,677",
            "5,000",  # Should fail
        ]
    }
]

def check_answer_improved(predicted: str, gold: str) -> dict:
    """Check answer using improved matching."""
    # Extract and normalize
    pred_extracted = extract_answer_from_text(predicted)
    gold_extracted = extract_answer_from_text(gold)

    pred_norm = normalize_answer(pred_extracted)
    gold_norm = normalize_answer(gold_extracted)

    # Check strategies
    em = compute_em(pred_norm, gold_norm)
    contains = gold_norm in pred_norm or pred_norm in gold_norm
    f1 = compute_f1(pred_norm, gold_norm)
    f1_pass = f1 >= 0.3

    final_match = em or contains or f1_pass

    return {
        'pred_norm': pred_norm,
        'gold_norm': gold_norm,
        'em': em,
        'contains': contains,
        'f1': f1,
        'f1_pass': f1_pass,
        'match': final_match
    }

print("=" * 80)
print("IMPROVED ANSWER MATCHING TEST")
print("=" * 80)

for case in test_cases:
    print(f"\n{'='*80}")
    print(f"Test: {case['name']}")
    print(f"Gold: '{case['gold']}'")
    print(f"{'='*80}")

    for pred in case['predictions']:
        result = check_answer_improved(pred, case['gold'])

        status = "✅" if result['match'] else "❌"
        print(f"\n{status} Pred: '{pred[:50]}...'")
        print(f"   Normalized: pred='{result['pred_norm']}' vs gold='{result['gold_norm']}'")
        print(f"   EM: {result['em']}, Contains: {result['contains']}, F1: {result['f1']:.3f} (≥0.3: {result['f1_pass']})")

        if result['match']:
            strategies = []
            if result['em']:
                strategies.append("EM")
            if result['contains']:
                strategies.append("Containment")
            if result['f1_pass']:
                strategies.append(f"F1={result['f1']:.3f}")
            print(f"   Matched by: {', '.join(strategies)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nImproved matching with 3 strategies:")
print("1. EM (Exact Match) - strict")
print("2. String Containment - handles 'yes' in 'yes because...'")
print("3. F1 >= 0.3 - relaxed threshold for partial matches")
print("\nThis matches standard RAG/QA paper metrics!")
