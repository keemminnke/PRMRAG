#!/usr/bin/env python3
"""Debug MC estimation to understand why it's always 0.0"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import (
    normalize_answer,
    extract_answer_from_text,
    compute_f1,
    compute_em
)

# Test cases from the actual logs
test_cases = [
    {
        "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
        "gold": "yes",
        "rollout_samples": [
            "yes, scott derrickson and ed wood were of the same",
            "yes, scott derrickson and ed wood were of the same"
        ]
    },
    {
        "question": "What government position was held by the woman who portrayed Corliss Archer?",
        "gold": "Chief of Protocol",
        "rollout_samples": [
            "[none, as meg ryan did not hold any government pos",
            "that meg ryan did not hold any government position"
        ]
    },
    {
        "question": "Are the Laleli Mosque and Esma Sultan Mansion located in the same neighborhood?",
        "gold": "no",
        "rollout_samples": [
            "yes, the laleli mosque and esma sultan mansion are",
            "yes, the laleli mosque and esma sultan mansion are"
        ]
    }
]

print("=" * 80)
print("MC ESTIMATION DEBUG")
print("=" * 80)

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test Case {i}")
    print(f"{'='*80}")
    print(f"Question: {case['question'][:60]}...")
    print(f"Gold answer: '{case['gold']}'")

    # Process gold answer
    gold_extracted = extract_answer_from_text(case['gold'])
    gold_norm = normalize_answer(gold_extracted)
    print(f"\nGold (extracted): '{gold_extracted}'")
    print(f"Gold (normalized): '{gold_norm}'")

    print(f"\n--- Rollout Analysis ---")
    for j, rollout in enumerate(case['rollout_samples'], 1):
        print(f"\nRollout {j}: '{rollout}...'")

        # Extract and normalize
        pred_extracted = extract_answer_from_text(rollout)
        pred_norm = normalize_answer(pred_extracted)

        print(f"  Extracted: '{pred_extracted}'")
        print(f"  Normalized: '{pred_norm}'")

        # Check match
        em = compute_em(pred_norm, gold_norm)
        f1 = compute_f1(pred_norm, gold_norm)

        print(f"  EM: {em}")
        print(f"  F1: {f1:.3f}")
        print(f"  Match (F1 >= 0.5): {f1 >= 0.5}")

        # Show why it might not match
        if not (em or f1 >= 0.5):
            print(f"  ❌ NO MATCH")
            print(f"     Gold tokens: {gold_norm.split()}")
            print(f"     Pred tokens: {pred_norm.split()}")
        else:
            print(f"  ✅ MATCH!")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("\nPotential Issues:")
print("1. extract_answer_from_text() might not extract the right part")
print("2. normalize_answer() might be too aggressive")
print("3. F1 threshold of 0.5 might be too strict")
print("4. Rollouts are truncated at 50 chars for logging")
print("\nNext: Check full rollout outputs (not truncated)")
