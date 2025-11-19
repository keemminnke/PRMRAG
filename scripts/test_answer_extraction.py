#!/usr/bin/env python3
"""Test answer extraction with realistic rollout outputs"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import (
    normalize_answer,
    extract_answer_from_text,
    compute_f1,
    compute_em
)

# Realistic rollout outputs (full text, not truncated)
test_cases = [
    {
        "gold": "yes",
        "rollouts": [
            "Therefore, the answer is yes, because both Scott Derrickson and Ed Wood were American filmmakers.",
            "Based on the information provided, the answer is yes. Both were American directors.",
            "The answer is: yes",
        ]
    },
    {
        "gold": "Chief of Protocol",
        "rollouts": [
            "Therefore, the answer is Chief of Protocol, which was the position held by Shirley Temple.",
            "The government position was Chief of Protocol.",
            "Answer: Chief of Protocol",
        ]
    },
    {
        "gold": "no",
        "rollouts": [
            "Therefore, the answer is no, because they are in different neighborhoods.",
            "No, they are not in the same neighborhood.",
            "The answer is: no",
        ]
    }
]

print("=" * 80)
print("ANSWER EXTRACTION TEST")
print("=" * 80)

for i, case in enumerate(test_cases, 1):
    print(f"\n{'='*80}")
    print(f"Test Case {i}: Gold = '{case['gold']}'")
    print(f"{'='*80}")

    gold_norm = normalize_answer(case['gold'])

    for j, rollout in enumerate(case['rollouts'], 1):
        print(f"\n  Rollout {j}: {rollout[:70]}...")

        # Extract and normalize
        extracted = extract_answer_from_text(rollout)
        normalized = normalize_answer(extracted)

        # Check match
        em = compute_em(normalized, gold_norm)
        f1 = compute_f1(normalized, gold_norm)

        print(f"    Extracted: '{extracted}'")
        print(f"    Normalized: '{normalized}'")
        print(f"    EM: {em}, F1: {f1:.3f}")

        if em or f1 >= 0.5:
            print(f"    ✅ MATCH!")
        else:
            print(f"    ❌ NO MATCH (gold_norm='{gold_norm}')")

print("\n" + "=" * 80)
