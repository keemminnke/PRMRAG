#!/usr/bin/env python3
"""Test answer extraction improvements."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from prmrag.generation.adaptive_generator import extract_answer_from_text

def test_answer_extraction():
    """Test various answer formats."""

    test_cases = [
        # (input_text, expected_answer, description)
        (
            "Final Answer: 3,677 seated",
            "3,677 seated",
            "Number with comma and unit"
        ),
        (
            "Therefore, the answer is 3,677 seated.",
            "3,677 seated",
            "Number with comma after 'Therefore'"
        ),
        (
            "The arena has a seating capacity of 3,677 seats.",
            "the arena has a seating capacity of 3,677 seats",
            "Full sentence with comma"
        ),
        (
            "Final Answer: yes",
            "yes",
            "Simple yes/no"
        ),
        (
            "The answer is YG Entertainment.",
            "yg entertainment",
            "Company name with period"
        ),
        (
            "Final Answer: Scott Derrickson and Ed Wood are both American, so yes.",
            "scott derrickson and ed wood are both american, so yes",
            "Complex answer with 'and'"
        ),
        (
            "Therefore, both are from the United States.",
            "both are from the united states",
            "Location answer"
        ),
        (
            "The answer is Greenwich Village, New York City.",
            "greenwich village, new york city",
            "Location with comma"
        ),
        (
            "Final Answer: Eenasul Fateh, also known as Aladin.",
            "eenasul fateh, also known as aladin",
            "Name with description"
        ),
        (
            "Therefore, the answer is 1,234,567 people because it's a large city.",
            "1,234,567 people",
            "Large number with 'because' connector"
        ),
    ]

    print("=" * 80)
    print("TESTING ANSWER EXTRACTION")
    print("=" * 80)
    print()

    passed = 0
    failed = 0

    for i, (input_text, expected, description) in enumerate(test_cases, 1):
        result = extract_answer_from_text(input_text)

        # Compare (case-insensitive, whitespace-normalized)
        result_normalized = " ".join(result.lower().split())
        expected_normalized = " ".join(expected.lower().split())

        is_pass = result_normalized == expected_normalized

        if is_pass:
            status = "✅ PASS"
            passed += 1
        else:
            status = "❌ FAIL"
            failed += 1

        print(f"Test {i}: {description}")
        print(f"  Input:    {input_text[:60]}...")
        print(f"  Expected: {expected}")
        print(f"  Got:      {result}")
        print(f"  {status}")
        print()

    print("=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)

    return failed == 0

if __name__ == "__main__":
    success = test_answer_extraction()
    sys.exit(0 if success else 1)
