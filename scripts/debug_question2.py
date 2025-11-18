#!/usr/bin/env python3
"""Debug Question 2 to see why Final Answer is empty."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import extract_answer_from_text


def test_extraction():
    """Test answer extraction on Question 2 step content."""

    # Step 4 content from the output
    step4_content = 'J.K. Rowling wrote Harry Potter, and the first book was published in 1997 in the United Kingdom. Final Answer:'

    print("=" * 70)
    print("DEBUG: Question 2 Answer Extraction")
    print("=" * 70)

    print(f"\nStep 4 Content:")
    print(f"'{step4_content}'")

    print(f"\n[1] Testing extract_answer_from_text()...")
    extracted = extract_answer_from_text(step4_content)
    print(f"    Extracted: '{extracted}'")

    # Try different patterns manually
    import re

    print(f"\n[2] Testing regex patterns manually:")

    text = step4_content.lower()

    # Pattern 1
    pattern1 = r'(?:final\s+)?answer\s*(?:is)?\s*:?\s*(.+?)(?:\.\s+[A-Z]|\.$|$)'
    match1 = re.search(pattern1, text)
    print(f"\n    Pattern 1: {pattern1}")
    print(f"    Match: {match1}")
    if match1:
        print(f"    Group(1): '{match1.group(1)}'")

    # Try simpler pattern
    pattern_simple = r'final\s+answer\s*:\s*(.+)'
    match_simple = re.search(pattern_simple, text)
    print(f"\n    Simple Pattern: {pattern_simple}")
    print(f"    Match: {match_simple}")
    if match_simple:
        print(f"    Group(1): '{match_simple.group(1)}'")

    # Check what comes after "Final Answer:"
    if "final answer:" in text:
        idx = text.index("final answer:")
        after = text[idx + len("final answer:"):]
        print(f"\n[3] What comes after 'Final Answer:'?")
        print(f"    Text after marker: '{after}'")
        print(f"    Length: {len(after)}")
        print(f"    Stripped: '{after.strip()}'")
        print(f"    Is empty after strip? {len(after.strip()) == 0}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("\nThe problem:")
    print("  - Step 4 ends with 'Final Answer:' but has NO content after it")
    print("  - Model generated the marker but didn't write the actual answer")
    print("  - This is a GENERATION issue, not an extraction issue")
    print("\nPossible causes:")
    print("  1. Model hit max_tokens limit right after 'Final Answer:'")
    print("  2. Stop sequences [\"\\nStep\", \"\\n\\n\"] triggered early")
    print("  3. Model didn't complete the generation properly")


if __name__ == "__main__":
    test_extraction()
