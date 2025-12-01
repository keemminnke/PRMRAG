#!/usr/bin/env python3
"""
Test script to verify RAG content parsing logic.
"""
import sys
import json
import jsonlines
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.generation.adaptive_generator import parse_rag_content


def test_parsing_on_existing_data():
    """Test parsing on existing trajectory data."""

    # Load existing trajectories
    traj_file = Path("outputs/batch_train_test/results_20251126_145231.jsonl")
    if not traj_file.exists():
        print(f"❌ File not found: {traj_file}")
        return

    print(f"Loading trajectories from {traj_file}")
    print("=" * 60)

    rag_steps_found = 0
    parsing_stats = {
        'total_rag_steps': 0,
        'thought_parsed': 0,
        'action_parsed': 0,
        'observation_parsed': 0,
        'all_fields_parsed': 0,
    }

    with jsonlines.open(traj_file) as reader:
        for i, traj in enumerate(reader):
            if i >= 10:  # Test first 10 trajectories
                break

            for step in traj.get('steps', []):
                if step.get('step_type') == 'rag':
                    parsing_stats['total_rag_steps'] += 1
                    content = step.get('content', '')

                    if content:
                        # Parse the content
                        parsed = parse_rag_content(content)

                        # Count what was parsed
                        if parsed['thought']:
                            parsing_stats['thought_parsed'] += 1
                        if parsed['action']:
                            parsing_stats['action_parsed'] += 1
                        if parsed['observation']:
                            parsing_stats['observation_parsed'] += 1
                        if all(v is not None for v in parsed.values()):
                            parsing_stats['all_fields_parsed'] += 1

                        # Print first few examples
                        if rag_steps_found < 3:
                            print(f"\n[Example {rag_steps_found + 1}]")
                            print(f"Question: {traj.get('question', 'N/A')[:80]}...")
                            print(f"\nOriginal content:")
                            print(content[:200] + "..." if len(content) > 200 else content)
                            print(f"\n✓ Parsed fields:")
                            print(f"  - Thought: {parsed['thought'][:80] + '...' if parsed['thought'] and len(parsed['thought']) > 80 else parsed['thought']}")
                            print(f"  - Action: {parsed['action']}")
                            print(f"  - Action Input: {parsed['action_input']}")
                            print(f"  - Observation: {parsed['observation'][:80] + '...' if parsed['observation'] and len(parsed['observation']) > 80 else parsed['observation']}")
                            print("-" * 60)

                        rag_steps_found += 1

    # Print statistics
    print(f"\n{'=' * 60}")
    print("PARSING STATISTICS")
    print(f"{'=' * 60}")
    print(f"Total RAG steps: {parsing_stats['total_rag_steps']}")
    print(f"Thought parsed: {parsing_stats['thought_parsed']} ({parsing_stats['thought_parsed']/parsing_stats['total_rag_steps']*100:.1f}%)")
    print(f"Action parsed: {parsing_stats['action_parsed']} ({parsing_stats['action_parsed']/parsing_stats['total_rag_steps']*100:.1f}%)")
    print(f"Observation parsed: {parsing_stats['observation_parsed']} ({parsing_stats['observation_parsed']/parsing_stats['total_rag_steps']*100:.1f}%)")
    print(f"All fields parsed: {parsing_stats['all_fields_parsed']} ({parsing_stats['all_fields_parsed']/parsing_stats['total_rag_steps']*100:.1f}%)")

    if parsing_stats['all_fields_parsed'] == parsing_stats['total_rag_steps']:
        print(f"\n✅ SUCCESS: All RAG steps were parsed successfully!")
    else:
        missed = parsing_stats['total_rag_steps'] - parsing_stats['all_fields_parsed']
        print(f"\n⚠ WARNING: {missed} RAG steps had incomplete parsing")


if __name__ == "__main__":
    test_parsing_on_existing_data()
