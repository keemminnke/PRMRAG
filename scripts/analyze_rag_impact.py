#!/usr/bin/env python3
"""Analyze RAG impact on reasoning quality.

This script analyzes batch test results to determine:
1. When does RAG help? (MC improvement, accuracy improvement)
2. What types of questions benefit from RAG?
3. Are RAG steps labeled as 'good' or 'bad'?
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def analyze_rag_effectiveness(results: List[Dict[str, Any]]):
    """Analyze when RAG helps vs when it doesn't."""

    print("\n" + "=" * 70)
    print("RAG EFFECTIVENESS ANALYSIS")
    print("=" * 70)

    # Separate questions by RAG usage
    with_rag = [r for r in results if r['has_rag']]
    without_rag = [r for r in results if not r['has_rag']]

    print(f"\n1. Basic Statistics:")
    print(f"   Total questions: {len(results)}")
    print(f"   - With RAG: {len(with_rag)} ({len(with_rag)/len(results)*100:.1f}%)")
    print(f"   - Without RAG (CoT only): {len(without_rag)} ({len(without_rag)/len(results)*100:.1f}%)")

    # Accuracy comparison
    if with_rag:
        rag_correct = sum(1 for r in with_rag if r['is_correct'])
        print(f"\n2. Accuracy by RAG Usage:")
        print(f"   With RAG: {rag_correct}/{len(with_rag)} ({rag_correct/len(with_rag)*100:.1f}%)")

    if without_rag:
        cot_correct = sum(1 for r in without_rag if r['is_correct'])
        print(f"   CoT only: {cot_correct}/{len(without_rag)} ({cot_correct/len(without_rag)*100:.1f}%)")

    # Analyze RAG interventions
    print(f"\n3. RAG Intervention Analysis:")

    # Categorize RAG steps by label and MC improvement
    rag_good = []  # RAG steps labeled 'good'
    rag_bad = []   # RAG steps labeled 'bad'

    for result in with_rag:
        for intervention in result['rag_interventions']:
            if intervention['label'] == 'good':
                rag_good.append(intervention)
            else:
                rag_bad.append(intervention)

    print(f"   Total RAG steps: {len(rag_good) + len(rag_bad)}")
    print(f"   - Labeled 'good' (RPE >= 0.8): {len(rag_good)} ({len(rag_good)/(len(rag_good)+len(rag_bad))*100:.1f}%)")
    print(f"   - Labeled 'bad' (RPE < 0.8): {len(rag_bad)} ({len(rag_bad)/(len(rag_good)+len(rag_bad))*100:.1f}%)")

    # MC improvement distribution
    if rag_good:
        avg_improvement_good = sum(r['mc_improvement'] for r in rag_good) / len(rag_good)
        print(f"\n   Average MC improvement for 'good' RAG steps: {avg_improvement_good:.3f}")

    if rag_bad:
        avg_improvement_bad = sum(r['mc_improvement'] for r in rag_bad) / len(rag_bad)
        print(f"   Average MC improvement for 'bad' RAG steps: {avg_improvement_bad:.3f}")

    # Find best RAG interventions
    print(f"\n4. Most Effective RAG Interventions:")
    print(f"   (Top 10 by MC improvement)")
    print(f"   " + "-" * 66)

    all_interventions = []
    for result in with_rag:
        for intervention in result['rag_interventions']:
            all_interventions.append({
                'question_id': result['question_id'],
                'question': result['question'],
                'step_num': intervention['step_num'],
                'mc_improvement': intervention['mc_improvement'],
                'rpe': intervention['rpe'],
                'label': intervention['label'],
                'is_correct': result['is_correct'],
            })

    all_interventions.sort(key=lambda x: x['mc_improvement'], reverse=True)

    for i, interv in enumerate(all_interventions[:10], 1):
        status = "✅" if interv['is_correct'] else "❌"
        print(f"   {i}. {status} {interv['question_id']} (Step {interv['step_num']})")
        print(f"      MC improvement: {interv['mc_improvement']:+.3f}, RPE: {interv['rpe']:.3f}, Label: {interv['label']}")
        print(f"      Q: {interv['question'][:70]}...")
        print()

    # Find worst RAG interventions (negative impact)
    negative_interventions = [x for x in all_interventions if x['mc_improvement'] < 0]
    if negative_interventions:
        print(f"\n5. RAG Interventions with Negative Impact:")
        print(f"   (MC decreased after RAG)")
        print(f"   " + "-" * 66)

        negative_interventions.sort(key=lambda x: x['mc_improvement'])

        for i, interv in enumerate(negative_interventions[:10], 1):
            status = "✅" if interv['is_correct'] else "❌"
            print(f"   {i}. {status} {interv['question_id']} (Step {interv['step_num']})")
            print(f"      MC improvement: {interv['mc_improvement']:+.3f}, RPE: {interv['rpe']:.3f}, Label: {interv['label']}")
            print(f"      Q: {interv['question'][:70]}...")
            print()


def find_interesting_cases(results: List[Dict[str, Any]]):
    """Find interesting cases for manual review."""

    print("\n" + "=" * 70)
    print("INTERESTING CASES FOR MANUAL REVIEW")
    print("=" * 70)

    # Case 1: RAG helped → became correct
    print(f"\n1. Cases where RAG likely helped (has RAG, correct answer):")
    print(f"   " + "-" * 66)

    rag_helped = [r for r in results if r['has_rag'] and r['is_correct']]
    for i, result in enumerate(rag_helped[:5], 1):
        print(f"   {i}. {result['question_id']}")
        print(f"      Q: {result['question']}")
        print(f"      Steps: {result['num_steps']} ({result['num_rag_steps']} RAG)")
        print(f"      Predicted: {result['predicted_answer'][:80]}...")
        print(f"      Gold: {result['gold_answer']}")
        print()

    # Case 2: RAG didn't help → still wrong
    print(f"\n2. Cases where RAG didn't help (has RAG, wrong answer):")
    print(f"   " + "-" * 66)

    rag_failed = [r for r in results if r['has_rag'] and not r['is_correct']]
    for i, result in enumerate(rag_failed[:5], 1):
        print(f"   {i}. {result['question_id']}")
        print(f"      Q: {result['question']}")
        print(f"      Steps: {result['num_steps']} ({result['num_rag_steps']} RAG)")
        print(f"      Predicted: {result['predicted_answer'][:80]}...")
        print(f"      Gold: {result['gold_answer']}")
        print()

    # Case 3: CoT only succeeded
    print(f"\n3. Cases where CoT alone succeeded (no RAG needed):")
    print(f"   " + "-" * 66)

    cot_success = [r for r in results if not r['has_rag'] and r['is_correct']]
    for i, result in enumerate(cot_success[:5], 1):
        print(f"   {i}. {result['question_id']}")
        print(f"      Q: {result['question']}")
        print(f"      Steps: {result['num_steps']} (all CoT)")
        print(f"      Predicted: {result['predicted_answer'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RAG impact from batch test results"
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to results JSONL file from batch_test_adaptive.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save detailed analysis to file",
    )
    args = parser.parse_args()

    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: File not found: {results_file}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {results_file}")
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} results")

    # Run analysis
    analyze_rag_effectiveness(results)
    find_interesting_cases(results)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
