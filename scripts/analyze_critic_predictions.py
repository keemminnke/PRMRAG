#!/usr/bin/env python3
"""Analyze critic model predictions to diagnose performance issues.

Usage:
    python scripts/analyze_critic_predictions.py \
        --results outputs/voting_comparison_v4.json \
        --trajectories outputs/hotpotqa_validation_500_no_rerank.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--trajectories", type=str, required=True)
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Load trajectories
    trajectories = []
    with open(args.trajectories) as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))

    print("=" * 70)
    print("CRITIC PREDICTION ANALYSIS")
    print("=" * 70)

    # 1. Overall stats
    print("\n[1] OVERALL RESULTS")
    print("-" * 50)
    for method in ['majority_voting', 'our_critic', 'versaprm']:
        data = results.get(method, {})
        if data.get('total', 0) > 0:
            acc = 100 * data['correct'] / data['total']
            print(f"  {method:20s}: {data['correct']}/{data['total']} ({acc:.1f}%)")

    # 2. Analyze where critic differs from majority
    print("\n[2] CRITIC vs MAJORITY VOTING DISAGREEMENTS")
    print("-" * 50)

    details = results.get('details', [])

    critic_correct_majority_wrong = 0
    critic_wrong_majority_correct = 0
    both_correct = 0
    both_wrong = 0

    critic_scores_when_correct = []
    critic_scores_when_wrong = []

    for detail in details:
        majority = detail.get('majority', {})
        critic = detail.get('critic', {})

        m_correct = majority.get('correct', False)
        c_correct = critic.get('correct', False)
        c_score = critic.get('score', 0)

        if c_correct and not m_correct:
            critic_correct_majority_wrong += 1
        elif not c_correct and m_correct:
            critic_wrong_majority_correct += 1
        elif c_correct and m_correct:
            both_correct += 1
        else:
            both_wrong += 1

        if c_correct:
            critic_scores_when_correct.append(c_score)
        else:
            critic_scores_when_wrong.append(c_score)

    print(f"  Both correct:              {both_correct}")
    print(f"  Both wrong:                {both_wrong}")
    print(f"  Critic ✓, Majority ✗:      {critic_correct_majority_wrong}")
    print(f"  Critic ✗, Majority ✓:      {critic_wrong_majority_correct}")

    print(f"\n  → Critic loses {critic_wrong_majority_correct - critic_correct_majority_wrong} questions compared to Majority")

    # 3. Score distribution analysis
    print("\n[3] CRITIC SCORE DISTRIBUTION")
    print("-" * 50)

    if critic_scores_when_correct:
        avg_correct = sum(critic_scores_when_correct) / len(critic_scores_when_correct)
        print(f"  Avg score when CORRECT: {avg_correct:.3f}")
    if critic_scores_when_wrong:
        avg_wrong = sum(critic_scores_when_wrong) / len(critic_scores_when_wrong)
        print(f"  Avg score when WRONG:   {avg_wrong:.3f}")

    if critic_scores_when_correct and critic_scores_when_wrong:
        diff = avg_correct - avg_wrong
        print(f"  Difference:             {diff:.3f}")
        if abs(diff) < 0.05:
            print("  ⚠️  WARNING: Scores are very similar - critic has poor discriminative power!")

    # 4. Check if critic always picks same answer as majority
    print("\n[4] ANSWER AGREEMENT")
    print("-" * 50)

    same_answer = 0
    diff_answer = 0
    for detail in details:
        m_ans = detail.get('majority', {}).get('answer', '').lower().strip()
        c_ans = detail.get('critic', {}).get('answer', '').lower().strip()
        if m_ans == c_ans:
            same_answer += 1
        else:
            diff_answer += 1

    print(f"  Same answer as majority: {same_answer}/{len(details)} ({100*same_answer/len(details):.1f}%)")
    print(f"  Different answer:        {diff_answer}/{len(details)} ({100*diff_answer/len(details):.1f}%)")

    if same_answer / len(details) > 0.9:
        print("  ⚠️  Critic almost always agrees with majority - not adding value!")

    # 5. Analyze specific failure cases
    print("\n[5] FAILURE CASE EXAMPLES (Critic wrong, Majority correct)")
    print("-" * 50)

    failure_count = 0
    for detail in details:
        majority = detail.get('majority', {})
        critic = detail.get('critic', {})

        if not critic.get('correct') and majority.get('correct'):
            failure_count += 1
            if failure_count <= 3:
                print(f"\n  Example {failure_count}:")
                print(f"    Question: {detail.get('question', '')[:80]}...")
                print(f"    Gold answer: {detail.get('gold_answer')}")
                print(f"    Majority picked: {majority.get('answer')} (votes: {majority.get('votes')})")
                print(f"    Critic picked: {critic.get('answer')} (score: {critic.get('score', 0):.3f})")

    # 6. Score variance check
    print("\n[6] SCORE VARIANCE CHECK")
    print("-" * 50)

    all_scores = [d.get('critic', {}).get('score', 0) for d in details if d.get('critic')]
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        variance = sum((s - avg_score) ** 2 for s in all_scores) / len(all_scores)
        std_dev = variance ** 0.5

        print(f"  Average score: {avg_score:.3f}")
        print(f"  Std deviation: {std_dev:.3f}")
        print(f"  Min score:     {min(all_scores):.3f}")
        print(f"  Max score:     {max(all_scores):.3f}")

        if std_dev < 0.1:
            print("  ⚠️  Low variance - critic gives similar scores to all trajectories!")

    # 7. Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    issues = []
    if critic_scores_when_correct and critic_scores_when_wrong:
        if abs(avg_correct - avg_wrong) < 0.05:
            issues.append("Poor discriminative power - scores don't correlate with correctness")

    if same_answer / len(details) > 0.85:
        issues.append("Critic mostly agrees with majority - not providing additional signal")

    if all_scores and std_dev < 0.1:
        issues.append("Low score variance - model might be predicting GOOD for everything")

    if issues:
        print("\nDetected issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\nPossible fixes:")
        print("  1. Balance training data (undersample GOOD or oversample BAD)")
        print("  2. Use focal loss to focus on hard examples")
        print("  3. Change selection strategy (e.g., use min score instead of avg)")
        print("  4. Add negative examples from wrong trajectories")
        print("  5. Train longer with lower learning rate")
    else:
        print("\nNo obvious issues detected. Consider:")
        print("  - Checking if test distribution matches training")
        print("  - Analyzing specific failure patterns")


if __name__ == "__main__":
    main()
