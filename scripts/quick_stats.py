#!/usr/bin/env python3
"""Quick statistics for trajectory data without interactive mode."""

import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_trajectories(jsonl_path: str):
    """Analyze trajectories and print detailed statistics."""

    print(f"Loading {jsonl_path}...")
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    print(f"✓ Loaded {len(results)} trajectories\n")

    # Overall stats
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    incorrect = total - correct

    # RAG vs CoT
    with_rag = sum(1 for r in results if r.get('has_rag', False))
    rag_correct = sum(1 for r in results if r.get('has_rag', False) and r['is_correct'])

    cot_only = total - with_rag
    cot_correct = sum(1 for r in results if not r.get('has_rag', False) and r['is_correct'])

    # MC analysis
    mc_stuck_at_zero = sum(1 for r in results if r['steps'] and r['steps'][-1]['mc_after'] == 0.0)

    # Step statistics
    total_steps = sum(r['num_steps'] for r in results)
    total_cot_steps = sum(r['num_cot_steps'] for r in results)
    total_rag_steps = sum(r['num_rag_steps'] for r in results)

    # Label distribution
    label_counts = defaultdict(int)
    for r in results:
        for step in r['steps']:
            label_counts[step.get('label', 'unknown')] += 1

    # Print report
    print("=" * 70)
    print("TRAJECTORY ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n📊 OVERALL STATISTICS")
    print(f"   Total trajectories: {total}")
    print(f"   ✓ Correct:   {correct} ({correct/total*100:.1f}%)")
    print(f"   ✗ Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")

    print(f"\n🔍 RETRIEVAL ANALYSIS")
    print(f"   With RAG:    {with_rag} trajectories")
    print(f"     ✓ Correct: {rag_correct}/{with_rag} ({rag_correct/with_rag*100:.1f}%)")
    print(f"   CoT Only:    {cot_only} trajectories")
    print(f"     ✓ Correct: {cot_correct}/{cot_only} ({cot_correct/cot_only*100:.1f}%)")

    print(f"\n📈 MONTE CARLO ANALYSIS")
    print(f"   MC stuck at 0.0: {mc_stuck_at_zero} trajectories ({mc_stuck_at_zero/total*100:.1f}%)")

    # MC distribution
    mc_ranges = {"High (0.8-1.0)": 0, "Medium (0.3-0.8)": 0, "Low (0.0-0.3)": 0}
    for r in results:
        if r['steps']:
            final_mc = r['steps'][-1]['mc_after']
            if final_mc >= 0.8:
                mc_ranges["High (0.8-1.0)"] += 1
            elif final_mc >= 0.3:
                mc_ranges["Medium (0.3-0.8)"] += 1
            else:
                mc_ranges["Low (0.0-0.3)"] += 1

    for range_name, count in mc_ranges.items():
        print(f"   {range_name}: {count} ({count/total*100:.1f}%)")

    print(f"\n🔢 STEP STATISTICS")
    print(f"   Total steps:      {total_steps}")
    print(f"   CoT steps:        {total_cot_steps} ({total_cot_steps/total_steps*100:.1f}%)")
    print(f"   RAG steps:        {total_rag_steps} ({total_rag_steps/total_steps*100:.1f}%)")
    print(f"   Avg steps/traj:   {total_steps/total:.1f}")

    print(f"\n🏷️  LABEL DISTRIBUTION (RPE)")
    total_labeled = sum(label_counts.values())
    for label, count in sorted(label_counts.items()):
        print(f"   {label:12s}: {count:5d} ({count/total_labeled*100:.1f}%)")

    print("\n" + "=" * 70)

    # Additional insights
    print(f"\n💡 KEY INSIGHTS")

    # Insight 1: RAG performance
    if rag_correct < with_rag * 0.5:
        print(f"   ⚠️  RAG accuracy ({rag_correct/with_rag*100:.1f}%) is lower than overall")
        print(f"      → RAG may need better retrieval or query formulation")

    # Insight 2: MC stuck
    if mc_stuck_at_zero > total * 0.3:
        print(f"   ⚠️  {mc_stuck_at_zero/total*100:.1f}% trajectories have MC=0.0")
        print(f"      → May indicate rollout issues or answer extraction problems")

    # Insight 3: CoT performance
    if cot_correct > cot_only * 0.9:
        print(f"   ✓ CoT-only accuracy is very high ({cot_correct/cot_only*100:.1f}%)")
        print(f"      → Model strong at reasoning without retrieval")

    # Insight 4: Label balance
    good_ratio = label_counts.get('good', 0) / total_labeled if total_labeled > 0 else 0
    if good_ratio > 0.9 or good_ratio < 0.1:
        print(f"   ⚠️  Label distribution is imbalanced (good: {good_ratio*100:.1f}%)")
        print(f"      → May need to adjust RPE threshold")

    print("\n" + "=" * 70)

    # Export quick summary
    summary_file = jsonl_path.replace('.jsonl', '_quick_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Trajectories: {total}\n")
        f.write(f"Accuracy: {correct/total*100:.1f}%\n")
        f.write(f"With RAG: {with_rag} ({rag_correct/with_rag*100:.1f}% acc)\n")
        f.write(f"CoT Only: {cot_only} ({cot_correct/cot_only*100:.1f}% acc)\n")
        f.write(f"MC stuck: {mc_stuck_at_zero} ({mc_stuck_at_zero/total*100:.1f}%)\n")
        f.write(f"Total steps: {total_steps} (CoT: {total_cot_steps}, RAG: {total_rag_steps})\n")
        f.write(f"\nLabels:\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"  {label}: {count} ({count/total_labeled*100:.1f}%)\n")

    print(f"\n✓ Quick summary saved to: {summary_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_stats.py <jsonl_file>")
        sys.exit(1)

    jsonl_path = sys.argv[1]

    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)

    analyze_trajectories(jsonl_path)


if __name__ == "__main__":
    main()
