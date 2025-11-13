#!/usr/bin/env python3
"""
Analyze consensus agreement between RPE and Judge labels.

This script provides detailed analysis of:
- Agreement rates
- Disagreement patterns
- Label distributions
- Confidence distributions
"""

import argparse
import sys
from pathlib import Path
import json
import jsonlines
from collections import Counter, defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.data.schemas import LabeledTrajectory


def analyze_consensus(labeled_trajectories_path: Path, output_path: Path):
    """Analyze consensus from labeled trajectories."""

    print("Loading labeled trajectories...")
    labeled_trajectories = []

    with jsonlines.open(labeled_trajectories_path) as reader:
        for obj in reader:
            # Simplified loading (would use proper from_dict in production)
            labeled_trajectories.append(obj)

    print(f"Loaded {len(labeled_trajectories)} labeled trajectories\n")

    # Analysis
    total_trajectories = len(labeled_trajectories)
    filtered_trajectories = sum(1 for lt in labeled_trajectories if lt.get("is_filtered", False))
    kept_trajectories = total_trajectories - filtered_trajectories

    # Step-level analysis
    total_steps = 0
    positive_consensus = 0
    negative_consensus = 0
    disagreements = 0

    rpe_label_dist = Counter()
    judge_label_dist = Counter()
    consensus_label_dist = Counter()

    disagreement_patterns = defaultdict(int)

    for lt in labeled_trajectories:
        consensus_labels = lt.get("consensus_labels", [])

        for cons in consensus_labels:
            total_steps += 1

            rpe_label = cons.get("rpe_label")
            judge_label = cons.get("judge_label")
            consensus_label = cons.get("consensus_label")

            rpe_label_dist[rpe_label] += 1
            judge_label_dist[judge_label] += 1
            consensus_label_dist[consensus_label] += 1

            if consensus_label == 1:
                positive_consensus += 1
            elif consensus_label == 0:
                negative_consensus += 1
            else:
                disagreements += 1
                pattern = f"RPE={rpe_label}, Judge={judge_label}"
                disagreement_patterns[pattern] += 1

    # Compute metrics
    agreement_rate = (positive_consensus + negative_consensus) / total_steps if total_steps > 0 else 0
    disagreement_rate = disagreements / total_steps if total_steps > 0 else 0

    # Build analysis report
    analysis = {
        "trajectory_level": {
            "total": total_trajectories,
            "kept": kept_trajectories,
            "filtered": filtered_trajectories,
            "keep_rate": kept_trajectories / total_trajectories if total_trajectories > 0 else 0,
        },
        "step_level": {
            "total_steps": total_steps,
            "positive_consensus": positive_consensus,
            "negative_consensus": negative_consensus,
            "disagreements": disagreements,
            "agreement_rate": agreement_rate,
            "disagreement_rate": disagreement_rate,
        },
        "label_distributions": {
            "rpe": dict(rpe_label_dist),
            "judge": dict(judge_label_dist),
            "consensus": dict(consensus_label_dist),
        },
        "disagreement_patterns": dict(disagreement_patterns),
    }

    # Print report
    print("=" * 60)
    print("CONSENSUS ANALYSIS REPORT")
    print("=" * 60)

    print("\n--- Trajectory Level ---")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Kept: {kept_trajectories} ({kept_trajectories/total_trajectories*100:.1f}%)")
    print(f"Filtered: {filtered_trajectories} ({filtered_trajectories/total_trajectories*100:.1f}%)")

    print("\n--- Step Level ---")
    print(f"Total steps: {total_steps}")
    print(f"Positive consensus (label=1): {positive_consensus} ({positive_consensus/total_steps*100:.1f}%)")
    print(f"Negative consensus (label=0): {negative_consensus} ({negative_consensus/total_steps*100:.1f}%)")
    print(f"Disagreements (filtered): {disagreements} ({disagreements/total_steps*100:.1f}%)")
    print(f"\nAgreement rate: {agreement_rate:.1%}")

    print("\n--- RPE Label Distribution ---")
    for label, count in sorted(rpe_label_dist.items()):
        pct = count / total_steps * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\n--- Judge Label Distribution ---")
    for label, count in sorted(judge_label_dist.items()):
        pct = count / total_steps * 100
        print(f"  {label}: {count} ({pct:.1f}%)")

    print("\n--- Disagreement Patterns ---")
    for pattern, count in sorted(disagreement_patterns.items(), key=lambda x: x[1], reverse=True):
        pct = count / disagreements * 100 if disagreements > 0 else 0
        print(f"  {pattern}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)

    # Save to JSON
    print(f"\nSaving analysis to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze consensus agreement between RPE and Judge labels"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to labeled trajectories (output from label_dataset.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save analysis JSON",
    )

    args = parser.parse_args()

    analyze_consensus(args.input, args.output)


if __name__ == "__main__":
    main()
