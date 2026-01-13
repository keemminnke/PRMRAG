#!/usr/bin/env python3
"""Consensus filtering using RPE + Judge labels.

This script compares RPE labels with Judge labels and filters
trajectories where both evaluators agree.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_labels(filepath: Path) -> Dict[str, Dict[int, str]]:
    """Load labels from JSONL file.

    Returns:
        Dict mapping question_id -> {step_num: label}
    """
    labels = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = data["question_id"]
            step_labels = {
                step["step_num"]: step.get("label") or step.get("rpe_label") or step.get("judge_label")
                for step in data["steps"]
            }
            labels[question_id] = step_labels
    return labels


def compare_labels(rpe_labels: Dict[str, Dict[int, str]],
                   judge_labels: Dict[str, Dict[int, str]]) -> Tuple[Dict, Dict]:
    """Compare RPE and Judge labels.

    Returns:
        (consensus_stats, step_agreements)
        - consensus_stats: {"both_good": N, "both_bad": N, "disagree": N}
        - step_agreements: {question_id: {step_num: {"rpe": "good", "judge": "bad", "agree": False}}}
    """
    stats = {
        "both_good": 0,
        "both_bad": 0,
        "disagree": 0,
        "total": 0,
    }

    agreements = {}

    for question_id in rpe_labels.keys():
        if question_id not in judge_labels:
            continue

        agreements[question_id] = {}
        rpe_steps = rpe_labels[question_id]
        judge_steps = judge_labels[question_id]

        for step_num in rpe_steps.keys():
            if step_num not in judge_steps:
                continue

            rpe_label = rpe_steps[step_num]
            judge_label = judge_steps[step_num]

            stats["total"] += 1

            if rpe_label == "good" and judge_label == "good":
                stats["both_good"] += 1
                agree = True
            elif rpe_label == "bad" and judge_label == "bad":
                stats["both_bad"] += 1
                agree = True
            else:
                stats["disagree"] += 1
                agree = False

            agreements[question_id][step_num] = {
                "rpe": rpe_label,
                "judge": judge_label,
                "agree": agree,
            }

    return stats, agreements


def filter_trajectories(full_data_path: Path,
                        agreements: Dict,
                        output_path: Path,
                        keep_mode: str = "agree"):
    """Filter trajectories based on consensus.

    Args:
        full_data_path: Path to full trajectory data
        agreements: Step agreement data from compare_labels
        output_path: Where to save filtered data
        keep_mode: "agree" (both agree) or "all" (keep all with consensus info)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_trajectories = 0
    kept_steps = 0
    filtered_trajectories = 0
    filtered_steps = 0

    with open(full_data_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            trajectory = json.loads(line)
            question_id = trajectory["question_id"]

            if question_id not in agreements:
                filtered_trajectories += 1
                filtered_steps += len(trajectory["steps"])
                continue

            # Filter steps based on consensus
            filtered_traj_steps = []
            for step in trajectory["steps"]:
                step_num = step["step_num"]

                if step_num not in agreements[question_id]:
                    if keep_mode == "all":
                        filtered_traj_steps.append(step)
                    continue

                agreement = agreements[question_id][step_num]

                if keep_mode == "agree" and not agreement["agree"]:
                    filtered_steps += 1
                    continue

                # Add consensus info to step
                step["consensus"] = {
                    "rpe_label": agreement["rpe"],
                    "judge_label": agreement["judge"],
                    "agree": agreement["agree"],
                }
                filtered_traj_steps.append(step)
                kept_steps += 1

            # Keep trajectory if it has at least one step
            if filtered_traj_steps:
                trajectory["steps"] = filtered_traj_steps
                trajectory["num_steps"] = len(filtered_traj_steps)
                outfile.write(json.dumps(trajectory) + '\n')
                kept_trajectories += 1
            else:
                filtered_trajectories += 1

    return {
        "kept_trajectories": kept_trajectories,
        "kept_steps": kept_steps,
        "filtered_trajectories": filtered_trajectories,
        "filtered_steps": filtered_steps,
    }


def main():
    parser = argparse.ArgumentParser(description='Consensus filtering with RPE + Judge')
    parser.add_argument('--full-data', type=Path, required=True,
                       help='Full trajectory data (JSONL)')
    parser.add_argument('--rpe-labels', type=Path, required=True,
                       help='RPE labels (from extract_rpe_labels.py)')
    parser.add_argument('--judge-labels', type=Path, required=True,
                       help='Judge labels (from GPT-OSS-120B)')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output filtered data')
    parser.add_argument('--keep-mode', choices=['agree', 'all'], default='agree',
                       help='Keep only agreed steps (agree) or all with info (all)')

    args = parser.parse_args()

    # Check inputs exist
    for path in [args.full_data, args.rpe_labels, args.judge_labels]:
        if not path.exists():
            print(f"Error: File not found: {path}")
            return 1

    print("=" * 60)
    print("CONSENSUS FILTERING")
    print("=" * 60)
    print()

    # Load labels
    print("Loading labels...")
    rpe_labels = load_labels(args.rpe_labels)
    judge_labels = load_labels(args.judge_labels)
    print(f"✓ RPE labels: {len(rpe_labels)} trajectories")
    print(f"✓ Judge labels: {len(judge_labels)} trajectories")
    print()

    # Compare labels
    print("Comparing RPE vs Judge...")
    stats, agreements = compare_labels(rpe_labels, judge_labels)

    total = stats["total"]
    print(f"Total steps compared: {total}")
    print()
    print("Consensus Statistics:")
    print(f"  Both Good:  {stats['both_good']:5d} ({stats['both_good']/total*100:5.1f}%)")
    print(f"  Both Bad:   {stats['both_bad']:5d} ({stats['both_bad']/total*100:5.1f}%)")
    print(f"  Disagree:   {stats['disagree']:5d} ({stats['disagree']/total*100:5.1f}%)")
    print()

    agreement_rate = (stats['both_good'] + stats['both_bad']) / total * 100
    print(f"Agreement Rate: {agreement_rate:.1f}%")
    print()

    # Filter trajectories
    print(f"Filtering trajectories (mode: {args.keep_mode})...")
    filter_stats = filter_trajectories(
        args.full_data,
        agreements,
        args.output,
        args.keep_mode
    )

    print()
    print("Filtering Results:")
    print(f"  Kept:     {filter_stats['kept_trajectories']} trajectories, "
          f"{filter_stats['kept_steps']} steps")
    print(f"  Filtered: {filter_stats['filtered_trajectories']} trajectories, "
          f"{filter_stats['filtered_steps']} steps")
    print()

    total_original_steps = filter_stats['kept_steps'] + filter_stats['filtered_steps']
    if total_original_steps > 0:
        kept_pct = filter_stats['kept_steps'] / total_original_steps * 100
        print(f"Kept {kept_pct:.1f}% of steps")

    print()
    print(f"✓ Output: {args.output}")
    print()
    print("=" * 60)

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
