#!/usr/bin/env python3
"""Validate trajectory data for hallucinations and misclassifications."""

import json
import sys
import re
from pathlib import Path

def has_citations(content):
    """Check if content has [N] citations."""
    return bool(re.search(r'\[([1-5])\]', content))

def validate_trajectory(traj, traj_num):
    """Validate a single trajectory."""
    issues = []

    for step in traj['steps']:
        step_num = step['step_num']
        action = step['action']
        content = step['content']
        num_passages = step.get('num_passages', None)

        # Issue 1: Missing num_passages
        if num_passages is None:
            issues.append(f"Traj {traj_num}, Step {step_num}: Missing num_passages")

        # Issue 2: Hallucinated search (Reason with Observation)
        if action == 'Reason' and num_passages == 0:
            if 'Observation:' in content:
                # Check if it's real search (has citations) or hallucination
                if has_citations(content):
                    if 'no relevant information' in content.lower():
                        issues.append(f"Traj {traj_num}, Step {step_num}: Misclassified search (has [N], should be action='Search')")
                    else:
                        issues.append(f"Traj {traj_num}, Step {step_num}: Hallucinated search with citations")
                else:
                    issues.append(f"Traj {traj_num}, Step {step_num}: Hallucinated search without citations")

        # Issue 3: Inconsistent classification
        if action == 'Search' and num_passages == 0:
            issues.append(f"Traj {traj_num}, Step {step_num}: action='Search' but num_passages=0")

        if action == 'Reason' and num_passages and num_passages > 0:
            issues.append(f"Traj {traj_num}, Step {step_num}: action='Reason' but num_passages={num_passages}")

    return issues

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_trajectories.py <jsonl_file>")
        sys.exit(1)

    jsonl_file = Path(sys.argv[1])

    if not jsonl_file.exists():
        print(f"Error: File not found: {jsonl_file}")
        sys.exit(1)

    print(f"Validating {jsonl_file}...")
    print()

    total_trajs = 0
    total_steps = 0
    total_issues = 0
    issue_types = {
        'missing_num_passages': 0,
        'hallucinated_search': 0,
        'misclassified_search': 0,
        'inconsistent': 0,
    }

    with open(jsonl_file) as f:
        for line_num, line in enumerate(f, 1):
            traj = json.loads(line)
            total_trajs += 1
            total_steps += len(traj['steps'])

            issues = validate_trajectory(traj, line_num)
            total_issues += len(issues)

            for issue in issues:
                if 'Missing num_passages' in issue:
                    issue_types['missing_num_passages'] += 1
                elif 'Misclassified search' in issue:
                    issue_types['misclassified_search'] += 1
                elif 'Hallucinated' in issue:
                    issue_types['hallucinated_search'] += 1
                else:
                    issue_types['inconsistent'] += 1

                # Print first 10 issues
                if total_issues <= 10:
                    print(f"  {issue}")

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    print(f"Total trajectories: {total_trajs}")
    print(f"Total steps: {total_steps}")
    print()

    if total_issues == 0:
        print("✅ NO ISSUES FOUND!")
        print("   - No hallucinations")
        print("   - No misclassifications")
        print("   - All steps have num_passages")
        print()
        print("🎉 Data is clean and ready for Step 2!")
    else:
        print(f"⚠️  FOUND {total_issues} ISSUES:")
        print()
        for issue_type, count in issue_types.items():
            if count > 0:
                pct = count / total_steps * 100
                print(f"  {issue_type:25s}: {count:4d} ({pct:.1f}%)")
        print()
        print("❌ Data needs fixing before Step 2")

    print("=" * 70)

    return 0 if total_issues == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
