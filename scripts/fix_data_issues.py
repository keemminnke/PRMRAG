#!/usr/bin/env python3
"""Fix the 3 major issues in trajectory data."""

import json
import re
import sys
from pathlib import Path

def has_citations(content):
    """Check if content has [N] citations."""
    return bool(re.search(r'\[([1-5])\]', content))

def fix_misclassified_search(step):
    """Fix: RAG step wrongly classified as Reason."""
    if step['action'] == 'Reason':
        content = step['content']
        
        # Has "no relevant" + citations → was actually a search!
        if 'no relevant information' in content.lower():
            if has_citations(content):
                print(f"  Fixing misclassified search: Step {step['step_num']}")
                step['action'] = 'Search'
                step['num_passages'] = 5  # Estimate based on [1-5]
                return True
    return False

def remove_hallucinated_observation(step):
    """Remove hallucinated Observation from CoT steps."""
    if step['action'] == 'Reason' and step.get('num_passages', 0) == 0:
        content = step['content']
        
        # Has Action: Search + Observation but no real search
        if 'Action: Search[' in content and 'Observation:' in content:
            # Extract only Thought
            thought_match = re.search(r'Thought:\s*(.+?)(?=\nAction:|$)', content, re.DOTALL)
            if thought_match:
                print(f"  Removing hallucinated observation: Step {step['step_num']}")
                step['content'] = f"Thought: {thought_match.group(1).strip()}"
                step['observation'] = None
                step['sub_answer'] = None
                return True
    return False

def fix_trajectory(traj):
    """Fix all issues in a trajectory."""
    fixed_count = 0
    
    for step in traj['steps']:
        # Fix 1: Misclassified searches
        if fix_misclassified_search(step):
            fixed_count += 1
        
        # Fix 2: Remove hallucinations
        elif remove_hallucinated_observation(step):
            fixed_count += 1
    
    return fixed_count

def main():
    input_file = Path('outputs/hybrid_1000q_from_0/results_hybrid_all_996.jsonl')
    output_file = Path('outputs/hybrid_1000q_from_0/results_hybrid_all_996_FIXED.jsonl')
    
    print("Fixing data issues...")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print()
    
    total_trajs = 0
    total_fixed = 0
    
    with open(input_file) as inf, open(output_file, 'w') as outf:
        for line in inf:
            traj = json.loads(line)
            total_trajs += 1
            
            fixed = fix_trajectory(traj)
            total_fixed += fixed
            
            outf.write(json.dumps(traj) + '\n')
            
            if total_trajs % 100 == 0:
                print(f"Processed {total_trajs} trajectories...")
    
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total trajectories: {total_trajs}")
    print(f"Total fixes applied: {total_fixed}")
    print(f"Output: {output_file}")
    print()
    print("Next: Verify the fixed data")

if __name__ == '__main__':
    main()
