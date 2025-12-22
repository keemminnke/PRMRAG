#!/usr/bin/env python3
"""Judge labeling using QwQ-32B model."""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from prmrag.labeling.judge_labeler import JudgeLabeler
from prmrag.data.schemas import Trajectory, TrajectoryStep

def load_trajectories(filepath: Path):
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Convert to Trajectory schema
            steps = []
            for step_data in data['steps']:
                # Determine action_type
                if step_data.get('num_passages', 0) > 0:
                    action_type = 'retrieve'
                elif step_data.get('action') == 'Finish':
                    action_type = 'answer'
                else:
                    action_type = 'reason'

                # Build full action with thought
                thought = step_data.get('thought', '')
                action_name = step_data.get('action', 'Unknown')

                if thought:
                    full_action = f"Thought: {thought}\nAction: {action_name}"
                else:
                    full_action = f"Action: {action_name}"

                step = TrajectoryStep(
                    step_id=step_data['step_num'] - 1,  # 0-indexed
                    action_type=action_type,
                    action=full_action,
                    observation=step_data.get('observation', ''),
                    passages=[step_data.get('observation', '')] if step_data.get('observation') else [],
                )
                steps.append(step)
            
            trajectory = Trajectory(
                trajectory_id=data['question_id'],
                question=data['question'],
                gold_answer=data['gold_answer'],
                final_answer=data.get('predicted_answer', ''),
                steps=steps,
                supporting_facts=[],  # Not available in our data
            )
            trajectories.append((trajectory, data))  # Keep original data too
    
    return trajectories

def main():
    input_file = Path('/root/.local/PRMRAG/outputs/test_consensus_5q.jsonl')
    output_file = Path('/root/.local/PRMRAG/outputs/test_consensus_5q_judge_labels.jsonl')
    
    print("=" * 70)
    print("JUDGE LABELING with QwQ-32B")
    print("=" * 70)
    print()
    
    # Load trajectories
    print(f"Loading trajectories from: {input_file}")
    trajectory_pairs = load_trajectories(input_file)
    print(f"✓ Loaded {len(trajectory_pairs)} trajectories")
    print()
    
    # Initialize Judge labeler with QwQ-32B
    # QwQ-32B needs ~61GB for weights + ~30GB for KV cache
    config = {
        'model_name': 'Qwen/QwQ-32B',  # Use official release (not Preview)
        'temperature': 0.3,
        'max_tokens': 3072,  # Increased to allow QwQ's long reasoning to complete
        'gpu_memory_utilization': 0.95,  # Use 95% of 96GB = 91.2GB
        'tensor_parallel_size': 1,
        'prompt_style': 'versaprm',
        'use_gold_answer': True,
        'use_supporting_facts': False,
    }
    
    # Label each trajectory
    results = []
    for i, (trajectory, original_data) in enumerate(trajectory_pairs, 1):
        print(f"[{i}/{len(trajectory_pairs)}] Labeling: {trajectory.question[:60]}...")

        # Reinitialize model for each question (workaround for vLLM crash)
        print(f"  Initializing QwQ-32B...")
        try:
            judge_labeler = JudgeLabeler(config)
            judge_labels = []

            # Label each step
            for step_idx in range(len(trajectory.steps)):
                try:
                    judge_label = judge_labeler._judge_step(trajectory, step_idx)
                    judge_labels.append(judge_label)
                except Exception as e:
                    print(f"  ERROR labeling step {step_idx + 1}: {e}")
                    # Create dummy label on error
                    from prmrag.labeling.schemas import JudgeLabel
                    judge_labels.append(JudgeLabel(
                        step_id=step_idx,
                        label='UNKNOWN',
                        reasoning=f'Error: {str(e)}',
                        confidence=0.0,
                    ))

            # Cleanup vLLM resources
            del judge_labeler
            import gc
            gc.collect()

        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            continue
        
        # Prepare output
        output_data = {
            'question_id': trajectory.trajectory_id,
            'question': trajectory.question,
            'gold_answer': trajectory.gold_answer,
            'steps': []
        }
        
        for step_idx, (step, judge_label) in enumerate(zip(original_data['steps'], judge_labels)):
            step_output = {
                'step_num': step['step_num'],
                'rpe_label': step.get('label', 'unknown'),
                'judge_label': judge_label.label,
                'judge_reasoning': judge_label.reasoning,
                'judge_confidence': judge_label.confidence,
                'mc_before': step.get('mc_before'),
                'mc_after': step.get('mc_after'),
                'rpe': step.get('rpe'),
                'action': step.get('action'),
                'num_passages': step.get('num_passages', 0),
            }
            output_data['steps'].append(step_output)
            
            # Print summary
            consensus = '✓' if step_output['rpe_label'] == judge_label.label.lower() else '✗'
            print(f"  Step {step['step_num']}: RPE={step_output['rpe_label']}, Judge={judge_label.label} {consensus}")
        
        results.append(output_data)
        print()
    
    # Save results
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    # Compute consensus statistics
    total_steps = sum(len(r['steps']) for r in results)
    consensus_count = sum(
        1 for r in results 
        for step in r['steps']
        if step['rpe_label'] == step['judge_label'].lower()
    )
    
    print(f"Total steps evaluated: {total_steps}")
    print(f"Consensus (RPE=Judge): {consensus_count}/{total_steps} ({consensus_count/total_steps*100:.1f}%)")
    print(f"Disagreement: {total_steps - consensus_count}/{total_steps} ({(total_steps-consensus_count)/total_steps*100:.1f}%)")
    print()
    print(f"✓ Output saved to: {output_file}")
    print()

if __name__ == '__main__':
    main()
