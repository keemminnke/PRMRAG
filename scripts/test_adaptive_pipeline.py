#!/usr/bin/env python3
"""Test the full adaptive MC-CoT + RAG pipeline on a single question.

This script shows the complete workflow with detailed logging:
- MC estimation at each step
- RPE calculation
- CoT accept/reject decisions
- RAG intervention when needed
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator
from prmrag.retrieval import BM25Retriever


def main():
    print("=" * 70)
    print("TEST: Adaptive MC-CoT + RAG Pipeline")
    print("=" * 70)

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load model
    print("\n[1] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize BM25 retriever (placeholder for now)
    print("\n[2] Initializing BM25 retriever...")
    # For now, use dummy corpus
    dummy_corpus = [
        {"id": "doc1", "title": "Paris", "text": "Paris is the capital of France."},
        {"id": "doc2", "title": "France", "text": "France is a country in Western Europe."},
    ]
    retriever = BM25Retriever(corpus=dummy_corpus)
    print("✓ Retriever initialized")

    # Initialize adaptive generator
    print("\n[3] Initializing adaptive generator...")

    # Get generation config
    gen_config = config.get('generation', {})
    gen_config['num_rollouts'] = 5  # Start with 5 for speed
    gen_config['max_steps'] = 5

    generator = AdaptiveTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
        config=gen_config,
    )
    print("✓ Generator initialized")
    print(f"  - num_rollouts: {generator.num_rollouts}")


    # Test question - Use a harder multi-hop question from HotpotQA
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    gold_answer = "yes"

    # Alternative hard questions:
    # question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    # gold_answer = "Chief of Protocol"

    print("\n" + "=" * 70)
    print("GENERATING TRAJECTORY")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Gold Answer: {gold_answer}")
    print()

    # Enable detailed logging by monkey-patching
    original_mc_estimate = generator._monte_carlo_estimate

    def logged_mc_estimate(state, gold_answer):
        """Wrapper to log MC estimation."""
        mc_value = original_mc_estimate(state, gold_answer)

        # Show state info
        num_steps = len(state.get('reasoning_history', []))
        print(f"    [MC Estimation] After {num_steps} step(s): MC = {mc_value:.3f}")

        return mc_value

    generator._monte_carlo_estimate = logged_mc_estimate

    # Generate trajectory
    print("\nStarting trajectory generation...\n")

    trajectory = generator.generate_trajectory(
        question=question,
        gold_answer=gold_answer,
        trajectory_id="test_001",
    )

    # Results
    print("\n" + "=" * 70)
    print("TRAJECTORY RESULT")
    print("=" * 70)

    if trajectory is None:
        print("❌ Trajectory generation FAILED")
        print("   (All steps rejected by MC estimation)")
    else:
        print("✅ Trajectory generation SUCCEEDED")
        print(f"\nFinal Answer: {trajectory.final_answer}")
        print(f"Is Correct: {trajectory.is_correct}")

        print(f"\nNumber of Steps: {len(trajectory.steps)}")
        print(f"  - CoT steps: {trajectory.metadata['num_cot_steps']}")
        print(f"  - RAG steps: {trajectory.metadata['num_rag_steps']}")

        print("\nStep-by-step breakdown:")
        print("-" * 70)
        for i, step in enumerate(trajectory.steps, 1):
            print(f"\nStep {i} ({step.step_type.value.upper()}):")
            print(f"  Content: {step.content}")
            print(f"  MC Before: {step.mc_before:.3f}")
            print(f"  MC After:  {step.mc_after:.3f}")
            print(f"  RPE:       {step.rpe:.3f}")
            print(f"  Label:     {step.label if step.label else 'None'} {'✓ Good step!' if step.label == 'good' else ''}")
            print(f"  Threshold: {step.metadata.get('threshold', 'N/A')}")
            print(f"  Decision:  {step.metadata.get('accepted', 'N/A')}")

            if step.used_passages:
                print(f"  Retrieved: {len(step.used_passages)} passages")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
