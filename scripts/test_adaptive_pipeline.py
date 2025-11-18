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

    # Initialize BM25 retriever with supporting facts
    print("\n[2] Initializing BM25 retriever...")

    # Use a hard multi-hop question that requires external knowledge
    # Supporting facts corpus for the question
    supporting_corpus = [
        {
            "id": "doc1",
            "title": "Shirley Temple",
            "text": "Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, and diplomat. She was Hollywood's number one box-office draw as a child actress from 1934 to 1938. As an adult, she pursued a career in public service, serving as the United States Ambassador to Ghana and to Czechoslovakia, and as Chief of Protocol of the United States."
        },
        {
            "id": "doc2",
            "title": "Kiss and Tell (1945 film)",
            "text": "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer. In the film, Corliss is a mischievous teenager who becomes involved in a series of misunderstandings. The film was directed by Richard Wallace and based on the play by F. Hugh Herbert."
        },
        {
            "id": "doc3",
            "title": "Chief of Protocol",
            "text": "The Chief of Protocol is a U.S. government official responsible for advising the President, the Vice President, and the Secretary of State on matters of diplomatic protocol. Notable people who have held this position include Shirley Temple Black, who served from 1976 to 1977."
        },
        {
            "id": "doc4",
            "title": "Corliss Archer",
            "text": "Corliss Archer is a fictional character portrayed in various media. The character originated in short stories and was later adapted for radio, film, and television. Shirley Temple played Corliss Archer in the 1945 film Kiss and Tell."
        }
    ]

    retriever = BM25Retriever(corpus=supporting_corpus)
    print(f"✓ Retriever initialized with {len(supporting_corpus)} supporting documents")

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


    # Test question - Hard multi-hop question that requires RAG
    # This question requires:
    # 1. Identifying who played Corliss Archer in Kiss and Tell (Shirley Temple)
    # 2. Identifying what government position she held (Chief of Protocol)
    question = "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
    gold_answer = "Chief of Protocol"

    # Easy baseline question (for comparison):
    # question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    # gold_answer = "yes"

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
