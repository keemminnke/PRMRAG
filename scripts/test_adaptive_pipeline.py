#!/usr/bin/env python3
"""Test the full adaptive MC-CoT + RAG pipeline on a single question.

This script shows the complete workflow with detailed logging:
- MC estimation at each step
- RPE calculation
- CoT accept/reject decisions
- RAG intervention when needed
"""

import sys
import json
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.models import load_policy_model
from prmrag.utils import load_config
from prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator
from prmrag.retrieval import BM25Retriever


def load_example_data(data_dir: Path):
    """Load example questions and corpus from data files."""
    questions_file = data_dir / "raw" / "example_hotpotqa_questions.jsonl"
    corpus_file = data_dir / "raw" / "example_hotpotqa_corpus.jsonl"

    # Load questions
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    # Load corpus
    corpus = []
    with open(corpus_file, 'r') as f:
        for line in f:
            corpus.append(json.loads(line))

    return questions, corpus


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test adaptive MC-CoT + RAG pipeline with real data"
    )
    parser.add_argument(
        "--question-idx",
        type=int,
        default=0,
        help="Index of the question to test (default: 0)",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=5,
        help="Number of MC rollouts (default: 5)",
    )
    parser.add_argument(
        "--list-questions",
        action="store_true",
        help="List all available questions and exit",
    )
    args = parser.parse_args()

    # If user wants to list questions, do that and exit
    if args.list_questions:
        data_dir = Path(__file__).parent.parent / "data"
        questions, _ = load_example_data(data_dir)
        print(f"\n{'='*70}")
        print(f"Available Questions ({len(questions)} total)")
        print(f"{'='*70}\n")
        for idx, q in enumerate(questions):
            print(f"[{idx}] {q['_id']}")
            print(f"    Q: {q['question']}")
            print(f"    A: {q['answer']}")
            print()
        return

    print("=" * 70)
    print("TEST: Adaptive MC-CoT + RAG Pipeline")
    print("=" * 70)

    # Load config
    config_path = Path("configs/adaptive_generation.yaml")
    config = load_config(config_path)

    # Load example data
    print("\n[1] Loading example data from files...")
    data_dir = Path(__file__).parent.parent / "data"
    questions, corpus = load_example_data(data_dir)
    print(f"✓ Loaded {len(questions)} questions and {len(corpus)} corpus documents")

    # Load model
    print("\n[2] Loading Qwen2.5-7B model...")
    policy_model = load_policy_model(config['policy_model'])
    print("✓ Model loaded")

    # Initialize BM25 retriever with full corpus
    print("\n[3] Initializing BM25 retriever...")
    retriever = BM25Retriever(corpus=corpus)
    print(f"✓ Retriever initialized with {len(corpus)} supporting documents")

    # Initialize adaptive generator
    print("\n[4] Initializing adaptive generator...")

    # Get generation config from 'adaptive' section
    gen_config = config.get('adaptive', {})
    gen_config['num_rollouts'] = args.num_rollouts
    # max_steps는 config 파일의 값(15) 사용 - safety limit으로만 작동
    # 실제로는 모델이 "Final Answer:"를 생성할 때까지 계속 생성함

    generator = AdaptiveTrajectoryGenerator(
        policy_model=policy_model,
        retriever=retriever,
        config=gen_config,
    )
    print("✓ Generator initialized")
    print(f"  - num_rollouts: {generator.num_rollouts}")

    # Select a test question from the loaded data
    test_idx = args.question_idx
    if test_idx >= len(questions):
        print(f"\n❌ Error: Question index {test_idx} out of range (0-{len(questions)-1})")
        print(f"   Available questions: {len(questions)}")
        return

    test_question = questions[test_idx]

    question = test_question['question']
    gold_answer = test_question['answer']
    question_id = test_question['_id']

    print(f"\n[5] Selected test question:")
    print(f"  ID: {question_id}")
    print(f"  Question: {question}")
    print(f"  Answer: {gold_answer}")

    print("\n" + "=" * 70)
    print("GENERATING TRAJECTORY")
    print("=" * 70)

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
        trajectory_id=question_id,
    )

    # Results
    print("\n" + "=" * 70)
    print("TRAJECTORY RESULT")
    print("=" * 70)

    if trajectory is None:
        print(" Trajectory generation FAILED")
        print("   (All steps rejected by MC estimation)")
    else:
        print(" Trajectory generation SUCCEEDED")
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
