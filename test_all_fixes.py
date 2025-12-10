#!/usr/bin/env python3
"""Test all recent fixes: Step 1 MC, Dynamic K, CoT parsing."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.prmrag.generation.adaptive_generator import AdaptiveTrajectoryGenerator
from src.prmrag.retrieval.hybrid_retriever import HybridRetriever
from src.prmrag.data.loaders import load_kilt_dataset

def test_all_fixes():
    """Test that all fixes are working correctly."""

    print("=" * 80)
    print("Testing All Recent Fixes")
    print("=" * 80)

    # Load 2 test questions
    print("\n[1/5] Loading 2 test questions from KILT dataset...")
    questions = load_kilt_dataset(
        dataset_name='hotpotqa',
        split='dev',
        start_idx=600,
        end_idx=602
    )

    if len(questions) < 2:
        print("❌ Failed to load test questions")
        return

    print(f"✓ Loaded {len(questions)} questions")

    # Initialize retriever
    print("\n[2/5] Initializing hybrid retriever...")
    retriever = HybridRetriever(
        index_dir="/root/.local/PRMRAG/data/kilt_wikipedia_index_bm25_bge_multi",
        top_k_bm25=100,
        top_k_dense=100,
        top_k_final=5,
    )
    print("✓ Retriever initialized")

    # Initialize generator with small K for testing
    print("\n[3/5] Initializing adaptive generator...")
    generator = AdaptiveTrajectoryGenerator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        retriever=retriever,
        num_rollouts=4,  # Small K for quick test
        threshold=0.79,
        temperature=0.8,
        max_steps=3,
        use_step_forcing=True,
        use_dynamic_k=True,  # Test dynamic K
    )
    print("✓ Generator initialized")

    # Test both questions
    print("\n[4/5] Generating trajectories...")
    print("-" * 80)

    all_pass = True

    for i, test_q in enumerate(questions, 1):
        print(f"\n--- Question {i}/2 ---")
        print(f"Q: {test_q['question'][:80]}...")

        trajectory = generator.generate_trajectory(
            question=test_q['question'],
            gold_answer=test_q['gold_answer'],
            trajectory_id=f"test_q{i}"
        )

        if not trajectory:
            print(f"❌ Question {i}: Trajectory generation failed")
            all_pass = False
            continue

        # Check Step 1
        if len(trajectory.steps) == 0:
            print(f"❌ Question {i}: No steps generated")
            all_pass = False
            continue

        step1 = trajectory.steps[0]

        print(f"\n✓ Generated {len(trajectory.steps)} steps")
        print(f"  Final answer: {trajectory.final_answer[:50]}...")
        print(f"  Correct: {trajectory.is_correct}")

        # Verify Fix 1: Step 1 MC pre-calculation
        print(f"\n[Fix 1] Step 1 MC Pre-calculation:")
        print(f"  mc_before:  {step1.mc_before:.3f}")
        print(f"  mc_after:   {step1.mc_after:.3f}")
        print(f"  rpe:        {step1.rpe:.3f}")
        print(f"  label:      {step1.label}")

        if step1.mc_before == 0.0:
            print(f"  ❌ FAIL: mc_before is 0.0 (should be mc_question)")
            all_pass = False
        else:
            print(f"  ✓ PASS: mc_before is not 0.0")

        if step1.rpe == 1.0 and step1.mc_before != step1.mc_after:
            print(f"  ❌ FAIL: rpe is dummy 1.0")
            all_pass = False
        else:
            print(f"  ✓ PASS: rpe is calculated")

        # Verify Fix 2: Dynamic K based on mc_question
        if 'mc_question' in step1.metadata:
            mc_q = step1.metadata['mc_question']
            print(f"\n[Fix 2] Dynamic K based on question difficulty:")
            print(f"  mc_question: {mc_q:.3f}")
            print(f"  ✓ PASS: mc_question stored in metadata")
        else:
            print(f"\n[Fix 2] Dynamic K:")
            print(f"  ❌ FAIL: mc_question not in metadata")
            all_pass = False

        # Verify Fix 3: CoT parsing
        print(f"\n[Fix 3] CoT ReAct Format Parsing:")
        parsed_correctly = False

        for j, step in enumerate(trajectory.steps, 1):
            if step.step_type.value == "cot":
                print(f"  Step {j} (CoT):")
                print(f"    thought:     {'✓' if step.thought else '✗'} {step.thought[:50] if step.thought else 'None'}...")
                print(f"    action:      {step.action}")
                print(f"    observation: {'✓' if step.observation else '✗'} {step.observation[:50] if step.observation else 'None'}...")

                if step.thought is not None:
                    parsed_correctly = True
                    print(f"    ✓ CoT step is parsed")
                else:
                    print(f"    ⚠ CoT step not parsed (might not have Thought: marker)")
            elif step.step_type.value == "rag":
                print(f"  Step {j} (RAG): (already parsed before)")

        if parsed_correctly:
            print(f"  ✓ PASS: At least one CoT step is parsed")
        else:
            print(f"  ⚠ WARNING: No CoT steps were parsed (check if steps have Thought: markers)")

    print("\n" + "=" * 80)
    if all_pass:
        print("✅ All critical checks PASSED")
    else:
        print("⚠ Some checks FAILED - review above")
    print("=" * 80)

if __name__ == "__main__":
    test_all_fixes()
