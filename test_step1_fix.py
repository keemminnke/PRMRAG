#!/usr/bin/env python3
"""Test script to verify Step 1 MC pre-calculation fix."""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.prmrag.generation.adaptive_generator import AdaptiveGenerator
from src.prmrag.retrieval.hybrid_retriever import HybridRetriever
from src.prmrag.data.kilt_loader import load_kilt_dataset

def test_step1_labeling():
    """Test that Step 1 now gets proper RPE calculation."""

    print("=" * 80)
    print("Testing Step 1 MC Pre-calculation Fix")
    print("=" * 80)

    # Load test question
    print("\n[1/4] Loading test question from KILT dataset...")
    questions = load_kilt_dataset(
        dataset_name='hotpotqa',
        split='dev',
        start_idx=112,
        end_idx=113
    )

    if not questions:
        print("❌ Failed to load test question")
        return

    test_q = questions[0]
    print(f"✓ Loaded question: {test_q['question'][:80]}...")
    print(f"  Gold answer: {test_q['gold_answer']}")

    # Initialize retriever
    print("\n[2/4] Initializing hybrid retriever...")
    retriever = HybridRetriever(
        index_dir="/root/.local/PRMRAG/data/kilt_wikipedia_index_bm25_bge_multi",
        top_k_bm25=100,
        top_k_dense=100,
        top_k_final=5,
    )
    print("✓ Retriever initialized")

    # Initialize generator with small K for testing
    print("\n[3/4] Initializing adaptive generator...")
    generator = AdaptiveGenerator(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        retriever=retriever,
        num_rollouts=4,  # Small K for quick test
        threshold=0.79,
        temperature=0.8,
        max_steps=3,
        use_step_forcing=True,
        use_dynamic_k=False,  # Disable for predictable K
    )
    print("✓ Generator initialized")

    # Generate trajectory
    print("\n[4/4] Generating trajectory to test Step 1 labeling...")
    print("-" * 80)

    trajectory = generator.generate_trajectory(
        question=test_q['question'],
        gold_answer=test_q['gold_answer'],
        trajectory_id="test_step1_fix"
    )

    print("-" * 80)

    if not trajectory:
        print("❌ Trajectory generation failed")
        return

    # Analyze Step 1
    print("\n" + "=" * 80)
    print("Step 1 Analysis")
    print("=" * 80)

    if len(trajectory.steps) == 0:
        print("❌ No steps generated")
        return

    step1 = trajectory.steps[0]

    print(f"\nStep 1 Details:")
    print(f"  mc_before:  {step1.mc_before:.3f}")
    print(f"  mc_after:   {step1.mc_after:.3f}")
    print(f"  rpe:        {step1.rpe:.3f}")
    print(f"  label:      {step1.label}")
    print(f"  metadata:   {step1.metadata}")

    # Verify fix
    print("\n" + "=" * 80)
    print("Verification")
    print("=" * 80)

    success = True

    # Check 1: mc_before should not be 0.0
    if step1.mc_before == 0.0:
        print("❌ FAIL: mc_before is still 0.0 (should be mc_question)")
        success = False
    else:
        print(f"✓ PASS: mc_before = {step1.mc_before:.3f} (not 0.0)")

    # Check 2: RPE should be calculated (not dummy 1.0)
    if step1.rpe == 1.0 and step1.mc_before != step1.mc_after:
        print("❌ FAIL: RPE is dummy 1.0 (should be calculated)")
        success = False
    else:
        print(f"✓ PASS: RPE = {step1.rpe:.3f} (calculated from MC values)")

    # Check 3: Label should be based on RPE threshold
    expected_label = 'good' if step1.rpe >= 0.79 else 'bad'
    if step1.label != expected_label:
        print(f"❌ FAIL: label is '{step1.label}' but expected '{expected_label}' based on RPE={step1.rpe:.3f}")
        success = False
    else:
        print(f"✓ PASS: label = '{step1.label}' (correctly determined by RPE >= 0.79)")

    # Check 4: mc_question should be in metadata
    if 'mc_question' not in step1.metadata:
        print("❌ FAIL: mc_question not found in metadata")
        success = False
    else:
        mc_q = step1.metadata['mc_question']
        print(f"✓ PASS: mc_question stored in metadata = {mc_q:.3f}")

        # Verify mc_question == mc_before
        if abs(mc_q - step1.mc_before) > 0.001:
            print(f"❌ FAIL: mc_question ({mc_q:.3f}) != mc_before ({step1.mc_before:.3f})")
            success = False
        else:
            print(f"✓ PASS: mc_question == mc_before ({mc_q:.3f})")

    print("\n" + "=" * 80)
    if success:
        print("✅ All checks PASSED - Step 1 fix working correctly!")
    else:
        print("❌ Some checks FAILED - please review")
    print("=" * 80)

if __name__ == "__main__":
    test_step1_labeling()
