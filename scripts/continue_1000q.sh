#!/bin/bash
# Continue generating remaining questions (590-999)

echo "=== Continuing 1000 Question Generation ==="
echo "Completed: 589 questions (0-588)"
echo "Remaining: 411 questions (589-999)"
echo ""

# Run from question 590 (index 589) to question 1000 (index 999)
# Dynamic K is automatically enabled in AdaptiveTrajectoryGenerator
python3 scripts/batch_test_hybrid.py \
    --num-questions 411 \
    --start-idx 589 \
    --num-rollouts 4 \
    --output-dir outputs/hybrid_1000q_from_0

echo ""
echo "✓ Generation complete!"
echo "Check: outputs/hybrid_1000q_from_0/results_hybrid_*.jsonl"
