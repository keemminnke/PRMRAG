#!/bin/bash
# Hybrid vs Dense-only Comparison Test
# Runs both tests sequentially with updated citation prompt

set -e  # Exit on error

echo "=========================================="
echo "Hybrid vs Dense-only Comparison Test"
echo "=========================================="
echo ""
echo "Updated prompt with improved citation guidance"
echo "Testing 10 questions with 8 MC rollouts"
echo "Full KILT corpus (5.9M documents)"
echo ""

# Test parameters
NUM_QUESTIONS=10
START_IDX=0
NUM_ROLLOUTS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output directories
HYBRID_DIR="outputs/hybrid_citation_${TIMESTAMP}"
DENSE_DIR="outputs/dense_citation_${TIMESTAMP}"

echo "Output directories:"
echo "  Hybrid: $HYBRID_DIR"
echo "  Dense:  $DENSE_DIR"
echo ""

# ==========================================
# Test 1: Hybrid Retrieval (BM25 + BGE-M3)
# ==========================================

echo "=========================================="
echo "[1/2] Running Hybrid Retrieval Test"
echo "=========================================="
echo "Config: BM25 (k=50) + BGE-M3 (k=50), RRF fusion"
echo "Documents per RAG step: 5"
echo ""

python3 scripts/batch_test_hybrid.py \
    --num-questions $NUM_QUESTIONS \
    --start-idx $START_IDX \
    --fusion-method rrf \
    --k-sparse 50 \
    --k-dense 50 \
    --num-rollouts $NUM_ROLLOUTS \
    --output-dir $HYBRID_DIR \
    2>&1 | tee /tmp/hybrid_citation_test.log

echo ""
echo "✓ Hybrid test completed"
echo ""

# ==========================================
# Test 2: Dense-only Retrieval (BGE-M3)
# ==========================================

echo "=========================================="
echo "[2/2] Running Dense-only Retrieval Test"
echo "=========================================="
echo "Config: BGE-M3 only (k=50), no BM25"
echo "Documents per RAG step: 5"
echo ""

python3 scripts/batch_test_hybrid.py \
    --num-questions $NUM_QUESTIONS \
    --start-idx $START_IDX \
    --fusion-method rrf \
    --k-sparse 0 \
    --k-dense 50 \
    --num-rollouts $NUM_ROLLOUTS \
    --output-dir $DENSE_DIR \
    2>&1 | tee /tmp/dense_citation_test.log

echo ""
echo "✓ Dense-only test completed"
echo ""

# ==========================================
# Summary
# ==========================================

echo "=========================================="
echo "All Tests Completed!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  Hybrid:     $HYBRID_DIR"
echo "  Dense-only: $DENSE_DIR"
echo ""
echo "Logs:"
echo "  Hybrid:     /tmp/hybrid_citation_test.log"
echo "  Dense-only: /tmp/dense_citation_test.log"
echo ""
echo "Next steps:"
echo "  1. Compare accuracy between hybrid and dense-only"
echo "  2. Analyze citation usage improvements"
echo "  3. Check RPE distribution"
echo ""
